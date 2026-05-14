#!/usr/bin/env python3
"""
Distributed NVFP4 quantization for HF-format models across a 2-node Ray cluster.

Architecture:
  Actor 0: embed_tokens + layers[0:split]
  Actor 1: layers[split:N] + norm + lm_head
  Driver: orchestrates calibration via Ray RPC, then merges per-actor exports.

Model-agnostic: works for any LlamaForCausalLM / MistralForCausalLM / similar
HF model whose decoder layers use a RMSNorm-style input_layernorm and a
RotaryEmbedding-style positional encoding. Architecture-specific classes
are introspected from the loaded model rather than imported by name.

Run:
  python3 distrib_quant.py --source-model /path/to/BF16 --output-dir /path/to/NVFP4

See --help for all options.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import ray
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors.torch import load_file, save_file
from datasets import load_dataset

import modelopt.torch.quantization as mtq
import modelopt.torch.export as mte


def parse_args():
    p = argparse.ArgumentParser(
        description="Distributed NVFP4 quantization across a 2-node Ray cluster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source-model", "-s", required=False,
                   help="Path to BF16 source model directory (HF format with safetensors). "
                        "Required for normal runs; optional in --resume-from-checkpoint mode "
                        "(falls back to the path stored in the checkpoint's meta.json).")
    p.add_argument("--output-dir", "-o", required=True,
                   help="Output directory for the NVFP4 model.")
    p.add_argument("--calib-size", type=int, default=256,
                   help="Number of calibration samples.")
    p.add_argument("--seq-len", type=int, default=2048,
                   help="Max sequence length per calibration sample.")
    p.add_argument("--dataset", default="cnn_dailymail",
                   help="HF dataset id for calibration text.")
    p.add_argument("--dataset-config", default="3.0.0",
                   help="Dataset config name (use empty string '' if dataset has no config).")
    p.add_argument("--temp-base", default="/tmp/_distrib_quant",
                   help="Base temp dir for per-actor exports. MUST be node-local (NOT NFS) to avoid rmtree race conditions.")
    p.add_argument("--split", type=int, default=None,
                   help="Layer index to split between actors. Default: num_layers // 2.")
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Skip Phases 1-5 and run only Phase 6+7 against an existing "
                        "checkpoint dir base. Expects {BASE}_ckpt_shard0 and {BASE}_ckpt_shard1 "
                        "to exist (created by a previous Phase-5.5). Saves ~25 min when "
                        "iterating on Phase-6 bugs. Use the same --temp-base that the prior "
                        "run used, or pass the base path directly.")
    return p.parse_args()


# ============================================================
# Actor: one half of the model
# ============================================================
@ray.remote(num_gpus=1)
class ModelShard:
    def __init__(self, model_path, layer_start, layer_end, has_embed, has_head, name):
        self.model_path = model_path
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.has_embed = has_embed
        self.has_head = has_head
        self.name = name
        self.device = "cuda:0"

    # ----- Phase 1: load model parts -----
    def load_model(self):
        config = AutoConfig.from_pretrained(self.model_path)
        self.hf_config = config
        self.num_layers_total = config.num_hidden_layers

        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(
                config, torch_dtype=torch.bfloat16
            )

        index_path = Path(self.model_path) / "model.safetensors.index.json"
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        # Decide which keys to load
        needed_keys = []
        for key in weight_map:
            if key.startswith("model.layers."):
                idx = int(key.split(".")[2])
                if self.layer_start <= idx < self.layer_end:
                    needed_keys.append(key)
            elif self.has_embed and key.startswith("model.embed_tokens."):
                needed_keys.append(key)
            elif self.has_head and (key.startswith("model.norm.") or key.startswith("lm_head.")):
                needed_keys.append(key)
            elif self.has_head and key.startswith("model.rotary_emb"):
                # some Llama variants put rotary_emb at model level
                needed_keys.append(key)

        # Group keys by shard
        by_shard = {}
        for key in needed_keys:
            by_shard.setdefault(weight_map[key], []).append(key)

        print(f"[{self.name}] loading {len(needed_keys)} tensors from {len(by_shard)} shards", flush=True)
        # Per-tensor streaming via safetensors.safe_open instead of load_file's
        # whole-shard cache. load_file kept the entire shard (~4 GB) in RAM until
        # all tensors were materialized, which on Behemoth-class models (~120 GB
        # already in UMA after partial load) pushes peak memory over the 119 GB
        # UMA budget and triggers a silent kernel-level OOM-kill on the Ray
        # worker. safe_open + get_tensor reads each tensor lazily, so the peak
        # extra RAM is one tensor at a time (max ~600 MB for Behemoth's
        # gate_proj/up_proj, well within budget).
        from safetensors import safe_open
        import gc
        for shard_file, keys in sorted(by_shard.items()):
            shard_path = str(Path(self.model_path) / shard_file)
            with safe_open(shard_path, framework="pt") as sf:
                shard_keys = set(sf.keys())
                for k in keys:
                    if k in shard_keys:
                        t = sf.get_tensor(k).to(torch.bfloat16)
                        set_module_tensor_to_device(
                            self.model, k, self.device, value=t,
                        )
                        del t
            gc.collect()

        # Local references
        self.embed_tokens = self.model.model.embed_tokens if self.has_embed else None
        self.local_layers = self.model.model.layers  # full list, but only [start:end] populated
        self.norm = self.model.model.norm if self.has_head else None
        self.lm_head = self.model.lm_head if self.has_head else None
        self.rotary_emb = getattr(self.model.model, "rotary_emb", None)

        # Architecture-specific helper classes — introspect from the model so the
        # driver works for Llama / Mistral / Qwen / any HF arch with RMSNorm-style
        # input_layernorm and a top-level rotary_emb. Storing the *class*, not an
        # instance, so Phase 6 can build cuda-resident copies on a 1-layer dummy.
        a_layer = self.local_layers[self.layer_start]
        self.rms_norm_cls = type(a_layer.input_layernorm)
        if self.rotary_emb is not None:
            self.rotary_emb_cls = type(self.rotary_emb)
        else:
            # Some architectures keep rotary on the attention module instead.
            self.rotary_emb_cls = type(getattr(a_layer.self_attn, "rotary_emb", None))

        torch.cuda.empty_cache()
        return f"{self.name}: loaded layers {self.layer_start}..{self.layer_end}, {torch.cuda.memory_allocated()/1e9:.1f} GiB on GPU"

    # ----- Phase 2: insert NVFP4 quantizers -----
    def apply_quantization(self):
        """Insert quantizers into our local layers (and lm_head if applicable).
        After this, quantizers are in calibration mode - forward passes will
        accumulate amax statistics."""

        # Build a wrapper module containing only our quantizable parts
        class LocalQuantizable(nn.Module):
            pass
        wrapper = LocalQuantizable()
        wrapper.layers = nn.ModuleList(
            [self.local_layers[i] for i in range(self.layer_start, self.layer_end)]
        )
        if self.has_head:
            wrapper.lm_head = self.lm_head

        # NVFP4 config. forward_loop=None means: insert quantizers, set calib mode, don't run anything.
        cfg = mtq.NVFP4_DEFAULT_CFG
        mtq.quantize(wrapper, cfg, forward_loop=None)

        self._quant_wrapper = wrapper  # keep ref to avoid GC

        # Count quantizers
        from modelopt.torch.quantization.nn import TensorQuantizer
        n_q = sum(1 for m in wrapper.modules() if isinstance(m, TensorQuantizer))
        return f"{self.name}: inserted {n_q} quantizers"

    # ----- Phase 3: forward passes (one for each actor role) -----
    def _causal_mask(self, seq_len, dtype):
        m = torch.full((seq_len, seq_len), float("-inf"), device=self.device, dtype=dtype)
        m = torch.triu(m, diagonal=1)
        return m[None, None, :, :]

    def forward_first(self, input_ids):
        """Actor 0: input_ids -> hidden_states"""
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

            hidden = self.embed_tokens(input_ids)
            
            # DEBUG: check first layer's actual state
            first_layer = self.local_layers[self.layer_start]
            attn = first_layer.self_attn
            print(f"[DEBUG {self.name}] hidden.shape={list(hidden.shape)}", flush=True)
            print(f"[DEBUG {self.name}] attn.head_dim={attn.head_dim}", flush=True)
            print(f"[DEBUG {self.name}] q_proj type={type(attn.q_proj).__name__}", flush=True)
            print(f"[DEBUG {self.name}] q_proj weight shape={list(attn.q_proj.weight.shape)}", flush=True)
            
            # Try running q_proj manually to see output shape
            q_out = attn.q_proj(hidden)
            print(f"[DEBUG {self.name}] q_proj(hidden).shape={list(q_out.shape)}", flush=True)
            
            # Build position_embeddings using the arch's rotary_emb (re-init on cuda to avoid meta)
            if not hasattr(self, '_rotary_initialized'):
                self.rotary_emb_cuda = self.rotary_emb_cls(self.hf_config).to(self.device)
                self._rotary_initialized = True
            position_embeddings = self.rotary_emb_cuda(hidden, position_ids)
            print(f"[DEBUG {self.name}] cos.shape={list(position_embeddings[0].shape)}", flush=True)
            
            attn_mask = self._causal_mask(seq_len, hidden.dtype)

            for i in range(self.layer_start, self.layer_end):
                hidden = self.local_layers[i](
                    hidden,
                    position_embeddings=position_embeddings,
                    attention_mask=attn_mask,
                )

            return hidden.cpu(), position_ids.cpu()

    def forward_second(self, hidden_states, position_ids):
        """Actor 1: hidden_states -> (logits not needed, just calib pass)"""
        with torch.no_grad():
            hidden = hidden_states.to(self.device)
            position_ids = position_ids.to(self.device)
            seq_len = hidden.shape[1]

            # Build position_embeddings on cuda using the arch's rotary class
            if not hasattr(self, '_rotary_initialized'):
                self.rotary_emb_cuda = self.rotary_emb_cls(self.hf_config).to(self.device)
                self._rotary_initialized = True
            position_embeddings = self.rotary_emb_cuda(hidden, position_ids)
            attn_mask = self._causal_mask(seq_len, hidden.dtype)

            for i in range(self.layer_start, self.layer_end):
                hidden = self.local_layers[i](
                    hidden,
                    position_embeddings=position_embeddings,
                    attention_mask=attn_mask,
                )

            hidden = self.norm(hidden)
            _ = self.lm_head(hidden)
            return True

    # ----- Phase 4: finalize amax -> scales -----
    def finalize(self):
        from modelopt.torch.quantization.nn import TensorQuantizer
        n = 0
        for m in self._quant_wrapper.modules():
            if isinstance(m, TensorQuantizer):
                if hasattr(m, "load_calib_amax"):
                    try:
                        m.load_calib_amax()
                    except Exception as e:
                        pass  # some quantizers may not have collected amax
                if hasattr(m, "disable_calib"):
                    m.disable_calib()
                if hasattr(m, "enable_quant"):
                    m.enable_quant()
                n += 1
        return f"{self.name}: finalized {n} quantizers"

    # ----- Phase 5: export (placeholder, prints stats) -----
    def export_status(self):
        """Diagnostic: print weight_scale stats for sanity check."""
        from modelopt.torch.quantization.nn import TensorQuantizer
        nan_count = 0
        zero_count = 0
        good_count = 0
        sample_values = []
        for name, m in self._quant_wrapper.named_modules():
            if isinstance(m, TensorQuantizer) and hasattr(m, "_amax"):
                amax = m._amax
                if amax is None:
                    continue
                if torch.isnan(amax).any():
                    nan_count += 1
                elif (amax == 0).all():
                    zero_count += 1
                else:
                    good_count += 1
                    if len(sample_values) < 3:
                        sample_values.append((name, amax.flatten()[0].item() if amax.numel() > 0 else 0))
        return f"{self.name}: amax_stats - good={good_count}, zero={zero_count}, nan={nan_count}, samples={sample_values}"

    # ----- Resume: rebuild minimal actor state from a Phase-5.5 checkpoint -----
    def setup_for_resume(self, ckpt_dir):
        """Skip Phases 1-5. Build just enough state for Phase 6 to run:
        - self.hf_config (for cfg_t construction)
        - self.rms_norm_cls / self.rotary_emb_cls (introspected from a meta model,
          no UMA hit because we use init_empty_weights)
        - self.local_layers as a list of Identity placeholders sized to layer_end
        - self.embed_tokens / norm / lm_head left None — Phase 6b loads them
          from `{ckpt_dir}/embed_tokens.pt` etc. on demand
        - self.model_path, layer_start, layer_end, has_embed, has_head, name
          come from {ckpt_dir}/meta.json
        """
        with open(f"{ckpt_dir}/meta.json") as f:
            meta = json.load(f)
        self.model_path = meta["model_path"]
        self.layer_start = meta["layer_start"]
        self.layer_end = meta["layer_end"]
        self.has_embed = meta["has_embed"]
        self.has_head = meta["has_head"]
        self.name = meta["name"]

        config = AutoConfig.from_pretrained(self.model_path)
        self.hf_config = config
        self.num_layers_total = config.num_hidden_layers

        # Introspect arch helper classes from a meta model — no allocations.
        with init_empty_weights():
            m = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
        a_layer = m.model.layers[0]
        self.rms_norm_cls = type(a_layer.input_layernorm)
        rotary = getattr(m.model, "rotary_emb", None) or getattr(a_layer.self_attn, "rotary_emb", None)
        self.rotary_emb_cls = type(rotary) if rotary is not None else None
        del m

        # Placeholder list — Phase 6 ckpt path doesn't read from self.local_layers,
        # but other code (export_status, etc.) might iterate it harmlessly.
        self.local_layers = [nn.Identity() for _ in range(self.num_layers_total)]
        self.embed_tokens = None
        self.norm = None
        self.lm_head = None

        return f"{self.name}: resumed from {ckpt_dir}, layers {self.layer_start}..{self.layer_end}"

    # ----- Phase 5.5: evict layers to disk so Phase 6 can stream them back -----
    def save_layers_to_disk(self, ckpt_dir):
        """Save each owned (quantized, calibrated) decoder layer to its own
        torch-pickle file, then replace the in-memory reference with nn.Identity()
        so Phase 6 can run with ~95 % of UMA freed.

        Each layer file contains the full module pickle including TensorQuantizer
        state (amax, scales), so Phase 6 can rebuild the per-layer 1-layer template
        from disk without re-doing calibration. The same files also serve as a
        resume checkpoint if Phase 6 crashes — re-running Phase 6 will pick up the
        per-layer state from disk.

        Drops the calibration wrapper too (frees ~1 GB) — must happen before
        export anyway, see note in export_shard.
        """
        import gc
        os.makedirs(ckpt_dir, exist_ok=True)

        if hasattr(self, "_quant_wrapper"):
            del self._quant_wrapper

        n = 0
        for i in range(self.layer_start, self.layer_end):
            path = f"{ckpt_dir}/layer_{i:04d}.pt"
            # Move to CPU before pickling to keep the on-disk artifact arch-neutral
            # (also halves I/O vs. pickling a cuda tensor and forcing a copy).
            torch.save(self.local_layers[i].cpu(), path)
            self.local_layers[i] = nn.Identity()
            n += 1

        # Also save embed/norm/lm_head so a resume entry point can complete
        # Phase 6b without needing Phases 1-5. Each is just a few GB on disk
        # but reclaims that UMA — important when next Phase actually starts.
        extras_saved = []
        if self.has_embed and self.embed_tokens is not None:
            torch.save(self.embed_tokens.cpu(), f"{ckpt_dir}/embed_tokens.pt")
            self.embed_tokens = None
            extras_saved.append("embed_tokens")
        if self.has_head and self.norm is not None:
            torch.save(self.norm.cpu(), f"{ckpt_dir}/norm.pt")
            self.norm = None
            extras_saved.append("norm")
        if self.has_head and self.lm_head is not None:
            torch.save(self.lm_head.cpu(), f"{ckpt_dir}/lm_head.pt")
            self.lm_head = None
            extras_saved.append("lm_head")

        # Persist a tiny meta file so a resume entry point can rebuild
        # actor state without consulting the driver.
        meta = {
            "model_path": self.model_path,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "has_embed": self.has_embed,
            "has_head": self.has_head,
            "name": self.name,
        }
        with open(f"{ckpt_dir}/meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        gc.collect()
        torch.cuda.empty_cache()
        mem = torch.cuda.memory_allocated() / 1e9
        return f"{self.name}: evicted {n} layers + {extras_saved} to {ckpt_dir}, GPU mem now {mem:.1f}G"

    # ----- Phase 6: per-actor export to local-NFS dir (per-layer streaming) -----
    def export_shard(self, output_dir, ckpt_dir=None):
        """Per-layer streaming export with tiny dummies (vocab=2) for the parts
        we don't own. The real embed/norm/lm_head are saved separately as bf16
        in a final 'special' shard.

        If ckpt_dir is given, each layer is loaded fresh from disk inside the
        loop and released after export — bounds Phase 6 peak memory to
        ~1 layer + export overhead (~10 GB) regardless of model size, which is
        what makes 120 B+ models fit on a single 128 GB UMA node.

        If ckpt_dir is None, falls back to the legacy in-memory behaviour
        (layers must already live in self.local_layers)."""
        import shutil
        import gc
        from safetensors import safe_open
        from safetensors.torch import save_file
        from modelopt.torch.quantization.nn import TensorQuantizer

        os.makedirs(output_dir, exist_ok=True)

        # Drop calibration wrapper (frees ~1 GB) if save_layers_to_disk didn't already.
        # NOTE: do NOT set mod._calibrator = None — modelopt 0.43's export path uses
        # set_quantizer_by_cfg_context internally, whose __exit__ does
        # setattr(self._calibrator, key, ...) and AttributeErrors on None.
        if hasattr(self, "_quant_wrapper"):
            del self._quant_wrapper
        gc.collect()
        torch.cuda.empty_cache()

        mem0 = torch.cuda.memory_allocated() / 1e9
        print(f"[{self.name}] start export (ckpt={'yes' if ckpt_dir else 'no'}), mem={mem0:.1f}G", flush=True)

        # cfg with tiny vocab_size so dummies are ~16 KB each instead of 2.1 GB
        cfg_t = AutoConfig.from_pretrained(self.model_path)
        cfg_t.num_hidden_layers = 1
        cfg_t.vocab_size = 2
        cfg_t.pad_token_id = None
        cfg_t.bos_token_id = None
        cfg_t.eos_token_id = None

        n_local = self.layer_end - self.layer_start
        weight_map = {}
        total_size = 0
        config_content = None
        aux_files = {}

        # Phase 6a: per-layer export with tiny dummies
        for idx_in, orig_i in enumerate(range(self.layer_start, self.layer_end)):
            local_idx = idx_in

            with init_empty_weights():
                m_one = AutoModelForCausalLM.from_config(cfg_t, torch_dtype=torch.bfloat16)

            # Real quantized layer — either still resident in self.local_layers
            # (legacy path) or loaded from disk per-iter (memory-streaming path)
            if ckpt_dir is not None:
                layer = torch.load(
                    f"{ckpt_dir}/layer_{orig_i:04d}.pt",
                    map_location=self.device,
                    weights_only=False,
                )
                # Saved on CPU; ensure all params/buffers land on the actor's device
                layer = layer.to(self.device)
                m_one.model.layers[0] = layer
            else:
                m_one.model.layers[0] = self.local_layers[orig_i]

            # Tiny dummies (16 KB each, totally negligible)
            de = nn.Embedding(cfg_t.vocab_size, cfg_t.hidden_size).to(self.device, dtype=torch.bfloat16)
            with torch.no_grad():
                de.weight.zero_()
            m_one.model.embed_tokens = de

            dn = self.rms_norm_cls(cfg_t.hidden_size, eps=cfg_t.rms_norm_eps).to(self.device, dtype=torch.bfloat16)
            with torch.no_grad():
                dn.weight.zero_()
            m_one.model.norm = dn

            dl = nn.Linear(cfg_t.hidden_size, cfg_t.vocab_size, bias=False).to(self.device, dtype=torch.bfloat16)
            with torch.no_grad():
                dl.weight.zero_()
            m_one.lm_head = dl

            m_one.model.rotary_emb = self.rotary_emb_cls(cfg_t).to(self.device)

            tmp = f"/tmp/_distrib_quant_{self.name}_l{orig_i:04d}"
            shutil.rmtree(tmp, ignore_errors=True)
            mte.export_hf_checkpoint(m_one, dtype=torch.bfloat16, export_dir=tmp)

            # Read back only model.layers.0.* keys, rename to local_idx
            layer_sd = {}
            with safe_open(f"{tmp}/model.safetensors", framework="pt") as sf:
                for k in sf.keys():
                    if k.startswith("model.layers.0."):
                        new_k = k.replace("model.layers.0.", f"model.layers.{local_idx}.", 1)
                        layer_sd[new_k] = sf.get_tensor(k)
                    # ignore everything else (dummies)

            # Capture aux config files from the very first iteration
            if idx_in == 0:
                with open(f"{tmp}/config.json") as f:
                    config_content = f.read()
                for fn in ("generation_config.json", "hf_quant_config.json"):
                    src = f"{tmp}/{fn}"
                    if os.path.exists(src):
                        with open(src) as f:
                            aux_files[fn] = f.read()

            shard_fn = f"shard_l{orig_i:04d}.safetensors"
            save_file(layer_sd, f"{output_dir}/{shard_fn}")
            for k, t in layer_sd.items():
                weight_map[k] = shard_fn
                total_size += t.numel() * t.element_size()

            shutil.rmtree(tmp, ignore_errors=True)
            del layer_sd, m_one
            # Drop our reference to this layer so it can be GC'd.
            # In ckpt mode the layer was loaded fresh per-iter — nothing held
            # in self.local_layers, just collect what fell out of scope above.
            if ckpt_dir is None:
                self.local_layers[orig_i] = nn.Identity()
            gc.collect()
            torch.cuda.empty_cache()

            if (local_idx + 1) % 5 == 0 or local_idx == n_local - 1:
                mem = torch.cuda.memory_allocated() / 1e9
                print(f"  [{self.name}] {local_idx + 1}/{n_local} layers (mem={mem:.1f}G)", flush=True)

        # Phase 6b: save real embed/norm/lm_head as bf16 in a separate "special"
        # shard. In ckpt mode they may have been evicted to disk by
        # save_layers_to_disk — load them back from there if needed.
        def _maybe_load(attr_name, file_name):
            """Get the attr from self, falling back to <ckpt_dir>/<file_name>.pt."""
            obj = getattr(self, attr_name, None)
            if obj is not None:
                return obj
            if ckpt_dir is not None and os.path.exists(f"{ckpt_dir}/{file_name}.pt"):
                return torch.load(f"{ckpt_dir}/{file_name}.pt", map_location="cpu",
                                  weights_only=False)
            return None

        special_sd = {}
        if self.has_embed:
            embed = _maybe_load("embed_tokens", "embed_tokens")
            if embed is not None:
                special_sd["model.embed_tokens.weight"] = embed.weight.detach().to(torch.bfloat16).cpu()
                self.embed_tokens = None
                del embed
        if self.has_head:
            norm = _maybe_load("norm", "norm")
            if norm is not None:
                special_sd["model.norm.weight"] = norm.weight.detach().to(torch.bfloat16).cpu()
                self.norm = None
                del norm
            lm_head = _maybe_load("lm_head", "lm_head")
            if lm_head is not None:
                # Save as bf16; will add lm_head to ignore list in merged config
                special_sd["lm_head.weight"] = lm_head.weight.detach().to(torch.bfloat16).cpu()
                self.lm_head = None
                del lm_head

        if special_sd:
            special_fn = "shard_special.safetensors"
            save_file(special_sd, f"{output_dir}/{special_fn}")
            for k, t in special_sd.items():
                weight_map[k] = special_fn
                total_size += t.numel() * t.element_size()
            print(f"  [{self.name}] saved special shard: {list(special_sd.keys())}", flush=True)

        # Save aux config files
        if config_content:
            with open(f"{output_dir}/config.json", "w") as f:
                f.write(config_content)
        for fn, c in aux_files.items():
            with open(f"{output_dir}/{fn}", "w") as f:
                f.write(c)

        # Final cleanup
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        torch.cuda.empty_cache()
        mem_end = torch.cuda.memory_allocated() / 1e9

        return f"{self.name}: exported {n_local} layers + special, {len(weight_map)} keys, {total_size/1e9:.1f}G, end mem={mem_end:.1f}G"


# ============================================================
# Driver-side: merge per-actor exports into final HF model dir
# ============================================================
def merge_exports(shard0_dir, shard1_dir, final_dir, source_model_path,
                  num_layers_shard0, num_layers_total):
    """Merge per-actor exports into a single HF compressed-tensors model dir."""
    import shutil
    from safetensors import safe_open
    from safetensors.torch import save_file

    os.makedirs(final_dir, exist_ok=True)

    def load_dir_sd(d):
        sd = {}
        for f in sorted(os.listdir(d)):
            if f.endswith(".safetensors"):
                with safe_open(os.path.join(d, f), framework="pt") as sf:
                    for k in sf.keys():
                        sd[k] = sf.get_tensor(k)
        return sd

    print("  Loading shard0 safetensors...", flush=True)
    sd0 = load_dir_sd(shard0_dir)
    print(f"    shard0: {len(sd0)} keys", flush=True)
    print("  Loading shard1 safetensors...", flush=True)
    sd1 = load_dir_sd(shard1_dir)
    print(f"    shard1: {len(sd1)} keys", flush=True)

    # Merge with proper key renaming.
    # shard0 owns: embed_tokens + model.layers.0..N-1 (already correct indices)
    # shard1 owns: model.layers.0..M-1 (need rename to N..N+M-1) + model.norm + lm_head
    merged = {}

    # shard0: drop dummy lm_head.* and dummy model.norm.weight
    n_skipped_0 = 0
    for k, v in sd0.items():
        if k.startswith("lm_head") or k == "model.norm.weight":
            n_skipped_0 += 1
            continue
        merged[k] = v
    print(f"    shard0: kept {len(sd0) - n_skipped_0} keys, dropped {n_skipped_0} dummies", flush=True)

    # shard1: drop dummy model.embed_tokens.weight, rename layer indices
    n_skipped_1 = 0
    n_renamed = 0
    for k, v in sd1.items():
        if k == "model.embed_tokens.weight":
            n_skipped_1 += 1
            continue
        if k.startswith("model.layers."):
            parts = k.split(".")
            old_idx = int(parts[2])
            new_idx = old_idx + num_layers_shard0
            new_k = f"model.layers.{new_idx}." + ".".join(parts[3:])
            merged[new_k] = v
            n_renamed += 1
        else:
            merged[k] = v
    print(f"    shard1: kept {len(sd1) - n_skipped_1} keys (renamed {n_renamed} layer keys), dropped {n_skipped_1} dummies", flush=True)
    print(f"  Merged total: {len(merged)} keys", flush=True)

    # Inject input_scale=1.0 for every quantized Linear. modelopt 0.43 omits these
    # keys entirely; vLLM's modelopt loader (process_weights_after_loading in
    # modelopt.py) registers an uninitialized Parameter for input_scale, then
    # computes alpha = input_global_scale * weight_global_scale from random values
    # → garbage output. 1.0 is the identity for dynamic-input quantization, where
    # the runtime per-token scale is applied inside the NVFP4 kernel.
    n_input_scale = 0
    for k in list(merged.keys()):
        if k.endswith(".weight_scale_2"):
            in_key = k.replace(".weight_scale_2", ".input_scale")
            if in_key not in merged:
                merged[in_key] = torch.tensor(1.0, dtype=torch.float32)
                n_input_scale += 1
    print(f"  Injected {n_input_scale} input_scale=1.0 keys", flush=True)

    # Compute total size
    total_bytes = sum(t.numel() * t.element_size() for t in merged.values())
    print(f"  Total size: {total_bytes / 1e9:.2f} GB", flush=True)

    # Shard if >5GB per file
    SHARD_SIZE = 5 * 1024**3
    if total_bytes <= SHARD_SIZE * 1.05:
        save_file(merged, f"{final_dir}/model.safetensors")
        print(f"  Saved single model.safetensors", flush=True)
    else:
        n_shards = max(1, (total_bytes + SHARD_SIZE - 1) // SHARD_SIZE)
        shards = [{} for _ in range(n_shards)]
        shard_sizes = [0] * n_shards
        weight_map = {}
        # Sort keys for reproducible sharding
        for k in sorted(merged.keys()):
            v = merged[k]
            size = v.numel() * v.element_size()
            i = min(range(n_shards), key=lambda j: shard_sizes[j])
            shards[i][k] = v
            shard_sizes[i] += size
            weight_map[k] = f"model-{i+1:05d}-of-{n_shards:05d}.safetensors"

        for i, shard_dict in enumerate(shards):
            fname = f"model-{i+1:05d}-of-{n_shards:05d}.safetensors"
            save_file(shard_dict, f"{final_dir}/{fname}")
            print(f"  Saved {fname} ({shard_sizes[i]/1e9:.2f} GB, {len(shard_dict)} keys)", flush=True)

        index = {
            "metadata": {"total_size": total_bytes},
            "weight_map": weight_map,
        }
        with open(f"{final_dir}/model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)
        print(f"  Saved model.safetensors.index.json", flush=True)

    # config.json: take shard0's (has quantization_config), restore from source what we
    # shrank for Phase-6 dummies (num_hidden_layers, vocab_size, pad/bos/eos token IDs).
    with open(f"{shard0_dir}/config.json") as fh:
        cfg = json.load(fh)
    with open(f"{source_model_path}/config.json") as fh:
        src_cfg = json.load(fh)
    cfg["num_hidden_layers"] = num_layers_total
    cfg["vocab_size"] = src_cfg["vocab_size"]
    for tok_key in ("pad_token_id", "bos_token_id", "eos_token_id"):
        if tok_key in src_cfg:
            cfg[tok_key] = src_cfg[tok_key]
    # Preserve architectures from source — modelopt's per-layer export may rewrite it.
    if "architectures" in src_cfg:
        cfg["architectures"] = src_cfg["architectures"]
    if "quantization_config" in cfg:
        # Add lm_head to ignore list — we saved it as bf16, not quantized
        ignore = cfg["quantization_config"].get("ignore", [])
        if "lm_head" not in ignore:
            ignore.append("lm_head")
        cfg["quantization_config"]["ignore"] = ignore
        # modelopt 0.43 writes input_activations.dynamic=false but does NOT emit
        # static input_scale keys. vLLM then falls back to input_scale=1.0 and
        # produces garbage output. Force dynamic=true so vLLM uses per-token
        # dynamic input quantization at inference (no static scales required).
        for group in cfg["quantization_config"].get("config_groups", {}).values():
            if isinstance(group, dict) and "input_activations" in group:
                group["input_activations"]["dynamic"] = True
    with open(f"{final_dir}/config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"  Saved config.json (num_hidden_layers={num_layers_total}, vocab_size={cfg['vocab_size']}, ignore={cfg.get('quantization_config', {}).get('ignore', [])})", flush=True)

    # Copy aux config files from shard0
    for fn in ("generation_config.json", "hf_quant_config.json"):
        src = f"{shard0_dir}/{fn}"
        if os.path.exists(src):
            shutil.copy(src, f"{final_dir}/{fn}")
            print(f"  Copied {fn}", flush=True)

    # Copy tokenizer from source model
    for fn in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model"):
        src = f"{source_model_path}/{fn}"
        if os.path.exists(src):
            shutil.copy(src, f"{final_dir}/{fn}")
            print(f"  Copied {fn} from source", flush=True)


# ============================================================
# Driver
# ============================================================
def main():
    args = parse_args()
    ray.init(address="auto")
    print(f"Cluster resources: {ray.cluster_resources()}", flush=True)

    # Resolve resume-vs-fresh-run parameters early so node setup is shared.
    if args.resume_from_checkpoint:
        ckpt0 = f"{args.resume_from_checkpoint}_ckpt_shard0"
        ckpt1 = f"{args.resume_from_checkpoint}_ckpt_shard1"
        for c in (ckpt0, ckpt1):
            if not os.path.exists(f"{c}/meta.json"):
                print(f"FATAL: {c}/meta.json missing — pass the same base path used in the prior --temp-base.", flush=True)
                sys.exit(1)
        with open(f"{ckpt0}/meta.json") as f:
            meta0 = json.load(f)
        with open(f"{ckpt1}/meta.json") as f:
            meta1 = json.load(f)
        source_model = args.source_model or meta0["model_path"]
        split = meta0["layer_end"]
        num_layers = meta1["layer_end"]
        print(f"RESUME mode — ckpt0={ckpt0}, ckpt1={ckpt1}", flush=True)
        print(f"Source: {source_model} (from {'CLI' if args.source_model else 'meta.json'})", flush=True)
        print(f"Output: {args.output_dir}", flush=True)
        print(f"Layers: {num_layers} total, split at {split}", flush=True)
    else:
        if not args.source_model:
            print("FATAL: --source-model is required for normal (non-resume) runs.", flush=True)
            sys.exit(1)
        source_model = args.source_model
        print(f"Source: {source_model}", flush=True)
        print(f"Output: {args.output_dir}", flush=True)

        # Determine layer count from model config
        config = AutoConfig.from_pretrained(source_model)
        num_layers = config.num_hidden_layers
        split = args.split if args.split is not None else (num_layers // 2)
        print(f"Model: {config.architectures[0]}, hidden_size={config.hidden_size}, layers={num_layers}, split at {split}", flush=True)

    # Identify nodes
    nodes = [n for n in ray.nodes() if n["Alive"]]
    if len(nodes) < 2:
        print(f"FATAL: only {len(nodes)} live nodes, need 2", flush=True)
        sys.exit(1)
    node_ids = [n["NodeID"] for n in nodes]
    print(f"Using nodes: {[n['NodeManagerAddress'] for n in nodes]}", flush=True)

    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    shard0 = ModelShard.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_ids[0], soft=False),
        name="shard0",
    ).remote(source_model, 0, split, has_embed=True, has_head=False, name="shard0")

    shard1 = ModelShard.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_ids[1], soft=False),
        name="shard1",
    ).remote(source_model, split, num_layers, has_embed=False, has_head=True, name="shard1")

    # --- Resume mode: skip Phases 1-5.5 and go straight to Phase 6 ---
    if args.resume_from_checkpoint:
        print("\n=== Resume setup ===", flush=True)
        for r in ray.get([
            shard0.setup_for_resume.remote(ckpt0),
            shard1.setup_for_resume.remote(ckpt1),
        ]):
            print(f"  {r}", flush=True)

        print("\n=== Phase 6: Per-actor export (resumed from checkpoint) ===", flush=True)
        shard0_temp = f"{args.temp_base}_shard0"
        shard1_temp = f"{args.temp_base}_shard1"
        t0 = time.time()
        for r in ray.get([
            shard0.export_shard.remote(shard0_temp, ckpt0),
            shard1.export_shard.remote(shard1_temp, ckpt1),
        ]):
            print(f"  {r}", flush=True)
        print(f"  Per-actor export done in {(time.time()-t0):.1f}s", flush=True)

        # Phase 7: merge
        print(f"\n=== Phase 7: Merge to {args.output_dir} ===", flush=True)
        t0 = time.time()
        merge_exports(
            shard0_temp, shard1_temp, args.output_dir, source_model,
            num_layers_shard0=split, num_layers_total=num_layers,
        )
        print(f"  Merge done in {(time.time()-t0):.1f}s", flush=True)

        print(f"\nNVFP4 model written to: {args.output_dir}")
        print(f"Resume checkpoint dirs still at {ckpt0} and {ckpt1} — delete when satisfied.")
        return

    # Phase 1
    print("\n=== Phase 1: Loading model halves ===", flush=True)
    t0 = time.time()
    for r in ray.get([shard0.load_model.remote(), shard1.load_model.remote()]):
        print(f"  {r}", flush=True)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    # Phase 2
    print("\n=== Phase 2: Inserting NVFP4 quantizers ===", flush=True)
    for r in ray.get([shard0.apply_quantization.remote(), shard1.apply_quantization.remote()]):
        print(f"  {r}", flush=True)

    # Phase 3: smoke test with 1 sample first
    print("\n=== Phase 3a: Smoke test (1 sample) ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.source_model)
    test_text = "The quick brown fox jumps over the lazy dog. " * 50
    inputs = tokenizer(test_text, truncation=True, max_length=args.seq_len, return_tensors="pt")
    t0 = time.time()
    hidden, pos_ids = ray.get(shard0.forward_first.remote(inputs.input_ids))
    print(f"  shard0 returned hidden_states shape={list(hidden.shape)}, dtype={hidden.dtype}, took {time.time()-t0:.1f}s", flush=True)
    t1 = time.time()
    ray.get(shard1.forward_second.remote(hidden, pos_ids))
    print(f"  shard1 forward done, took {time.time()-t1:.1f}s", flush=True)

    # Phase 3b: full calibration
    print(f"\n=== Phase 3b: Calibration with {args.calib_size} samples ===", flush=True)
    ds_args = (args.dataset, args.dataset_config) if args.dataset_config else (args.dataset,)
    dataset = load_dataset(*ds_args, split="train", streaming=True)
    t0 = time.time()
    count = 0
    for sample in dataset:
        if count >= args.calib_size:
            break
        text = sample.get("article", sample.get("text", ""))
        if not text or len(text) < 100:
            continue

        inputs = tokenizer(text, truncation=True, max_length=args.seq_len, return_tensors="pt")
        hidden, pos_ids = ray.get(shard0.forward_first.remote(inputs.input_ids))
        ray.get(shard1.forward_second.remote(hidden, pos_ids))

        count += 1
        if count % 10 == 0:
            elapsed = time.time() - t0
            eta = (args.calib_size - count) * elapsed / count
            print(f"  {count}/{args.calib_size} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    print(f"  Calibration done in {(time.time()-t0)/60:.1f} min", flush=True)

    # Phase 4
    print("\n=== Phase 4: Finalizing quantization ===", flush=True)
    for r in ray.get([shard0.finalize.remote(), shard1.finalize.remote()]):
        print(f"  {r}", flush=True)

    # Phase 5 (diagnostic, not full export yet)
    print("\n=== Phase 5: Diagnostic - amax stats ===", flush=True)
    for r in ray.get([shard0.export_status.remote(), shard1.export_status.remote()]):
        print(f"  {r}", flush=True)

    # Phase 5.5: evict layers to disk so Phase 6 can stream from disk.
    # This drops peak Phase-6 memory from O(N_layers * layer_size) down to
    # O(1 layer + export overhead) — required for 120B+ models that occupy
    # >90% of a 128 GB UMA pool after load.
    print("\n=== Phase 5.5: Evicting layers to disk for streaming export ===", flush=True)
    ckpt0 = f"{args.temp_base}_ckpt_shard0"
    ckpt1 = f"{args.temp_base}_ckpt_shard1"
    t0 = time.time()
    for r in ray.get([
        shard0.save_layers_to_disk.remote(ckpt0),
        shard1.save_layers_to_disk.remote(ckpt1),
    ]):
        print(f"  {r}", flush=True)
    print(f"  Eviction done in {(time.time()-t0):.1f}s", flush=True)

    # Phase 6: per-actor export, streaming layers from the checkpoint dirs.
    # If Phase 6 crashes mid-way, the checkpoint dirs survive — a future
    # resume entry could re-run just Phase 6 against them.
    print("\n=== Phase 6: Per-actor export (disk-streaming) ===", flush=True)
    shard0_temp = f"{args.temp_base}_shard0"
    shard1_temp = f"{args.temp_base}_shard1"
    t0 = time.time()
    for r in ray.get([
        shard0.export_shard.remote(shard0_temp, ckpt0),
        shard1.export_shard.remote(shard1_temp, ckpt1),
    ]):
        print(f"  {r}", flush=True)
    print(f"  Per-actor export done in {(time.time()-t0):.1f}s", flush=True)

    # Phase 7: merge
    print(f"\n=== Phase 7: Merge to {args.output_dir} ===", flush=True)
    t0 = time.time()
    merge_exports(
        shard0_temp, shard1_temp, args.output_dir, args.source_model,
        num_layers_shard0=split, num_layers_total=num_layers,
    )
    print(f"  Merge done in {(time.time()-t0):.1f}s", flush=True)

    print(f"\nNVFP4 model written to: {args.output_dir}")
    print(f"Temp dirs can be deleted: rm -rf {args.temp_base}_shard0 {args.temp_base}_shard1 {args.temp_base}_ckpt_shard0 {args.temp_base}_ckpt_shard1")


if __name__ == "__main__":
    main()
