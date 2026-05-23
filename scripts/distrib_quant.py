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
                   help="Legacy 2-shard split point. Default: num_layers // 2. "
                        "Overridden by --shard-layers if given.")
    p.add_argument("--shard-layers", default=None,
                   help="Comma-separated layer counts per shard, e.g. '40,40,8' for "
                        "3-shard. Must sum to num_hidden_layers. Enables N-shard mode.")
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Skip Phases 1-5 and run only Phase 6+7 against an existing "
                        "checkpoint dir base. Expects {BASE}_ckpt_shard0..N to exist "
                        "(created by a previous Phase-5.5).")
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

        # Tied embeddings (e.g. Cohere2): the checkpoint has no lm_head.weight —
        # the output projection reuses model.embed_tokens.weight.
        tied_lm_head = "lm_head.weight" not in weight_map
        self.tied_lm_head = tied_lm_head

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
        # Tied-embedding models: the has_head shard needs embed_tokens.weight to
        # populate lm_head (it has no lm_head.* keys of its own).
        if (self.has_head and tied_lm_head
                and "model.embed_tokens.weight" in weight_map
                and "model.embed_tokens.weight" not in needed_keys):
            needed_keys.append("model.embed_tokens.weight")

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
                        # Tied-embedding has_head shard: route embed_tokens.weight
                        # straight into lm_head.weight (no lm_head.* in checkpoint;
                        # avoids double-storing the embedding on this shard).
                        if (k == "model.embed_tokens.weight" and self.has_head
                                and tied_lm_head and not self.has_embed):
                            target_key = "lm_head.weight"
                        else:
                            target_key = k
                        set_module_tensor_to_device(
                            self.model, target_key, self.device, value=t,
                        )
                        del t
            gc.collect()

        # Tied embeddings, single-shard case (has_embed AND has_head): the load
        # loop filled embed_tokens.weight; mirror it into lm_head.weight.
        if self.has_head and tied_lm_head and self.has_embed:
            ew = self.model.model.embed_tokens.weight
            set_module_tensor_to_device(
                self.model, "lm_head.weight", self.device, value=ew.data,
            )

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

    def forward_middle(self, hidden_states, position_ids):
        """Middle actor (N-shard mode): hidden_states -> hidden_states, no norm/lm_head."""
        with torch.no_grad():
            hidden = hidden_states.to(self.device)
            position_ids = position_ids.to(self.device)
            seq_len = hidden.shape[1]

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

        Uses cloudpickle as pickle_module — modelopt 0.43's QuantLinear is a
        dynamically-generated subclass that vanilla pickle can't serialize
        (fails with `attribute lookup QuantLinear on modelopt.torch.opt.dynamic
        failed` on some platforms — observed on x86_64 + torch 2.11+cu130 VM).
        cloudpickle handles dynamic classes by inlining their bytecode.
        """
        import gc, cloudpickle
        os.makedirs(ckpt_dir, exist_ok=True)

        if hasattr(self, "_quant_wrapper"):
            del self._quant_wrapper

        n = 0
        for i in range(self.layer_start, self.layer_end):
            path = f"{ckpt_dir}/layer_{i:04d}.pt"
            torch.save(self.local_layers[i].cpu(), path, pickle_module=cloudpickle)
            self.local_layers[i] = nn.Identity()
            # Free the evicted layer's GPU memory immediately. On GB10 UMA the
            # .cpu() copy doubles the layer in the shared pool; without this the
            # peak climbs and Ray's memory monitor kills the worker.
            gc.collect()
            torch.cuda.empty_cache()
            n += 1

        extras_saved = []
        if self.has_embed and self.embed_tokens is not None:
            torch.save(self.embed_tokens.cpu(), f"{ckpt_dir}/embed_tokens.pt", pickle_module=cloudpickle)
            self.embed_tokens = None
            extras_saved.append("embed_tokens")
        if self.has_head and self.norm is not None:
            torch.save(self.norm.cpu(), f"{ckpt_dir}/norm.pt", pickle_module=cloudpickle)
            self.norm = None
            extras_saved.append("norm")
        if self.has_head and self.lm_head is not None:
            torch.save(self.lm_head.cpu(), f"{ckpt_dir}/lm_head.pt", pickle_module=cloudpickle)
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
        # use_cache=False so transformers' DynamicCache doesn't try to index
        # the loaded layer's original layer_idx (e.g. 50) against the 1-slot
        # cache of our 1-layer template — IndexError otherwise.
        cfg_t.use_cache = False

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
            # Force layer_idx=0 on the loaded layer (it kept its original index
            # from the full model, e.g. 50, but our 1-layer template only has
            # slot 0). Required for transformers>=4.50 DynamicCache to not
            # IndexError even with use_cache=False set on the config.
            placed = m_one.model.layers[0]
            if hasattr(placed, "self_attn") and hasattr(placed.self_attn, "layer_idx"):
                placed.self_attn.layer_idx = 0
            if hasattr(placed, "layer_idx"):
                placed.layer_idx = 0

            # Tiny dummies (16 KB each, totally negligible)
            de = nn.Embedding(cfg_t.vocab_size, cfg_t.hidden_size).to(self.device, dtype=torch.bfloat16)
            with torch.no_grad():
                de.weight.zero_()
            m_one.model.embed_tokens = de

            _norm_eps = getattr(cfg_t, "rms_norm_eps", None)
            if _norm_eps is None:
                _norm_eps = getattr(cfg_t, "layer_norm_eps", 1e-5)
            dn = self.rms_norm_cls(cfg_t.hidden_size, eps=_norm_eps).to(self.device, dtype=torch.bfloat16)
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
            # Cohere2's generation_config sets cache_implementation=hybrid, which
            # transformers 4.57 rejects together with the use_cache=False export
            # template. Drop it — the final model gets its real generation_config
            # copied from the source in the merge phase.
            if getattr(m_one, "generation_config", None) is not None:
                m_one.generation_config.cache_implementation = None
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
def merge_exports(shard_dirs, layer_counts, final_dir, source_model_path,
                  num_layers_total):
    """Merge per-actor exports into a single HF compressed-tensors model dir.
    N-shard generalization:
      shard_dirs: list of per-shard export directories in order
      layer_counts: list of layer counts per shard (cumulative sum = num_layers_total)
      First shard contributes embed_tokens.
      Last shard contributes model.norm + lm_head.
      Middle shards just contribute their layers (renamed with cumulative offset).
    """
    import shutil
    from safetensors import safe_open
    from safetensors.torch import save_file

    assert len(shard_dirs) == len(layer_counts), "shard_dirs and layer_counts must align"
    assert sum(layer_counts) == num_layers_total, \
        f"layer_counts sum {sum(layer_counts)} != num_layers_total {num_layers_total}"

    os.makedirs(final_dir, exist_ok=True)

    def load_dir_sd(d):
        sd = {}
        for f in sorted(os.listdir(d)):
            if f.endswith(".safetensors"):
                with safe_open(os.path.join(d, f), framework="pt") as sf:
                    for k in sf.keys():
                        sd[k] = sf.get_tensor(k)
        return sd

    merged = {}
    offset = 0
    N = len(shard_dirs)
    for i, (d, n_layers) in enumerate(zip(shard_dirs, layer_counts)):
        print(f"  Loading shard{i} safetensors from {d}...", flush=True)
        sd = load_dir_sd(d)
        print(f"    shard{i}: {len(sd)} keys", flush=True)

        is_first, is_last = (i == 0), (i == N - 1)
        n_skipped, n_renamed = 0, 0
        for k, v in sd.items():
            # Embed_tokens: keep only on first shard, drop dummy on others
            if k == "model.embed_tokens.weight":
                if not is_first:
                    n_skipped += 1
                    continue
                merged[k] = v
            # lm_head + model.norm: keep only on last shard, drop dummy on others
            elif k.startswith("lm_head") or k == "model.norm.weight":
                if not is_last:
                    n_skipped += 1
                    continue
                merged[k] = v
            # Layer tensors: rename with cumulative offset
            elif k.startswith("model.layers."):
                parts = k.split(".")
                old_idx = int(parts[2])
                new_idx = old_idx + offset if not is_first else old_idx
                new_k = f"model.layers.{new_idx}." + ".".join(parts[3:])
                merged[new_k] = v
                if not is_first:
                    n_renamed += 1
            else:
                merged[k] = v
        print(f"    shard{i}: kept {len(sd) - n_skipped} (renamed {n_renamed}), dropped {n_skipped}", flush=True)
        offset += n_layers
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

    # config.json: take first shard's (has quantization_config), restore from source what we
    # shrank for Phase-6 dummies (num_hidden_layers, vocab_size, pad/bos/eos token IDs).
    with open(f"{shard_dirs[0]}/config.json") as fh:
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

    # Copy aux config files from first shard
    for fn in ("generation_config.json", "hf_quant_config.json"):
        src = f"{shard_dirs[0]}/{fn}"
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
        # Detect how many shard checkpoints exist
        ckpt_dirs = []
        for i in range(10):  # max 10 shards
            cand = f"{args.resume_from_checkpoint}_ckpt_shard{i}"
            if os.path.exists(f"{cand}/meta.json"):
                ckpt_dirs.append(cand)
            else:
                break
        if len(ckpt_dirs) < 2:
            print(f"FATAL: no checkpoint dirs found at {args.resume_from_checkpoint}_ckpt_shard0..", flush=True)
            sys.exit(1)
        metas = []
        for c in ckpt_dirs:
            with open(f"{c}/meta.json") as f:
                metas.append(json.load(f))
        source_model = args.source_model or metas[0]["model_path"]
        layer_counts = [m["layer_end"] - m["layer_start"] for m in metas]
        num_layers = sum(layer_counts)
        print(f"RESUME mode — {len(ckpt_dirs)} shards, layer_counts={layer_counts}", flush=True)
        print(f"Source: {source_model}", flush=True)
        print(f"Output: {args.output_dir}", flush=True)
    else:
        if not args.source_model:
            print("FATAL: --source-model is required for normal (non-resume) runs.", flush=True)
            sys.exit(1)
        source_model = args.source_model
        print(f"Source: {source_model}", flush=True)
        print(f"Output: {args.output_dir}", flush=True)

        config = AutoConfig.from_pretrained(source_model)
        num_layers = config.num_hidden_layers

        if args.shard_layers:
            layer_counts = [int(x.strip()) for x in args.shard_layers.split(",")]
            if sum(layer_counts) != num_layers:
                print(f"FATAL: --shard-layers sum {sum(layer_counts)} != num_layers {num_layers}", flush=True)
                sys.exit(1)
        else:
            split = args.split if args.split is not None else (num_layers // 2)
            layer_counts = [split, num_layers - split]
        print(f"Model: {config.architectures[0]}, hidden_size={config.hidden_size}, layers={num_layers}, shard_layers={layer_counts}", flush=True)

    # Identify nodes — sort by available memory descending so biggest shards
    # land on nodes with most RAM (e.g. Sparks get the big shards; small eGPU
    # boxes get the trailing tiny shard with lm_head).
    N = len(layer_counts)
    nodes = sorted([n for n in ray.nodes() if n["Alive"]],
                   key=lambda n: -n["Resources"].get("memory", 0))
    if len(nodes) < N:
        print(f"FATAL: only {len(nodes)} live nodes, need {N}", flush=True)
        sys.exit(1)
    node_ids = [n["NodeID"] for n in nodes[:N]]
    print(f"Using nodes (memory-sorted desc): {[(n['NodeManagerAddress'], int(n['Resources'].get('memory', 0)/1e9)) for n in nodes[:N]]}", flush=True)

    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    # Build N shards. First has embed, last has head, middles have neither.
    shards = []
    offset = 0
    for i, n_layers in enumerate(layer_counts):
        is_first, is_last = (i == 0), (i == N - 1)
        shard = ModelShard.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_ids[i], soft=False),
            name=f"shard{i}",
        ).remote(source_model, offset, offset + n_layers,
                 has_embed=is_first, has_head=is_last, name=f"shard{i}")
        shards.append(shard)
        offset += n_layers
    # Backwards-compat aliases used elsewhere in main()
    shard0, shard1 = shards[0], shards[-1]

    # --- Resume mode: skip Phases 1-5.5 and go straight to Phase 6 ---
    if args.resume_from_checkpoint:
        print("\n=== Resume setup ===", flush=True)
        for r in ray.get([shards[i].setup_for_resume.remote(ckpt_dirs[i]) for i in range(N)]):
            print(f"  {r}", flush=True)

        print("\n=== Phase 6: Per-actor export (resumed from checkpoint) ===", flush=True)
        temp_dirs = [f"{args.temp_base}_shard{i}" for i in range(N)]
        t0 = time.time()
        for r in ray.get([shards[i].export_shard.remote(temp_dirs[i], ckpt_dirs[i]) for i in range(N)]):
            print(f"  {r}", flush=True)
        print(f"  Per-actor export done in {(time.time()-t0):.1f}s", flush=True)

        # Phase 6.5: pull per-shard exports from remote nodes to driver-local NFS.
        # Same logic as the non-resume path; without this, multi-node resumes
        # whose actors are not NFS-shared with the driver fail Phase 7 with
        # FileNotFoundError on temp_dirs[i].
        import subprocess, socket
        driver_ip = socket.gethostbyname(socket.gethostname())
        print(f"\n=== Phase 6.5: Gather remote shard exports (driver={driver_ip}) ===", flush=True)
        for i in range(N):
            actor_ip = nodes[i]["NodeManagerAddress"]
            if actor_ip == driver_ip:
                continue
            local_files = len(os.listdir(temp_dirs[i])) if os.path.exists(temp_dirs[i]) else 0
            if local_files >= layer_counts[i]:
                print(f"  shard{i}@{actor_ip}: already visible locally ({local_files} files), skip", flush=True)
                continue
            print(f"  shard{i}@{actor_ip}: rsyncing to driver-local {temp_dirs[i]}/ ...", flush=True)
            subprocess.run([
                "rsync", "-a", "-e", "ssh -o StrictHostKeyChecking=no",
                f"kai@{actor_ip}:{temp_dirs[i]}/",
                f"{temp_dirs[i]}/",
            ], check=True)
            print(f"  shard{i}: {len(os.listdir(temp_dirs[i]))} files after sync", flush=True)

        # Phase 7: merge
        print(f"\n=== Phase 7: Merge to {args.output_dir} ===", flush=True)
        t0 = time.time()
        merge_exports(temp_dirs, layer_counts, args.output_dir, source_model, num_layers)
        print(f"  Merge done in {(time.time()-t0):.1f}s", flush=True)

        print(f"\nNVFP4 model written to: {args.output_dir}")
        print(f"Resume checkpoint dirs at {ckpt_dirs} — delete when satisfied.")
        return

    # Phase 1
    print(f"\n=== Phase 1: Loading model into {N} shards ===", flush=True)
    t0 = time.time()
    for r in ray.get([s.load_model.remote() for s in shards]):
        print(f"  {r}", flush=True)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    # Phase 2
    print("\n=== Phase 2: Inserting NVFP4 quantizers ===", flush=True)
    for r in ray.get([s.apply_quantization.remote() for s in shards]):
        print(f"  {r}", flush=True)

    def _forward_chain(input_ids):
        """Drive a sample through shard0 -> shard{1..N-2}.forward_middle -> shard{N-1}.forward_second"""
        hidden, pos_ids = ray.get(shards[0].forward_first.remote(input_ids))
        for s in shards[1:-1]:
            hidden, pos_ids = ray.get(s.forward_middle.remote(hidden, pos_ids))
        ray.get(shards[-1].forward_second.remote(hidden, pos_ids))

    # Phase 3: smoke test with 1 sample first
    print("\n=== Phase 3a: Smoke test (1 sample) ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.source_model)
    test_text = "The quick brown fox jumps over the lazy dog. " * 50
    inputs = tokenizer(test_text, truncation=True, max_length=args.seq_len, return_tensors="pt")
    t0 = time.time()
    _forward_chain(inputs.input_ids)
    print(f"  smoke forward chain through {N} shards: {time.time()-t0:.1f}s", flush=True)

    # Phase 3b: full calibration
    print(f"\n=== Phase 3b: Calibration with {args.calib_size} samples ===", flush=True)
    if args.dataset.startswith("/") or args.dataset.endswith(".jsonl"):
        # Local jsonl file — load via the json builder
        print(f"  loading local jsonl: {args.dataset}", flush=True)
        dataset = load_dataset("json", data_files=args.dataset, split="train", streaming=True)
    else:
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
        _forward_chain(inputs.input_ids)

        count += 1
        if count % 10 == 0:
            elapsed = time.time() - t0
            eta = (args.calib_size - count) * elapsed / count
            print(f"  {count}/{args.calib_size} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    print(f"  Calibration done in {(time.time()-t0)/60:.1f} min", flush=True)

    # Phase 4
    print("\n=== Phase 4: Finalizing quantization ===", flush=True)
    for r in ray.get([s.finalize.remote() for s in shards]):
        print(f"  {r}", flush=True)

    # Phase 5 (diagnostic, not full export yet)
    print("\n=== Phase 5: Diagnostic - amax stats ===", flush=True)
    for r in ray.get([s.export_status.remote() for s in shards]):
        print(f"  {r}", flush=True)

    # Phase 5.5: evict layers to disk so Phase 6 can stream from disk.
    print("\n=== Phase 5.5: Evicting layers to disk for streaming export ===", flush=True)
    ckpt_dirs = [f"{args.temp_base}_ckpt_shard{i}" for i in range(N)]
    t0 = time.time()
    for r in ray.get([shards[i].save_layers_to_disk.remote(ckpt_dirs[i]) for i in range(N)]):
        print(f"  {r}", flush=True)
    print(f"  Eviction done in {(time.time()-t0):.1f}s", flush=True)

    # Phase 6: per-actor export, streaming layers from the checkpoint dirs.
    print("\n=== Phase 6: Per-actor export (disk-streaming) ===", flush=True)
    temp_dirs = [f"{args.temp_base}_shard{i}" for i in range(N)]
    t0 = time.time()
    for r in ray.get([shards[i].export_shard.remote(temp_dirs[i], ckpt_dirs[i]) for i in range(N)]):
        print(f"  {r}", flush=True)
    print(f"  Per-actor export done in {(time.time()-t0):.1f}s", flush=True)

    # Phase 6.5: pull per-shard exports from remote nodes to driver-local NFS.
    # Required when a shard actor's /mnt/data is its OWN local disk (e.g. an
    # eGPU host VM) rather than the same NFS-shared filesystem the driver sees.
    # No-op for actors on driver or NFS-peer nodes (rsync sees identical state).
    import subprocess, socket
    driver_ip = socket.gethostbyname(socket.gethostname())
    print(f"\n=== Phase 6.5: Gather remote shard exports (driver={driver_ip}) ===", flush=True)
    for i in range(N):
        actor_ip = nodes[i]["NodeManagerAddress"]
        if actor_ip == driver_ip:
            continue
        local_files = len(os.listdir(temp_dirs[i])) if os.path.exists(temp_dirs[i]) else 0
        if local_files >= layer_counts[i]:
            print(f"  shard{i}@{actor_ip}: already visible locally ({local_files} files), skip", flush=True)
            continue
        print(f"  shard{i}@{actor_ip}: rsyncing to driver-local {temp_dirs[i]}/ ...", flush=True)
        subprocess.run([
            "rsync", "-a", "-e", "ssh -o StrictHostKeyChecking=no",
            f"kai@{actor_ip}:{temp_dirs[i]}/",
            f"{temp_dirs[i]}/",
        ], check=True)
        print(f"  shard{i}: {len(os.listdir(temp_dirs[i]))} files after sync", flush=True)

    # Phase 7: merge
    print(f"\n=== Phase 7: Merge to {args.output_dir} ===", flush=True)
    t0 = time.time()
    merge_exports(temp_dirs, layer_counts, args.output_dir, args.source_model, num_layers)
    print(f"  Merge done in {(time.time()-t0):.1f}s", flush=True)

    print(f"\nNVFP4 model written to: {args.output_dir}")
    print(f"Temp dirs can be deleted: rm -rf {args.temp_base}_shard* {args.temp_base}_ckpt_shard*")


if __name__ == "__main__":
    main()
