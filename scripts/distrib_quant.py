#!/usr/bin/env python3
"""
Distributed NVFP4 quantization for HF-format models across a Ray cluster.

Architecture:
  Shard 0:        embed_tokens + layers[0:a]
  Shard 1..N-2:   middle layer slices (N-shard mode)
  Shard N-1:      trailing layers + norm + lm_head
  Driver:         orchestrates calibration via Ray RPC (hidden states hop from
                  shard to shard), then merges the per-shard exports.

Model-agnostic: works for HF decoder models whose text stack is
`model.layers / model.embed_tokens / model.norm / lm_head` once loaded as
AutoModelForCausalLM from the *text* config. That covers Llama / Mistral /
Cohere2 and the hybrid Qwen3.5/3.6-MoE family (Gated-DeltaNet + full attention,
256-expert MoE, VL checkpoint with `model.language_model.*` keys). Vision tower
and MTP head of VL checkpoints are carried over verbatim in BF16.

Run:
  python3 distrib_quant.py --source-model /path/to/BF16 --output-dir /path/to/NVFP4

See --help for all options.
"""

import argparse
import collections
import copy
import gc
import getpass
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import ray
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import modelopt.torch.quantization as mtq
import modelopt.torch.export as mte


# ============================================================
# Architecture profiles
# ============================================================
# Extra modelopt disable patterns per *text* model_type, on top of
# NVFP4_DEFAULT_CFG (which already keeps lm_head, MoE routers `*mlp.gate.*`,
# `*shared_expert_gate*` and conv1d in BF16).
ARCH_DISABLE_PATTERNS = {
    # Qwen3.5/3.6 MoE (hybrid Gated-DeltaNet + full attention): experts and
    # shared expert -> NVFP4, every token-mixer projection stays BF16.
    #  - vLLM fuses GDN in_proj_qkv+in_proj_z (and in_proj_a+b) and rejects
    #    fused modules with mixed precision; in_proj_a/b are 32-wide gates.
    #  - attention is a small share of the params; NVIDIA's own recipe puts it
    #    in FP8, which this pipeline does not emit, so BF16 is the safe choice.
    "qwen3_5_moe_text": ["*linear_attn*", "*self_attn*"],
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Distributed NVFP4 quantization across a Ray cluster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source-model", "-s", required=False,
                   help="Path to BF16 source model directory (HF format with safetensors). "
                        "Required for normal runs; optional in --resume-from-checkpoint mode "
                        "(falls back to the path stored in the checkpoint's meta.json).")
    p.add_argument("--output-dir", "-o", required=True,
                   help="Output directory for the NVFP4 model.")
    p.add_argument("--calib-size", type=int, default=256,
                   help="Number of calibration samples. The run aborts if the dataset "
                        "yields fewer usable samples.")
    p.add_argument("--seq-len", type=int, default=2048,
                   help="Max sequence length per calibration sample.")
    p.add_argument("--min-chars", type=int, default=100,
                   help="Skip calibration samples shorter than this many characters.")
    p.add_argument("--dataset", default="abisee/cnn_dailymail",
                   help="HF dataset id for calibration text, or a local .jsonl file. "
                        "Rows may carry 'article', 'text' or chat 'messages'.")
    p.add_argument("--dataset-config", default="3.0.0",
                   help="Dataset config name (use empty string '' if dataset has no config).")
    p.add_argument("--temp-base", default="/tmp/_distrib_quant",
                   help="Base temp dir for per-shard checkpoints/exports. MUST be node-local "
                        "(NOT NFS). Wiped per shard at the start of Phase 5.5 / 6.")
    p.add_argument("--split", type=int, default=None,
                   help="Legacy 2-shard split point. Default: num_layers // 2. "
                        "Overridden by --shard-layers if given.")
    p.add_argument("--shard-layers", default=None,
                   help="Comma-separated layer counts per shard, e.g. '40,40,8' for "
                        "3-shard or '40' for single-shard. Must sum to num_hidden_layers.")
    p.add_argument("--gpus-per-actor", type=float, default=1.0,
                   help="GPU fraction per shard actor. Values < 1 allow several shards "
                        "on one node (e.g. 0.5 = two shards on a single Spark).")
    p.add_argument("--moe-calib-ratio", type=float, default=1.0,
                   help="MoE only: fraction of experts every calibration token is routed "
                        "to while collecting stats (modelopt _moe_calib_experts_ratio). "
                        "1.0 = every expert gets real input scales. Outputs still use the "
                        "model's own top-k routing.")
    p.add_argument("--disable-quant", action="append", default=[],
                   help="Extra modelopt wildcard of modules to keep in BF16 "
                        "(repeatable), e.g. '*self_attn*'.")
    p.add_argument("--ssh-user", default=getpass.getuser(),
                   help="SSH user for the Phase-6.5 rsync from remote shard nodes.")
    p.add_argument("--act-format", choices=["w4a16", "w4a4"], default="w4a16",
                   help="How the NVFP4 weights are declared to the runtime. w4a16 "
                        "(quant_algo W4A16_NVFP4): vLLM keeps "
                        "activations in BF16 — on Qwen3.6-35B-A3B KLD 0.110 -> 0.076 at equal "
                        "decode speed. w4a4: plain NVFP4, activations quantized to FP4 at runtime "
                        "using the calibrated input_scale.")
    p.add_argument("--sequential", action="store_true",
                   help="Sequential mode: process the model one decoder layer at a time "
                        "(memory ~1 layer per node, any model size on a single node). The "
                        "calibration samples are split across --seq-workers workers, whose "
                        "amax values are max-merged per layer. Resumes automatically after "
                        "a crash unless --fresh is given.")
    p.add_argument("--seq-workers", type=int, default=None,
                   help="Sequential mode: number of workers (default: one per live node). "
                        "With --gpus-per-actor < 1 several workers can share a node.")
    p.add_argument("--fresh", action="store_true",
                   help="Sequential mode: ignore resumable state from an earlier run.")
    p.add_argument("--layer-stats", action="store_true",
                   help="Sequential mode: after each layer, measure its quantization error on "
                        "BF16 inputs (no accumulation from earlier layers) — whole layer, "
                        "attention and MLP in isolation, and every MoE expert with its token "
                        "count. Error = relative L2 vs. the same layer unquantized, using the "
                        "quantizers active at serve time (--act-format). Written to "
                        "<output-dir>/layer_stats.json; outliers are listed at the end.")
    p.add_argument("--layer-stats-samples", type=int, default=32,
                   help="Calibration samples (of worker 0) used for --layer-stats.")
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Skip Phases 1-5 and run only Phase 6+7 against an existing "
                        "checkpoint dir base. Expects {BASE}_ckpt_shard0..N-1 to exist on "
                        "the nodes that ran them (created by a previous Phase 5.5).")
    return p.parse_args()


# ============================================================
# Shared helpers (also used by export_smoke_test.py)
# ============================================================
def load_text_config(model_path):
    """Return (full_config, text_config). For plain decoder models both are the
    same object; for VL composites the text config is the nested one."""
    full = AutoConfig.from_pretrained(model_path)
    return full, full.get_text_config()


def model_from_config(cfg):
    return AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)


def read_weight_map(model_path):
    with open(Path(model_path) / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def detect_text_prefix(weight_map):
    """Checkpoint prefix of the text stack: 'model.' for plain decoders,
    'model.language_model.' for Qwen3.5-style VL checkpoints."""
    if any(k.startswith("model.language_model.") for k in weight_map):
        return "model.language_model."
    return "model."


def src_to_module_key(key, text_prefix):
    """Map a checkpoint key into the text-only CausalLM module tree
    (model.* / lm_head.*). Returns None for keys outside the text model
    (vision tower, MTP head, ...), which are carried over verbatim."""
    if key.startswith(text_prefix):
        return "model." + key[len(text_prefix):]
    if key.startswith("lm_head."):
        return key
    return None


def module_to_out_key(key, text_prefix):
    if key.startswith("model."):
        return text_prefix + key[len("model."):]
    return key


def build_quant_cfg(text_cfg, extra_disable=()):
    cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    patterns = list(ARCH_DISABLE_PATTERNS.get(getattr(text_cfg, "model_type", ""), []))
    patterns += [p for p in extra_disable if p not in patterns]
    for pat in patterns:
        cfg["quant_cfg"][pat] = {"enable": False}
    return cfg, patterns


def build_ignore_list(tensors):
    """modelopt-style ignore list from the final tensor set.

    Every 2-D `.weight` without a `.weight_scale` sibling is an unquantized
    Linear. Whole unquantized subtrees collapse to `prefix*` (modelopt's own
    convention, which vLLM expands to `prefix` + `prefix.*`); unquantized
    leaves inside partly-quantized blocks (routers, shared_expert_gate) are
    listed by exact name."""
    quantized = {k[:-len(".weight_scale")] for k in tensors if k.endswith(".weight_scale")}
    linear = set()
    for k, t in tensors.items():
        if k.endswith(".weight") and t.dim() == 2 and "embed_tokens" not in k:
            linear.add(k[:-len(".weight")])
    quant_ancestors = set()
    for q in quantized:
        parts = q.split(".")
        for i in range(1, len(parts)):
            quant_ancestors.add(".".join(parts[:i]))
    out = set()
    for mod in sorted(linear - quantized):
        parts = mod.split(".")
        chosen = mod
        for i in range(1, len(parts)):
            anc = ".".join(parts[:i])
            if anc not in quant_ancestors:
                chosen = anc + "*"
                break
        out.add(chosen)
    return sorted(out)


def export_base_config(text_cfg):
    """Config of the 1-layer export template: vocab_size=2 (smallest value
    modelopt's dummy forward accepts), no special token ids (padding_idx must
    fit the tiny vocab), use_cache=False (a DynamicCache would index the
    layer's original layer_idx against a 1-slot cache)."""
    base = copy.deepcopy(text_cfg)
    base.num_hidden_layers = 1
    base.vocab_size = 2
    base.pad_token_id = base.bos_token_id = base.eos_token_id = None
    base.use_cache = False
    return base


def build_export_template(base_cfg, layer_type, device):
    """1-layer CausalLM with tiny zeroed embed/norm/lm_head dummies on `device`.
    Slot 0 (`model.layers[0]`) is still on meta — the caller fills it."""
    cfg_t = copy.deepcopy(base_cfg)
    if layer_type is not None:
        cfg_t.layer_types = [layer_type]
    with init_empty_weights():
        m = model_from_config(cfg_t)
    # Nested VL text configs carry architectures=None; modelopt's exporter
    # iterates over it (is_multimodal_model).
    if not getattr(m.config, "architectures", None):
        m.config.architectures = [type(m).__name__]
    eps = getattr(cfg_t, "rms_norm_eps", None)
    if eps is None:
        eps = getattr(cfg_t, "layer_norm_eps", 1e-5)  # Cohere2 naming
    de = nn.Embedding(cfg_t.vocab_size, cfg_t.hidden_size).to(device, dtype=torch.bfloat16)
    dl = nn.Linear(cfg_t.hidden_size, cfg_t.vocab_size, bias=False).to(device, dtype=torch.bfloat16)
    dn = type(m.model.norm)(cfg_t.hidden_size, eps=eps).to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        for p in (*de.parameters(), *dl.parameters(), *dn.parameters()):
            p.zero_()
    m.model.embed_tokens, m.model.norm, m.lm_head = de, dn, dl
    if getattr(m.model, "rotary_emb", None) is not None:
        m.model.rotary_emb = type(m.model.rotary_emb)(cfg_t).to(device)
    # Cohere2's generation_config sets cache_implementation=hybrid, which
    # transformers rejects together with use_cache=False.
    if getattr(m, "generation_config", None) is not None:
        m.generation_config.cache_implementation = None
    return m


def export_layer_tensors(m_one, layer_index, tmp_dir):
    """Run modelopt's HF exporter on a 1-layer template and return
    (tensors of the layer renamed to `model.layers.{layer_index}.*`,
     {config.json, hf_quant_config.json} texts)."""
    from safetensors import safe_open
    shutil.rmtree(tmp_dir, ignore_errors=True)
    mte.export_hf_checkpoint(m_one, dtype=torch.bfloat16, export_dir=tmp_dir)
    layer_sd, aux = {}, {}
    for fn in sorted(os.listdir(tmp_dir)):
        if fn.endswith(".safetensors"):
            with safe_open(f"{tmp_dir}/{fn}", framework="pt") as sf:
                for k in sf.keys():
                    if k.startswith("model.layers.0."):
                        layer_sd[k.replace("model.layers.0.", f"model.layers.{layer_index}.", 1)] = sf.get_tensor(k)
        elif fn in ("config.json", "hf_quant_config.json"):
            with open(f"{tmp_dir}/{fn}") as f:
                aux[fn] = f.read()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if not layer_sd:
        raise RuntimeError(f"export of layer {layer_index} produced no layer tensors")
    return layer_sd, aux


def causal_mask(seq_len, dtype, device):
    m = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    return torch.triu(m, diagonal=1)[None, None, :, :]


def finish_and_check(wrapper, name):
    """Close modelopt stats collection (amax <- calibrator), sync MoE expert
    amax, fail if an active static quantizer ended up without amax."""
    from modelopt.torch.quantization.model_calib import finish_stats_collection
    finish_stats_collection(wrapper)
    sync_moe_and_check(wrapper, name)


def sync_moe_and_check(wrapper, name):
    from modelopt.torch.quantization.nn import TensorQuantizer
    for m in wrapper.modules():
        if hasattr(m, "layer_sync_moe_local_experts_amax"):
            m.layer_sync_moe_local_experts_amax()
    missing = [n for n, m in wrapper.named_modules()
               if isinstance(m, TensorQuantizer) and not m._disabled
               and not getattr(m, "_dynamic", False)
               and getattr(m, "_amax", None) is None]
    if missing:
        raise RuntimeError(f"[{name}] {len(missing)} active quantizers have no amax "
                           f"after calibration (too few samples / unrouted experts?), "
                           f"e.g. {missing[:8]}")


def amax_stats(wrapper):
    from modelopt.torch.quantization.nn import TensorQuantizer
    stats = {"weight": [0, 0, 0, 0], "input": [0, 0, 0, 0]}  # good, zero, nan, none
    samples = []
    for name, m in wrapper.named_modules():
        if not isinstance(m, TensorQuantizer) or m._disabled:
            continue
        kind = "input" if name.endswith("input_quantizer") else "weight"
        amax = getattr(m, "_amax", None)
        if amax is None:
            stats[kind][3] += 1
        elif torch.isnan(amax).any():
            stats[kind][2] += 1
        elif (amax == 0).all():
            stats[kind][1] += 1
        else:
            stats[kind][0] += 1
            if kind == "input" and len(samples) < 3:
                samples.append((name, round(amax.flatten()[0].item(), 4)))
    fmt = lambda s: f"good={s[0]} zero={s[1]} nan={s[2]} none={s[3]}"
    return f"weight[{fmt(stats['weight'])}] input[{fmt(stats['input'])}] samples={samples}"


def _median(xs):
    xs = sorted(x for x in xs if x is not None and x == x)
    return xs[len(xs) // 2] if xs else float("nan")


def format_layer_stats(st):
    """One log line for --layer-stats of one layer."""
    parts = [f"{k} {st[k]:.4f}" for k in ("layer", "attn", "mlp", "shared") if k in st]
    parts.append(f"tok p99 {st['token_err_p99']:.3f} max {st['token_err_max']:.2f}")
    ex = [(v[0], int(e), v[1]) for e, v in st.get("experts", {}).items() if v[0] is not None]
    if ex:
        worst = max(ex)
        unrouted = sum(1 for v in st["experts"].values() if v[1] == 0)
        parts.append(f"experts med {_median([x[0] for x in ex]):.4f} worst e{worst[1]} "
                     f"{worst[0]:.4f} ({worst[2]} tok), unrouted {unrouted}")
    if st.get("route_mismatch"):
        parts.append(f"ROUTE MISMATCH {st['route_mismatch']}")
    return ", ".join(parts)


def report_layer_outliers(layer_stats, layer_factor=1.5, expert_factor=2.0, min_tokens=32):
    """Summary of --layer-stats: per-component median over layers and every layer
    above layer_factor x median; experts (>= min_tokens routed tokens) above
    expert_factor x the median of all measured experts."""
    lines = ["\n=== Layer stats (relative L2 error vs. BF16, per layer on BF16 inputs) ==="]
    keys = [k for k in ("layer", "attn", "mlp", "shared") if any(k in s for s in layer_stats.values())]
    for k in keys:
        vals = {i: s[k] for i, s in layer_stats.items() if k in s}
        med = _median(vals.values())
        hi = sorted(((v, i) for i, v in vals.items() if v > layer_factor * med), reverse=True)
        lines.append(f"  {k:6s} median {med:.4f}, max {max(vals.values()):.4f} (layer "
                     f"{max(vals, key=vals.get)}); > {layer_factor}x median: "
                     + (", ".join(f"L{i} {v:.4f}" for v, i in hi[:10]) or "none"))
    ex = [(v[0], i, int(e), v[1]) for i, s in layer_stats.items()
          for e, v in s.get("experts", {}).items() if v[0] is not None and v[1] >= min_tokens]
    if ex:
        med = _median([x[0] for x in ex])
        hi = sorted((x for x in ex if x[0] > expert_factor * med), reverse=True)
        n_unrouted = sum(1 for s in layer_stats.values() for v in s.get("experts", {}).values() if v[1] == 0)
        n_few = sum(1 for s in layer_stats.values() for v in s.get("experts", {}).values()
                    if 0 < v[1] < min_tokens)
        lines.append(f"  experts: {len(ex)} measured (>= {min_tokens} tok), median {med:.4f}, "
                     f"{len(hi)} above {expert_factor}x median; {n_few} with < {min_tokens} tok, "
                     f"{n_unrouted} unrouted in the stats samples")
        for v, i, e, t in hi[:15]:
            lines.append(f"    L{i} e{e}: {v:.4f} ({v / med:.1f}x, {t} tok)")
    return "\n".join(lines)


AUX_FILES = (
    "generation_config.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
    "added_tokens.json", "vocab.json", "merges.txt",
    "chat_template.jinja", "chat_template.json",
    "preprocessor_config.json", "video_preprocessor_config.json", "processor_config.json",
)


# ============================================================
# Shard: one contiguous slice of the model
# ============================================================
class ModelShardImpl:
    """Plain class so export_smoke_test.py can run the exact same code
    in-process; the Ray actor is `ModelShard` below."""

    def __init__(self, model_path, layer_start, layer_end, has_embed, has_head, name,
                 shard_index=0, n_shards=1, quant_opts=None):
        self.model_path = model_path
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.has_embed = has_embed
        self.has_head = has_head
        self.name = name
        self.shard_index = shard_index
        self.n_shards = n_shards
        self.quant_opts = quant_opts or {}
        self.device = "cuda:0"

    def _init_config(self):
        self.hf_config, self.text_cfg = load_text_config(self.model_path)
        self.num_layers_total = self.text_cfg.num_hidden_layers
        self.layer_types = getattr(self.text_cfg, "layer_types", None)
        self.weight_map = read_weight_map(self.model_path)
        self.text_prefix = detect_text_prefix(self.weight_map)

    def _introspect_classes(self, model):
        a_layer = model.model.layers[self.layer_start]
        self.rms_norm_cls = type(a_layer.input_layernorm)
        rotary = getattr(model.model, "rotary_emb", None)
        if rotary is None:
            attn = getattr(a_layer, "self_attn", None)
            rotary = getattr(attn, "rotary_emb", None) if attn is not None else None
        self.rotary_emb_cls = type(rotary) if rotary is not None else None

    def _owns(self, module_key):
        if module_key.startswith("model.layers."):
            return self.layer_start <= int(module_key.split(".")[2]) < self.layer_end
        if module_key.startswith("model.embed_tokens."):
            return self.has_embed
        if module_key.startswith("model.norm.") or module_key.startswith("lm_head."):
            return self.has_head
        return False

    # ----- Phase 1: load model parts -----
    def load_model(self):
        from safetensors import safe_open
        self._init_config()
        with init_empty_weights():
            self.model = model_from_config(self.text_cfg)

        wm, tp = self.weight_map, self.text_prefix
        embed_src = tp + "embed_tokens.weight"
        # Tied embeddings (e.g. Cohere2): no lm_head.weight in the checkpoint —
        # the output projection reuses embed_tokens.weight.
        self.tied_lm_head = "lm_head.weight" not in wm

        needed = []  # (checkpoint key, module key)
        for key in wm:
            mk = src_to_module_key(key, tp)
            if mk is None or "rotary_emb" in mk:
                continue  # non-text weights / rotary buffers (rebuilt on device)
            if self._owns(mk):
                needed.append((key, mk))
        # Tied-embedding has_head shard without embed: route embed_tokens.weight
        # straight into lm_head.weight (avoids double-storing the embedding).
        if self.has_head and self.tied_lm_head and not self.has_embed and embed_src in wm:
            needed.append((embed_src, "lm_head.weight"))

        by_file = {}
        for key, mk in needed:
            by_file.setdefault(wm[key], []).append((key, mk))

        print(f"[{self.name}] loading {len(needed)} tensors from {len(by_file)} files", flush=True)
        # Per-tensor streaming via safe_open: load_file kept the entire shard
        # (~4 GB) in RAM until all tensors were materialized, which on
        # Behemoth-class models pushes peak UMA over the 119 GB budget and
        # triggers a silent OOM-kill of the Ray worker.
        for fname, pairs in sorted(by_file.items()):
            with safe_open(str(Path(self.model_path) / fname), framework="pt") as sf:
                for key, mk in pairs:
                    t = sf.get_tensor(key).to(torch.bfloat16)
                    set_module_tensor_to_device(self.model, mk, self.device, value=t)
                    del t
            gc.collect()

        # Tied embeddings, single-shard case: mirror embed into lm_head.
        if self.has_head and self.tied_lm_head and self.has_embed:
            set_module_tensor_to_device(self.model, "lm_head.weight", self.device,
                                        value=self.model.model.embed_tokens.weight.data)

        # Fail loudly if any owned parameter is still on meta (key mismatch).
        left = [n for n, p in self.model.named_parameters()
                if p.device.type == "meta" and self._owns(n)]
        if left:
            raise RuntimeError(f"[{self.name}] {len(left)} owned params not loaded "
                               f"(checkpoint prefix {tp!r}), e.g. {left[:5]}")

        self.embed_tokens = self.model.model.embed_tokens if self.has_embed else None
        self.local_layers = self.model.model.layers  # full list, only [start:end] populated
        self.norm = self.model.model.norm if self.has_head else None
        self.lm_head = self.model.lm_head if self.has_head else None
        self._introspect_classes(self.model)

        torch.cuda.empty_cache()
        return (f"{self.name}: loaded layers {self.layer_start}..{self.layer_end} "
                f"({self.text_cfg.model_type}, prefix {tp!r}), "
                f"{torch.cuda.memory_allocated()/1e9:.1f} GiB on GPU")

    # ----- Phase 2: insert NVFP4 quantizers, open stats collection -----
    def apply_quantization(self):
        """Insert quantizers into our local layers (and lm_head if owned), then
        put them into stats-collection mode for Phase 3.

        mtq.quantize(..., forward_loop=None) runs modelopt's max_calibrate in
        weight-only mode, which ends with finish_stats_collection() — i.e.
        calibration OFF. Without re-enabling it the Phase-3 forwards collect
        nothing and no input_scale is ever exported."""
        from modelopt.torch.quantization.model_calib import enable_stats_collection
        from modelopt.torch.quantization.nn import TensorQuantizer

        class LocalQuantizable(nn.Module):
            pass
        wrapper = LocalQuantizable()
        wrapper.layers = nn.ModuleList(
            [self.local_layers[i] for i in range(self.layer_start, self.layer_end)]
        )
        if self.has_head:
            wrapper.lm_head = self.lm_head

        cfg, patterns = build_quant_cfg(self.text_cfg, self.quant_opts.get("disable", []))
        mtq.quantize(wrapper, cfg, forward_loop=None)

        ratio = self.quant_opts.get("moe_calib_ratio")
        n_moe = 0
        for m in wrapper.modules():
            if hasattr(m, "_moe_calib_experts_ratio"):
                m._moe_calib_experts_ratio = ratio
                n_moe += 1

        enable_stats_collection(wrapper)
        self._quant_wrapper = wrapper
        self._run_lm_head = self.has_head and any(
            isinstance(m, TensorQuantizer) and not m._disabled for m in self.lm_head.modules())

        n_q = sum(1 for m in wrapper.modules() if isinstance(m, TensorQuantizer))
        n_on = sum(1 for m in wrapper.modules()
                   if isinstance(m, TensorQuantizer) and not m._disabled)
        return (f"{self.name}: {n_q} quantizers ({n_on} active), BF16 patterns={patterns}, "
                f"moe_blocks={n_moe} (calib_ratio={ratio if n_moe else '-'})")

    # ----- Phase 3: calibration forward through this shard -----
    def forward_shard(self, x, position_ids=None):
        """Shard 0 takes input_ids, later shards take hidden states. Returns
        (hidden, position_ids) on CPU for the next shard, or (None, None) on
        the last shard."""
        with torch.no_grad():
            if self.has_embed:
                input_ids = x.to(self.device)
                position_ids = torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0)
                hidden = self.embed_tokens(input_ids)
            else:
                hidden = x.to(self.device)
                position_ids = position_ids.to(self.device)

            pe = None
            if self.rotary_emb_cls is not None:
                if not hasattr(self, "_rotary_cuda"):
                    self._rotary_cuda = self.rotary_emb_cls(self.text_cfg).to(self.device)
                pe = self._rotary_cuda(hidden, position_ids)
            causal = causal_mask(hidden.shape[1], hidden.dtype, self.device)

            for i in range(self.layer_start, self.layer_end):
                # Linear-attention (Gated DeltaNet) layers take a 2-D padding
                # mask or None — never the 4-D additive causal mask.
                lt = self.layer_types[i] if self.layer_types else None
                hidden = self.local_layers[i](
                    hidden,
                    position_embeddings=pe,
                    attention_mask=None if lt == "linear_attention" else causal,
                )

            if self.has_head:
                if self._run_lm_head:
                    _ = self.lm_head(self.norm(hidden))
                return None, None
            return hidden.cpu(), position_ids.cpu()

    # ----- Phase 4: close stats collection, verify every quantizer got amax -----
    def finalize(self):
        finish_and_check(self._quant_wrapper, self.name)
        return f"{self.name}: calibration finalized"

    # ----- Phase 5: diagnostics -----
    def export_status(self):
        return f"{self.name}: {amax_stats(self._quant_wrapper)}"

    # ----- Resume: rebuild minimal state from a Phase-5.5 checkpoint -----
    def setup_for_resume(self, ckpt_dir):
        with open(f"{ckpt_dir}/meta.json") as f:
            meta = json.load(f)
        self.model_path = meta["model_path"]
        self.layer_start = meta["layer_start"]
        self.layer_end = meta["layer_end"]
        self.has_embed = meta["has_embed"]
        self.has_head = meta["has_head"]
        self.name = meta["name"]
        self._init_config()
        with init_empty_weights():
            m = model_from_config(self.text_cfg)
        self._introspect_classes(m)
        del m
        self.local_layers = [nn.Identity() for _ in range(self.num_layers_total)]
        self.embed_tokens = self.norm = self.lm_head = None
        return f"{self.name}: resumed from {ckpt_dir}, layers {self.layer_start}..{self.layer_end}"

    # ----- Phase 5.5: evict layers to disk so Phase 6 can stream them back -----
    def save_layers_to_disk(self, ckpt_dir):
        """torch.save each owned (quantized, calibrated) layer with cloudpickle
        (modelopt's QuantLinear is a dynamically generated subclass vanilla
        pickle can't serialize), replacing every in-memory reference so ~95 %
        of UMA is freed before Phase 6."""
        import cloudpickle
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        os.makedirs(ckpt_dir)

        if hasattr(self, "_quant_wrapper"):
            del self._quant_wrapper
        if hasattr(self, "_rotary_cuda"):
            del self._rotary_cuda

        n = 0
        for i in range(self.layer_start, self.layer_end):
            torch.save(self.local_layers[i].cpu(), f"{ckpt_dir}/layer_{i:04d}.pt",
                       pickle_module=cloudpickle)
            self.local_layers[i] = nn.Identity()
            # GB10 UMA: the .cpu() copy doubles the layer in the shared pool;
            # free it immediately or Ray's memory monitor kills the worker.
            gc.collect()
            torch.cuda.empty_cache()
            n += 1

        extras = []
        if self.has_embed:
            torch.save(self.embed_tokens.cpu(), f"{ckpt_dir}/embed_tokens.pt", pickle_module=cloudpickle)
            self.model.model.embed_tokens = nn.Identity()
            self.embed_tokens = None
            extras.append("embed_tokens")
        if self.has_head:
            torch.save(self.norm.cpu(), f"{ckpt_dir}/norm.pt", pickle_module=cloudpickle)
            torch.save(self.lm_head.cpu(), f"{ckpt_dir}/lm_head.pt", pickle_module=cloudpickle)
            self.model.model.norm = nn.Identity()
            self.model.lm_head = nn.Identity()
            self.norm = self.lm_head = None
            extras += ["norm", "lm_head"]

        meta = {
            "model_path": self.model_path,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "has_embed": self.has_embed,
            "has_head": self.has_head,
            "name": self.name,
            "shard_index": self.shard_index,
            "n_shards": self.n_shards,
            "num_layers_total": self.num_layers_total,
        }
        with open(f"{ckpt_dir}/meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        gc.collect()
        torch.cuda.empty_cache()
        mem = torch.cuda.memory_allocated() / 1e9
        return f"{self.name}: evicted {n} layers + {extras} to {ckpt_dir}, GPU mem now {mem:.1f}G"

    # ----- Phase 6: per-layer streaming export -----
    def export_shard(self, output_dir, ckpt_dir):
        """Export each owned layer through modelopt's HF exporter inside a
        1-layer template with tiny (vocab=2) dummies; keep only the layer's
        tensors, renamed to their global index (`model.layers.{i}.*`). Real
        embed/norm/lm_head go to a BF16 'special' shard. Peak memory is ~1
        layer + export overhead regardless of model size."""
        from safetensors.torch import save_file

        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir)
        if hasattr(self, "_quant_wrapper"):
            del self._quant_wrapper
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{self.name}] start export, mem={torch.cuda.memory_allocated()/1e9:.1f}G", flush=True)

        base_cfg = export_base_config(self.text_cfg)
        n_local = self.layer_end - self.layer_start
        n_keys, total_size = 0, 0
        aux = {}

        for idx_in, orig_i in enumerate(range(self.layer_start, self.layer_end)):
            m_one = build_export_template(
                base_cfg, self.layer_types[orig_i] if self.layer_types else None, self.device)
            layer = torch.load(f"{ckpt_dir}/layer_{orig_i:04d}.pt",
                               map_location=self.device, weights_only=False).to(self.device)
            m_one.model.layers[0] = layer
            # The layer keeps its original index (e.g. 50); the template only
            # has slot 0.
            for sub in (layer, getattr(layer, "self_attn", None), getattr(layer, "linear_attn", None)):
                if sub is not None and hasattr(sub, "layer_idx"):
                    sub.layer_idx = 0

            layer_sd, layer_aux = export_layer_tensors(m_one, orig_i, f"{output_dir}/_tmp_l{orig_i:04d}")
            if idx_in == 0:
                aux = layer_aux
            save_file(layer_sd, f"{output_dir}/shard_l{orig_i:04d}.safetensors")
            n_keys += len(layer_sd)
            total_size += sum(t.numel() * t.element_size() for t in layer_sd.values())

            del layer_sd, m_one, layer
            gc.collect()
            torch.cuda.empty_cache()
            if (idx_in + 1) % 5 == 0 or idx_in == n_local - 1:
                print(f"  [{self.name}] {idx_in + 1}/{n_local} layers "
                      f"(mem={torch.cuda.memory_allocated()/1e9:.1f}G)", flush=True)

        # Real embed/norm/lm_head as BF16 (evicted to disk in Phase 5.5).
        def _load(file_name):
            return torch.load(f"{ckpt_dir}/{file_name}.pt", map_location="cpu", weights_only=False)

        special = {}
        if self.has_embed:
            special["model.embed_tokens.weight"] = _load("embed_tokens").weight.detach().to(torch.bfloat16)
        if self.has_head:
            special["model.norm.weight"] = _load("norm").weight.detach().to(torch.bfloat16)
            special["lm_head.weight"] = _load("lm_head").weight.detach().to(torch.bfloat16)
        if special:
            save_file(special, f"{output_dir}/shard_special.safetensors")
            n_keys += len(special)
            total_size += sum(t.numel() * t.element_size() for t in special.values())
            print(f"  [{self.name}] saved special shard: {list(special)}", flush=True)

        for fn, content in aux.items():
            with open(f"{output_dir}/{fn}", "w") as f:
                f.write(content)
        meta = {"name": self.name, "layer_start": self.layer_start, "layer_end": self.layer_end,
                "has_embed": self.has_embed, "has_head": self.has_head}
        with open(f"{output_dir}/export_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        if hasattr(self, "model"):
            del self.model
        gc.collect()
        torch.cuda.empty_cache()
        return (f"{self.name}: exported {n_local} layers + special, {n_keys} keys, "
                f"{total_size/1e9:.1f}G, end mem={torch.cuda.memory_allocated()/1e9:.1f}G")


ModelShard = ray.remote(num_gpus=1)(ModelShardImpl)


# ============================================================
# Sequential mode: layer by layer, calibration samples split across workers
# ============================================================
class SeqWorkerImpl:
    """One worker of the sequential mode. Every worker holds the hidden states
    of its share of the calibration samples (node-local, a few GB at most)
    and, per step, exactly one decoder layer:

        calib_layer(i)  load layer i (BF16) into a 1-layer template, insert
                        quantizers, push the local samples through it with
                        stats collection on; the outputs become the inputs
                        of layer i+1. Returns this worker's amax per quantizer.
        export_layer(i) rank 0 only: apply the driver's element-wise max over
                        all workers (exact for max calibration) and export.
        commit(i)       persist hidden states -> resumable after a crash.

    Peak memory per worker ~ one layer + hidden states, independent of model
    size, so 1 node handles any size and N nodes divide the calibration time."""

    def __init__(self, model_path, rank, n_workers, temp_base, quant_opts):
        self.model_path = model_path
        self.rank = rank
        self.n_workers = n_workers
        self.work_dir = f"{temp_base}_seq_w{rank}"
        self.export_dir = f"{temp_base}_seq_export"
        self.quant_opts = quant_opts or {}
        self.device = "cuda:0"
        self.name = f"seq{rank}"
        self.hf_config, self.text_cfg = load_text_config(model_path)
        self.num_layers = self.text_cfg.num_hidden_layers
        self.layer_types = getattr(self.text_cfg, "layer_types", None)
        self.weight_map = read_weight_map(model_path)
        self.text_prefix = detect_text_prefix(self.weight_map)
        self.base_cfg = export_base_config(self.text_cfg)
        self._pending = None
        self.hidden = None

    def _read_tensor(self, key):
        from safetensors import safe_open
        with safe_open(str(Path(self.model_path) / self.weight_map[key]), framework="pt") as sf:
            return sf.get_tensor(key)

    def probe(self, sample_key):
        """Return (rank, node ip, last committed layer or -1, export ok up to it)."""
        ip = ray.util.get_node_ip_address()
        try:
            with open(f"{self.work_dir}/state.json") as f:
                st = json.load(f)
        except FileNotFoundError:
            return self.rank, ip, -1, False
        done = st["done"] if st.get("sample_key") == sample_key else -1
        export_ok = True
        if self.rank == 0 and done >= 0:
            export_ok = os.path.exists(f"{self.export_dir}/config.json") and all(
                os.path.exists(f"{self.export_dir}/shard_l{i:04d}.safetensors") for i in range(done + 1))
        return self.rank, ip, done, export_ok

    def setup(self, sample_ids, sample_key, resume_done):
        """Fresh start: embed the local samples. Resume: reload hidden states."""
        self.sample_key = sample_key
        if resume_done >= 0:
            st = torch.load(f"{self.work_dir}/state.pt", map_location=self.device, weights_only=False)
            assert st["done"] == resume_done and st["sample_key"] == sample_key
            self.hidden = st["hidden"]
            return f"{self.name}: resumed after layer {resume_done}, {len(self.hidden)} samples"
        shutil.rmtree(self.work_dir, ignore_errors=True)
        os.makedirs(self.work_dir)
        if self.rank == 0:
            shutil.rmtree(self.export_dir, ignore_errors=True)
            os.makedirs(self.export_dir)
        w = self._read_tensor(self.text_prefix + "embed_tokens.weight").to(self.device, torch.bfloat16)
        with torch.no_grad():
            self.hidden = [w[torch.tensor(ids, device=self.device)].unsqueeze(0) for ids in sample_ids]
        del w
        gc.collect()
        torch.cuda.empty_cache()
        n_tok = sum(h.shape[1] for h in self.hidden)
        return (f"{self.name}: {len(self.hidden)} samples / {n_tok} tokens embedded, "
                f"hidden states {n_tok * self.text_cfg.hidden_size * 2 / 1e9:.2f} GB")

    def calib_layer(self, i):
        from modelopt.torch.quantization.model_calib import enable_stats_collection, finish_stats_collection
        from modelopt.torch.quantization.nn import TensorQuantizer
        t0 = time.time()
        lt = self.layer_types[i] if self.layer_types else None
        m_one = build_export_template(self.base_cfg, lt, self.device)
        prefix = f"{self.text_prefix}layers.{i}."
        keys = [k for k in self.weight_map if k.startswith(prefix)]
        if not keys:
            raise RuntimeError(f"[{self.name}] no checkpoint tensors for layer {i} ({prefix}*)")
        for k in keys:
            mk = "model.layers.0." + k[len(prefix):]
            set_module_tensor_to_device(m_one, mk, self.device,
                                        value=self._read_tensor(k).to(torch.bfloat16))
        layer = m_one.model.layers[0]
        left = [n for n, p in layer.named_parameters() if p.device.type == "meta"]
        if left:
            raise RuntimeError(f"[{self.name}] layer {i}: {len(left)} params not loaded, e.g. {left[:5]}")
        t_load = time.time() - t0

        class LocalQuantizable(nn.Module):
            pass
        wrapper = LocalQuantizable()
        wrapper.layers = nn.ModuleList([layer])
        cfg, _ = build_quant_cfg(self.text_cfg, self.quant_opts.get("disable", []))
        mtq.quantize(wrapper, cfg, forward_loop=None)
        ratio = self.quant_opts.get("moe_calib_ratio")
        for m in wrapper.modules():
            if hasattr(m, "_moe_calib_experts_ratio"):
                m._moe_calib_experts_ratio = ratio
        enable_stats_collection(wrapper)

        rotary = getattr(m_one.model, "rotary_emb", None)
        new_hidden = []
        with torch.no_grad():
            for h in self.hidden:
                T = h.shape[1]
                pos = torch.arange(T, device=self.device).unsqueeze(0)
                pe = rotary(h, pos) if rotary is not None else None
                mask = None if lt == "linear_attention" else causal_mask(T, h.dtype, self.device)
                out = layer(h, position_embeddings=pe, attention_mask=mask)
                new_hidden.append(out[0] if isinstance(out, tuple) else out)
        finish_stats_collection(wrapper)
        self._pending = (i, m_one, wrapper, new_hidden)

        amax = {n: m._amax.detach().float().cpu() for n, m in wrapper.named_modules()
                if isinstance(m, TensorQuantizer) and getattr(m, "_amax", None) is not None}
        return amax, {"load_s": t_load, "calib_s": time.time() - t0 - t_load}

    def export_layer(self, i, merged_amax):
        from modelopt.torch.quantization.nn import TensorQuantizer
        from safetensors.torch import save_file
        pi, m_one, wrapper, _ = self._pending
        assert pi == i
        t0 = time.time()
        for n, m in wrapper.named_modules():
            if isinstance(m, TensorQuantizer) and n in merged_amax:
                m.amax = merged_amax[n].to(self.device)
        sync_moe_and_check(wrapper, f"{self.name} layer {i}")
        stats = amax_stats(wrapper)
        layer_sd, aux = export_layer_tensors(m_one, i, f"{self.export_dir}/_tmp_l{i:04d}")
        save_file(layer_sd, f"{self.export_dir}/shard_l{i:04d}.safetensors")
        if not os.path.exists(f"{self.export_dir}/config.json"):
            for fn, content in aux.items():
                with open(f"{self.export_dir}/{fn}", "w") as f:
                    f.write(content)
        size = sum(t.numel() * t.element_size() for t in layer_sd.values())
        return {"export_s": time.time() - t0, "bytes": size, "stats": stats}

    def layer_stats(self, i, merged_amax, act_format, n_samples):
        """Quantization error of layer i on its BF16 inputs. Call BEFORE export_layer
        (the export packs the weights to FP4 in place); applies the same merged amax
        and MoE amax sync as the export. Reference: every quantizer disabled.
        Candidate: the quantizers active at serve time (w4a16: weight quantizers only).
        Reported as relative L2 error sqrt(sum|q - ref|^2 / sum|ref|^2):
          layer   whole layer, residual removed (|out_q - out_ref| / |out_ref - h|);
                  includes routing changes caused by the quantized attention
          attn    attention block alone, on the reference inputs
          mlp     MLP block alone, on the reference inputs (same routing)
          shared  MoE shared expert alone
          experts per routed expert: [rel err, tokens routed to it]"""
        from modelopt.torch.quantization.nn import TensorQuantizer
        pi, m_one, wrapper, _ = self._pending
        assert pi == i
        t0 = time.time()
        for n, m in wrapper.named_modules():
            if isinstance(m, TensorQuantizer) and n in merged_amax:
                m.amax = merged_amax[n].to(self.device)
        sync_moe_and_check(wrapper, f"{self.name} layer {i} (stats)")
        layer = wrapper.layers[0]
        lt = self.layer_types[i] if self.layer_types else None
        qs = [(n, m) for n, m in wrapper.named_modules() if isinstance(m, TensorQuantizer)]
        orig = {n: m.is_enabled for n, m in qs}
        serve = {n: orig[n] and (act_format == "w4a4" or n.endswith("weight_quantizer")) for n, _ in qs}

        # Fake-quantize every served weight once and swap it in, instead of letting
        # modelopt re-quantize all expert weights on every forward.
        wq = {}
        with torch.no_grad():
            for n, m in qs:
                if not (serve[n] and n.endswith("weight_quantizer")):
                    continue
                parent = wrapper.get_submodule(n.rsplit(".", 1)[0])
                w = getattr(parent, "weight", None)
                if isinstance(w, torch.Tensor) and w.device.type != "meta":
                    m.enable()
                    wq[n] = (parent, w.data, m(w).detach())

        def set_quant(on):
            for n, m in qs:
                if n in wq:
                    parent, w_bf16, w_q = wq[n]
                    parent.weight.data = w_q if on else w_bf16
                    m.disable()
                else:
                    m.enable() if (on and serve[n]) else m.disable()

        attn = getattr(layer, "self_attn", None) or getattr(layer, "linear_attn", None)
        attn_quantized = attn is not None and any(
            serve[n] for n, _ in qs if ".self_attn." in n or ".linear_attn." in n)
        mlp = getattr(layer, "mlp", None)
        shared = getattr(mlp, "shared_expert", None) if mlp is not None else None
        experts = {}
        for n, m in layer.named_modules():
            parts = n.split(".")
            if len(parts) == 4 and parts[:2] == ["mlp", "experts"] and parts[3] == "down_proj":
                experts[int(parts[2])] = m

        cap = {}
        mode = {"tag": None}
        hooks = []

        def first(o):
            return o[0] if isinstance(o, tuple) else o

        def out_hook(name):
            def h(mod, args, out):
                if mode["tag"] is not None:
                    cap[(mode["tag"], name)] = first(out).detach()
            return h
        if attn is not None:
            hooks.append(attn.register_forward_pre_hook(
                lambda mod, a, kw: cap.__setitem__("attn_in", (a, kw)) if mode["tag"] == "ref" else None,
                with_kwargs=True))
            hooks.append(attn.register_forward_hook(out_hook("attn")))
        if mlp is not None:
            hooks.append(mlp.register_forward_pre_hook(
                lambda mod, a: cap.__setitem__("mlp_in", a) if mode["tag"] == "ref" else None))
            hooks.append(mlp.register_forward_hook(out_hook("mlp")))
        if shared is not None:
            hooks.append(shared.register_forward_hook(out_hook("shared")))
        for e, m in experts.items():
            hooks.append(m.register_forward_hook(out_hook(("expert", e))))

        acc = collections.defaultdict(lambda: [0.0, 0.0, 0])
        tok_err = []
        route_mismatch = 0

        def add(key, q, ref, base=None):
            q, ref = q.float(), ref.float()
            a = acc[key]
            a[0] += (q - ref).pow(2).sum().item()
            a[1] += (ref if base is None else ref - base.float()).pow(2).sum().item()
            a[2] += q.shape[-2] if q.dim() >= 2 else 1

        rotary = getattr(m_one.model, "rotary_emb", None)
        try:
            with torch.no_grad():
                for h in self.hidden[:n_samples]:
                    T = h.shape[1]
                    pos = torch.arange(T, device=self.device).unsqueeze(0)
                    pe = rotary(h, pos) if rotary is not None else None
                    mask = None if lt == "linear_attention" else causal_mask(T, h.dtype, self.device)
                    cap.clear()
                    set_quant(False)
                    mode["tag"] = "ref"
                    out_ref = first(layer(h, position_embeddings=pe, attention_mask=mask))
                    set_quant(True)
                    mode["tag"] = None
                    out_q = first(layer(h, position_embeddings=pe, attention_mask=mask))
                    add("layer", out_q, out_ref, base=h)
                    d_ref = (out_ref - h).float().reshape(-1, out_ref.shape[-1])
                    d_err = (out_q - out_ref).float().reshape(-1, out_ref.shape[-1])
                    tok_err.append((d_err.norm(dim=-1) / d_ref.norm(dim=-1).clamp_min(1e-6)).cpu())
                    mode["tag"] = "q"
                    if attn_quantized and "attn_in" in cap:
                        a, kw = cap["attn_in"]
                        attn(*a, **kw)
                        add("attn", cap[("q", "attn")], cap[("ref", "attn")])
                    if mlp is not None and "mlp_in" in cap:
                        mlp(*cap["mlp_in"])
                        add("mlp", cap[("q", "mlp")], cap[("ref", "mlp")])
                        if shared is not None:
                            add("shared", cap[("q", "shared")], cap[("ref", "shared")])
                        for e in experts:
                            r, q = cap.get(("ref", ("expert", e))), cap.get(("q", ("expert", e)))
                            if r is None and q is None:
                                continue
                            if r is None or q is None or r.shape != q.shape:
                                route_mismatch += 1
                                continue
                            add(("expert", e), q, r)
                    mode["tag"] = None
        finally:
            for hk in hooks:
                hk.remove()
            for n, (parent, w_bf16, _) in wq.items():
                parent.weight.data = w_bf16
            for n, m in qs:
                m.enable() if orig[n] else m.disable()
            cap.clear()
            wq.clear()

        def rel(a):
            return math.sqrt(a[0] / a[1]) if a[1] > 0 else float("nan")
        te = torch.cat(tok_err) if tok_err else torch.zeros(1)
        out = {"layer_type": lt, "samples": min(n_samples, len(self.hidden)),
               "tokens": int(te.numel()), "seconds": round(time.time() - t0, 1),
               "token_err_p50": float(te.quantile(0.5)), "token_err_p99": float(te.quantile(0.99)),
               "token_err_max": float(te.max()), "route_mismatch": route_mismatch}
        for k in ("layer", "attn", "mlp", "shared"):
            if k in acc:
                out[k] = rel(acc[k])
        if experts:
            out["experts"] = {e: [rel(acc[("expert", e)]), acc[("expert", e)][2]]
                              if ("expert", e) in acc else [None, 0] for e in sorted(experts)}
        return out

    def commit(self, i):
        pi, m_one, wrapper, new_hidden = self._pending
        assert pi == i
        self._pending = None
        del m_one, wrapper
        self.hidden = new_hidden
        tmp = f"{self.work_dir}/state.pt.tmp"
        torch.save({"done": i, "sample_key": self.sample_key, "hidden": self.hidden}, tmp)
        os.replace(tmp, f"{self.work_dir}/state.pt")
        with open(f"{self.work_dir}/state.json.tmp", "w") as f:
            json.dump({"done": i, "sample_key": self.sample_key}, f)
        os.replace(f"{self.work_dir}/state.json.tmp", f"{self.work_dir}/state.json")
        gc.collect()
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated() / 1e9

    def write_special(self):
        """Rank 0, after the last layer: BF16 embed/norm/lm_head + export_meta."""
        from safetensors.torch import save_file
        self.hidden = None
        gc.collect()
        torch.cuda.empty_cache()
        tp = self.text_prefix
        special = {
            "model.embed_tokens.weight": self._read_tensor(tp + "embed_tokens.weight").to(torch.bfloat16),
            "model.norm.weight": self._read_tensor(tp + "norm.weight").to(torch.bfloat16),
        }
        # Tied embeddings (e.g. Cohere2): no lm_head.weight in the checkpoint.
        head_key = "lm_head.weight" if "lm_head.weight" in self.weight_map else tp + "embed_tokens.weight"
        special["lm_head.weight"] = self._read_tensor(head_key).to(torch.bfloat16)
        save_file(special, f"{self.export_dir}/shard_special.safetensors")
        with open(f"{self.export_dir}/export_meta.json", "w") as f:
            json.dump({"name": "sequential", "layer_start": 0, "layer_end": self.num_layers,
                       "has_embed": True, "has_head": True}, f, indent=2)
        return f"{self.name}: special shard written ({list(special)})"

    def release(self):
        self.hidden = None
        self._pending = None
        gc.collect()
        torch.cuda.empty_cache()


SeqWorker = ray.remote(num_gpus=1)(SeqWorkerImpl)


@ray.remote(num_cpus=0)
def _scan_checkpoints(base):
    """Runs on one node: report the Phase-5.5 checkpoints that live there."""
    found = []
    for i in range(64):
        d = f"{base}_ckpt_shard{i}"
        if os.path.exists(f"{d}/meta.json"):
            with open(f"{d}/meta.json") as f:
                found.append((i, d, json.load(f)))
    return ray.util.get_node_ip_address(), found


# ============================================================
# Driver-side: gather + merge
# ============================================================
def gather_shard_exports(temp_dirs, shard_ips, ssh_user):
    """Phase 6.5: pull per-shard exports from remote nodes to the driver.
    Always a full rsync --delete, so a stale dir from an earlier run can never
    be merged."""
    driver_ip = ray.util.get_node_ip_address()
    print(f"\n=== Phase 6.5: Gather remote shard exports (driver={driver_ip}) ===", flush=True)
    for i, (d, ip) in enumerate(zip(temp_dirs, shard_ips)):
        if ip == driver_ip:
            print(f"  shard{i}: local", flush=True)
            continue
        os.makedirs(d, exist_ok=True)
        print(f"  shard{i}@{ip}: rsync -> {d}/ ...", flush=True)
        subprocess.run([
            "rsync", "-a", "--delete", "-e", "ssh -o StrictHostKeyChecking=no",
            f"{ssh_user}@{ip}:{d}/", f"{d}/",
        ], check=True)
        print(f"  shard{i}: {len(os.listdir(d))} files after sync", flush=True)


def merge_exports(shard_dirs, layer_counts, final_dir, source_model_path, num_layers_total,
                  act_format="w4a16"):
    """Merge per-shard exports into one HF NVFP4 model dir.

    Shard exports use text-only module names with global layer indices; they
    are renamed to the source checkpoint's layout (e.g. model.language_model.*).
    Source tensors outside the text model (vision tower, MTP head) are copied
    verbatim in BF16. config.json = source config + quantization_config."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    assert len(shard_dirs) == len(layer_counts), "shard_dirs and layer_counts must align"
    assert sum(layer_counts) == num_layers_total, \
        f"layer_counts sum {sum(layer_counts)} != num_layers_total {num_layers_total}"

    src_wm = read_weight_map(source_model_path)
    text_prefix = detect_text_prefix(src_wm)

    merged = {}  # module-space keys
    start = 0
    N = len(shard_dirs)
    for i, (d, n_layers) in enumerate(zip(shard_dirs, layer_counts)):
        end = start + n_layers
        with open(f"{d}/export_meta.json") as f:
            em = json.load(f)
        if (em["layer_start"], em["layer_end"]) != (start, end):
            raise RuntimeError(f"{d} holds layers {em['layer_start']}..{em['layer_end']}, "
                               f"expected {start}..{end} (stale export dir?)")
        files = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
        expected = {f"shard_l{j:04d}.safetensors" for j in range(start, end)}
        if i == 0 or i == N - 1:
            expected.add("shard_special.safetensors")
        if set(files) != expected:
            raise RuntimeError(f"{d}: unexpected file set — missing "
                               f"{sorted(expected - set(files))[:5]}, extra {sorted(set(files) - expected)[:5]}")
        n = 0
        for fn in files:
            with safe_open(os.path.join(d, fn), framework="pt") as sf:
                for k in sf.keys():
                    if k.startswith("model.layers."):
                        li = int(k.split(".")[2])
                        if not start <= li < end:
                            raise RuntimeError(f"{d}/{fn}: layer {li} outside shard range {start}..{end}")
                    if k in merged:
                        raise RuntimeError(f"duplicate tensor {k} in {d}/{fn}")
                    merged[k] = sf.get_tensor(k)
                    n += 1
        print(f"  shard{i}: {n} tensors (layers {start}..{end})", flush=True)
        start = end

    seen = {int(k.split(".")[2]) for k in merged if k.startswith("model.layers.")}
    if seen != set(range(num_layers_total)):
        raise RuntimeError(f"layers missing from merge: {sorted(set(range(num_layers_total)) - seen)[:10]}")
    for k in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        if k not in merged:
            raise RuntimeError(f"{k} missing from merge")

    out = {module_to_out_key(k, text_prefix): v for k, v in merged.items()}
    del merged

    # Non-text source tensors (vision tower, MTP head): verbatim BF16 copy.
    extra = [k for k in src_wm if src_to_module_key(k, text_prefix) is None]
    if extra:
        by_file = {}
        for k in extra:
            by_file.setdefault(src_wm[k], []).append(k)
        for fn, keys in sorted(by_file.items()):
            with safe_open(os.path.join(source_model_path, fn), framework="pt") as sf:
                for k in keys:
                    out[k] = sf.get_tensor(k)
        groups = sorted({k.split(".")[0] + "." + k.split(".")[1] for k in extra})
        print(f"  carried over {len(extra)} non-text tensors verbatim: {groups}", flush=True)

    # input_scale: with working calibration modelopt emits one per quantized
    # Linear. Legacy checkpoints (pre-calibration-fix resume) have none — for
    # those, keep the old vLLM workaround (input_scale=1.0 + dynamic=true).
    quant_mods = [k[:-len(".weight_scale_2")] for k in out if k.endswith(".weight_scale_2")]
    missing = [m for m in quant_mods if m + ".input_scale" not in out]
    legacy_uncalibrated = bool(quant_mods) and len(missing) == len(quant_mods)
    if missing and not legacy_uncalibrated:
        raise RuntimeError(f"{len(missing)}/{len(quant_mods)} quantized modules lack input_scale, "
                           f"e.g. {missing[:5]}")
    if legacy_uncalibrated:
        print("  WARNING: no input_scale anywhere (uncalibrated legacy checkpoint) — "
              "injecting input_scale=1.0 and dynamic=true", flush=True)
        for m in missing:
            out[m + ".input_scale"] = torch.tensor(1.0, dtype=torch.float32)
    print(f"  quantized modules: {len(quant_mods)}, with calibrated input_scale: "
          f"{0 if legacy_uncalibrated else len(quant_mods)}", flush=True)

    ignore = build_ignore_list(out)

    total_bytes = sum(t.numel() * t.element_size() for t in out.values())
    print(f"  Merged total: {len(out)} tensors, {total_bytes / 1e9:.2f} GB", flush=True)

    os.makedirs(final_dir, exist_ok=True)
    for fn in os.listdir(final_dir):  # stale weights from an earlier run into the same dir
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            os.remove(os.path.join(final_dir, fn))

    SHARD_SIZE = 5 * 1024**3
    if total_bytes <= SHARD_SIZE * 1.05:
        save_file(out, f"{final_dir}/model.safetensors")
        print("  Saved single model.safetensors", flush=True)
    else:
        n_files = max(1, (total_bytes + SHARD_SIZE - 1) // SHARD_SIZE)
        files = [{} for _ in range(n_files)]
        sizes = [0] * n_files
        weight_map = {}
        for k in sorted(out):
            v = out[k]
            j = min(range(n_files), key=lambda x: sizes[x])
            files[j][k] = v
            sizes[j] += v.numel() * v.element_size()
            weight_map[k] = f"model-{j+1:05d}-of-{n_files:05d}.safetensors"
        for j, sd in enumerate(files):
            fname = f"model-{j+1:05d}-of-{n_files:05d}.safetensors"
            save_file(sd, f"{final_dir}/{fname}")
            print(f"  Saved {fname} ({sizes[j]/1e9:.2f} GB, {len(sd)} tensors)", flush=True)
        with open(f"{final_dir}/model.safetensors.index.json", "w") as f:
            json.dump({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, f, indent=2)

    # config.json: the source config verbatim + a quantization_config.
    with open(f"{shard_dirs[0]}/config.json") as f:
        qcfg = json.load(f)["quantization_config"]
    with open(f"{source_model_path}/config.json") as f:
        cfg = json.load(f)
    hq_path = f"{shard_dirs[0]}/hf_quant_config.json"
    hqc = None
    if os.path.exists(hq_path):
        with open(hq_path) as f:
            hqc = json.load(f)

    qcfg["ignore"] = ignore
    if legacy_uncalibrated:
        for group in qcfg.get("config_groups", {}).values():
            if isinstance(group, dict) and "input_activations" in group:
                group["input_activations"]["dynamic"] = True
    if hqc is not None and isinstance(hqc.get("quantization"), dict):
        hqc["quantization"]["exclude_modules"] = ignore
    if act_format == "w4a16":
        # Same NVFP4 weights declared W4A16_NVFP4: vLLM keeps activations in
        # BF16 (Marlin FP4 GEMM) instead of quantizing them to FP4 at runtime.
        # Qwen3.6-35B-A3B, top-K KLD vs BF16: 0.110 (NVFP4/W4A4) -> 0.076,
        # decode speed unchanged, prefill -9 %.
        qcfg["quant_algo"] = "W4A16_NVFP4"
        if hqc is not None and isinstance(hqc.get("quantization"), dict):
            hqc["quantization"]["quant_algo"] = "W4A16_NVFP4"
    cfg["quantization_config"] = qcfg
    summary = (f"{qcfg['quant_algo']}, {len(ignore)} ignore entries: "
               f"{ignore[:6]}{' ...' if len(ignore) > 6 else ''}")

    with open(f"{final_dir}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Saved config.json ({summary})", flush=True)
    if hqc is not None:
        with open(f"{final_dir}/hf_quant_config.json", "w") as f:
            json.dump(hqc, f, indent=2)
        print("  Saved hf_quant_config.json", flush=True)

    for fn in AUX_FILES:
        src = f"{source_model_path}/{fn}"
        if os.path.exists(src):
            shutil.copy(src, f"{final_dir}/{fn}")
            print(f"  Copied {fn} from source", flush=True)


# ============================================================
# Driver
# ============================================================
def iter_calib_texts(args, tokenizer):
    from datasets import load_dataset
    if args.dataset.startswith("/") or args.dataset.endswith(".jsonl"):
        print(f"  loading local jsonl: {args.dataset}", flush=True)
        ds = load_dataset("json", data_files=args.dataset, split="train", streaming=True)
    else:
        ds_args = (args.dataset, args.dataset_config) if args.dataset_config else (args.dataset,)
        ds = load_dataset(*ds_args, split="train", streaming=True)
    for sample in ds:
        text = sample.get("article") or sample.get("text")
        if not text and sample.get("messages") and getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
        if text and len(text) >= args.min_chars:
            yield text


def resolve_resume(base):
    """Find the Phase-5.5 checkpoints on every live node; validate that they
    form one complete, contiguous model."""
    nodes = [n for n in ray.nodes() if n["Alive"]]
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    refs = [_scan_checkpoints.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)
            ).remote(base) for n in nodes]
    by_index = {}
    for node, (ip, found) in zip(nodes, ray.get(refs)):
        for i, d, meta in found:
            if i in by_index and by_index[i][0] != node["NodeID"]:
                print(f"  note: shard{i} checkpoint exists on several nodes, using {by_index[i][1]}", flush=True)
                continue
            by_index[i] = (node["NodeID"], ip, d, meta)
    if not by_index:
        print(f"FATAL: no checkpoint dirs {base}_ckpt_shard* on any live node", flush=True)
        sys.exit(1)
    n = by_index[0][3].get("n_shards") if 0 in by_index else None
    if n is None:  # legacy meta.json: infer from the head shard
        heads = [i for i, v in by_index.items() if v[3]["has_head"]]
        n = heads[0] + 1 if heads else None
    if n is None or sorted(by_index) != list(range(n)):
        print(f"FATAL: incomplete checkpoint set — found shards {sorted(by_index)}, "
              f"expected 0..{(n or 0) - 1}", flush=True)
        sys.exit(1)
    shards = [by_index[i] for i in range(n)]
    metas = [s[3] for s in shards]
    pos = 0
    for i, m in enumerate(metas):
        ok = (m["layer_start"] == pos and m["has_embed"] == (i == 0) and m["has_head"] == (i == n - 1))
        if not ok:
            print(f"FATAL: shard{i} meta inconsistent: {m}", flush=True)
            sys.exit(1)
        pos = m["layer_end"]
    total = metas[0].get("num_layers_total", pos)
    if pos != total:
        print(f"FATAL: checkpoints cover {pos} layers, model has {total}", flush=True)
        sys.exit(1)
    return shards


def run_sequential(args):
    import hashlib
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    source_model = args.source_model
    full_cfg, text_cfg = load_text_config(source_model)
    L = text_cfg.num_hidden_layers
    print(f"SEQUENTIAL mode — {(full_cfg.architectures or ['?'])[0]} ({text_cfg.model_type}), "
          f"{L} layers\nSource: {source_model}\nOutput: {args.output_dir}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(source_model)
    ids = []
    for text in iter_calib_texts(args, tokenizer):
        if len(ids) >= args.calib_size:
            break
        ids.append(tokenizer(text, truncation=True, max_length=args.seq_len).input_ids)
    if len(ids) < args.calib_size:
        print(f"FATAL: dataset yielded only {len(ids)}/{args.calib_size} usable samples", flush=True)
        sys.exit(1)
    quant_opts = {"disable": args.disable_quant, "moe_calib_ratio": args.moe_calib_ratio}

    nodes = sorted([n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0],
                   key=lambda n: n["NodeManagerAddress"])
    W = args.seq_workers or len(nodes)
    if W > len(nodes) and args.gpus_per_actor >= 1:
        print(f"FATAL: {W} workers but {len(nodes)} GPU nodes (use --gpus-per-actor < 1)", flush=True)
        sys.exit(1)
    placement = [nodes[r % len(nodes)] for r in range(W)]
    key_src = json.dumps({"model": os.path.realpath(source_model), "ids": ids, "W": W,
                          "quant": quant_opts}, sort_keys=True)
    sample_key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    print(f"Calibration: {len(ids)} samples, {sum(map(len, ids))} tokens, {W} workers on "
          f"{[n['NodeManagerAddress'] for n in placement]}, key={sample_key}", flush=True)

    workers = [SeqWorker.options(
                   num_gpus=args.gpus_per_actor, name=f"seq{r}",
                   scheduling_strategy=NodeAffinitySchedulingStrategy(placement[r]["NodeID"], soft=False),
               ).remote(source_model, r, W, args.temp_base, quant_opts) for r in range(W)]

    probes = ray.get([w.probe.remote(sample_key) for w in workers])
    dones = {p[2] for p in probes}
    resume_done = -1
    if not args.fresh and len(dones) == 1 and probes[0][2] >= 0 and probes[0][3]:
        resume_done = probes[0][2]
        print(f"  resuming after layer {resume_done}", flush=True)
    elif any(p[2] >= 0 for p in probes):
        print(f"  existing state ignored (fresh={args.fresh}, per-worker done={[p[2] for p in probes]}, "
              f"export ok={probes[0][3]})", flush=True)
    for r in ray.get([w.setup.remote(ids[r::W], sample_key, resume_done) for r, w in enumerate(workers)]):
        print(f"  {r}", flush=True)

    stats_path = f"{args.temp_base}_seq_layer_stats.json"
    layer_stats = {}
    if args.layer_stats and resume_done >= 0 and os.path.exists(stats_path):
        with open(stats_path) as f:
            layer_stats = {int(k): v for k, v in json.load(f).items() if int(k) <= resume_done}

    t_start = time.time()
    total_bytes = 0
    for i in range(resume_done + 1, L):
        t0 = time.time()
        res = ray.get([w.calib_layer.remote(i) for w in workers])
        merged = {}
        for amax, _ in res:
            for n, v in amax.items():
                merged[n] = v if n not in merged else torch.maximum(merged[n], v)
        if args.layer_stats:
            # before the export, which packs the weights to FP4 in place
            st = ray.get(workers[0].layer_stats.remote(i, merged, args.act_format,
                                                       args.layer_stats_samples))
            layer_stats[i] = st
            with open(stats_path + ".tmp", "w") as f:
                json.dump(layer_stats, f)
            os.replace(stats_path + ".tmp", stats_path)
        ex = ray.get(workers[0].export_layer.remote(i, merged))
        mem = ray.get([w.commit.remote(i) for w in workers])
        total_bytes += ex["bytes"]
        done_n = i - resume_done
        eta = (time.time() - t_start) / done_n * (L - 1 - i)
        lt = text_cfg.layer_types[i] if getattr(text_cfg, "layer_types", None) else ""
        print(f"  layer {i + 1}/{L} {lt} {time.time() - t0:.0f}s "
              f"(load {max(r[1]['load_s'] for r in res):.0f}s, calib {max(r[1]['calib_s'] for r in res):.0f}s, "
              f"export {ex['export_s']:.0f}s) {ex['bytes'] / 1e9:.2f} GB, mem={max(mem):.1f}G, "
              f"eta {eta / 60:.0f} min", flush=True)
        if args.layer_stats:
            print(f"    err {format_layer_stats(layer_stats[i])}", flush=True)
        if i == resume_done + 1 or (i + 1) % 10 == 0 or i == L - 1:
            print(f"    amax: {ex['stats']}", flush=True)
    print(f"  all layers done in {(time.time() - t_start) / 60:.1f} min", flush=True)

    print(ray.get(workers[0].write_special.remote()), flush=True)
    ray.get([w.release.remote() for w in workers])
    export_dir = f"{args.temp_base}_seq_export"
    gather_shard_exports([export_dir], [placement[0]["NodeManagerAddress"]], args.ssh_user)
    print(f"\n=== Merge to {args.output_dir} ===", flush=True)
    t0 = time.time()
    merge_exports([export_dir], [L], args.output_dir, source_model, L, act_format=args.act_format)
    print(f"  Merge done in {(time.time() - t0):.1f}s", flush=True)
    if args.layer_stats:
        with open(f"{args.output_dir}/layer_stats.json", "w") as f:
            json.dump({"act_format": args.act_format, "samples": args.layer_stats_samples,
                       "layers": {str(k): v for k, v in sorted(layer_stats.items())}}, f, indent=1)
        print(report_layer_outliers(layer_stats), flush=True)
    print(f"\nNVFP4 model written to: {args.output_dir}")
    print(f"Temp dirs can be deleted: rm -rf {args.temp_base}_seq_*  (on every worker node)")


def main():
    args = parse_args()
    ray.init(address="auto")
    print(f"Cluster resources: {ray.cluster_resources()}", flush=True)
    if args.sequential:
        if not args.source_model:
            print("FATAL: --source-model is required.", flush=True)
            sys.exit(1)
        run_sequential(args)
        return
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    def make_actor(i, node_id, *ctor_args, **ctor_kw):
        return ModelShard.options(
            num_gpus=args.gpus_per_actor,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
            name=f"shard{i}",
        ).remote(*ctor_args, **ctor_kw)

    # ---------------- Resume: Phase 6 + 7 only ----------------
    if args.resume_from_checkpoint:
        found = resolve_resume(args.resume_from_checkpoint)
        N = len(found)
        metas = [f[3] for f in found]
        source_model = args.source_model or metas[0]["model_path"]
        layer_counts = [m["layer_end"] - m["layer_start"] for m in metas]
        num_layers = sum(layer_counts)
        shard_ips = [f[1] for f in found]
        ckpt_dirs = [f[2] for f in found]
        print(f"RESUME mode — {N} shards, layer_counts={layer_counts}, nodes={shard_ips}", flush=True)
        print(f"Source: {source_model}\nOutput: {args.output_dir}", flush=True)
        shards = [make_actor(i, found[i][0], source_model, 0, 0, False, False, f"shard{i}")
                  for i in range(N)]
        print("\n=== Resume setup ===", flush=True)
        for r in ray.get([shards[i].setup_for_resume.remote(ckpt_dirs[i]) for i in range(N)]):
            print(f"  {r}", flush=True)
    else:
        if not args.source_model:
            print("FATAL: --source-model is required for normal (non-resume) runs.", flush=True)
            sys.exit(1)
        source_model = args.source_model
        print(f"Source: {source_model}\nOutput: {args.output_dir}", flush=True)
        full_cfg, text_cfg = load_text_config(source_model)
        num_layers = text_cfg.num_hidden_layers
        if args.shard_layers:
            layer_counts = [int(x.strip()) for x in args.shard_layers.split(",")]
            if sum(layer_counts) != num_layers:
                print(f"FATAL: --shard-layers sum {sum(layer_counts)} != num_layers {num_layers}", flush=True)
                sys.exit(1)
        else:
            split = args.split if args.split is not None else (num_layers // 2)
            layer_counts = [split, num_layers - split]
        N = len(layer_counts)
        # Open tokenizer + dataset before loading anything heavy, so a bad
        # dataset id fails in seconds instead of after Phase 1/2.
        tokenizer = AutoTokenizer.from_pretrained(source_model)
        calib_iter = iter_calib_texts(args, tokenizer)
        try:
            first_text = next(calib_iter)
        except StopIteration:
            print(f"FATAL: dataset {args.dataset} yielded no usable samples", flush=True)
            sys.exit(1)
        arch = (full_cfg.architectures or ["?"])[0]
        print(f"Model: {arch} ({text_cfg.model_type}), hidden_size={text_cfg.hidden_size}, "
              f"layers={num_layers}, shard_layers={layer_counts}", flush=True)

        # Biggest-memory nodes first (Sparks get the big shards, a small eGPU
        # box the trailing shard with lm_head). With --gpus-per-actor < 1
        # several shards may share a node.
        nodes = sorted([n for n in ray.nodes() if n["Alive"]],
                       key=lambda n: -n["Resources"].get("memory", 0))
        if len(nodes) >= N:
            placement = nodes[:N]
        elif args.gpus_per_actor < 1:
            placement = [nodes[i % len(nodes)] for i in range(N)]
        else:
            print(f"FATAL: only {len(nodes)} live nodes for {N} shards "
                  f"(use --gpus-per-actor < 1 to stack shards)", flush=True)
            sys.exit(1)
        shard_ips = [n["NodeManagerAddress"] for n in placement]
        print(f"Placement: {[(ip, int(n['Resources'].get('memory', 0)/1e9)) for ip, n in zip(shard_ips, placement)]}",
              flush=True)

        quant_opts = {"disable": args.disable_quant, "moe_calib_ratio": args.moe_calib_ratio}
        shards = []
        offset = 0
        for i, n_layers in enumerate(layer_counts):
            shards.append(make_actor(i, placement[i]["NodeID"], source_model, offset, offset + n_layers,
                                     i == 0, i == N - 1, f"shard{i}",
                                     shard_index=i, n_shards=N, quant_opts=quant_opts))
            offset += n_layers

        print(f"\n=== Phase 1: Loading model into {N} shards ===", flush=True)
        t0 = time.time()
        for r in ray.get([s.load_model.remote() for s in shards]):
            print(f"  {r}", flush=True)
        print(f"  done in {time.time()-t0:.1f}s", flush=True)

        print("\n=== Phase 2: Inserting NVFP4 quantizers ===", flush=True)
        for r in ray.get([s.apply_quantization.remote() for s in shards]):
            print(f"  {r}", flush=True)

        def forward_chain(input_ids):
            hidden, pos = ray.get(shards[0].forward_shard.remote(input_ids))
            for s in shards[1:]:
                hidden, pos = ray.get(s.forward_shard.remote(hidden, pos))

        import itertools
        print(f"\n=== Phase 3: Calibration with {args.calib_size} samples ===", flush=True)
        t0 = time.time()
        count = 0
        for text in itertools.chain([first_text], calib_iter):
            if count >= args.calib_size:
                break
            inputs = tokenizer(text, truncation=True, max_length=args.seq_len, return_tensors="pt")
            forward_chain(inputs.input_ids)
            count += 1
            if count == 1 or count % 10 == 0:
                elapsed = time.time() - t0
                eta = (args.calib_size - count) * elapsed / count
                print(f"  {count}/{args.calib_size} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
        if count < args.calib_size:
            print(f"FATAL: dataset yielded only {count}/{args.calib_size} usable samples", flush=True)
            sys.exit(1)
        print(f"  Calibration done in {(time.time()-t0)/60:.1f} min", flush=True)

        print("\n=== Phase 4: Finalizing calibration ===", flush=True)
        for r in ray.get([s.finalize.remote() for s in shards]):
            print(f"  {r}", flush=True)

        print("\n=== Phase 5: Diagnostic - amax stats ===", flush=True)
        for r in ray.get([s.export_status.remote() for s in shards]):
            print(f"  {r}", flush=True)

        print("\n=== Phase 5.5: Evicting layers to disk for streaming export ===", flush=True)
        ckpt_dirs = [f"{args.temp_base}_ckpt_shard{i}" for i in range(N)]
        t0 = time.time()
        for r in ray.get([shards[i].save_layers_to_disk.remote(ckpt_dirs[i]) for i in range(N)]):
            print(f"  {r}", flush=True)
        print(f"  Eviction done in {(time.time()-t0):.1f}s", flush=True)

    # ---------------- Phase 6 / 6.5 / 7 (shared) ----------------
    print("\n=== Phase 6: Per-shard export (disk-streaming) ===", flush=True)
    temp_dirs = [f"{args.temp_base}_shard{i}" for i in range(N)]
    t0 = time.time()
    for r in ray.get([shards[i].export_shard.remote(temp_dirs[i], ckpt_dirs[i]) for i in range(N)]):
        print(f"  {r}", flush=True)
    print(f"  Per-shard export done in {(time.time()-t0):.1f}s", flush=True)

    gather_shard_exports(temp_dirs, shard_ips, args.ssh_user)

    print(f"\n=== Phase 7: Merge to {args.output_dir} ===", flush=True)
    t0 = time.time()
    merge_exports(temp_dirs, layer_counts, args.output_dir, source_model, num_layers,
                  act_format=args.act_format)
    print(f"  Merge done in {(time.time()-t0):.1f}s", flush=True)

    print(f"\nNVFP4 model written to: {args.output_dir}")
    print(f"Temp dirs can be deleted: rm -rf {args.temp_base}_shard* {args.temp_base}_ckpt_shard*")


if __name__ == "__main__":
    main()
