#!/usr/bin/env python3
"""
End-to-end smoke test of the shard code path on a few real layers, in-process
(no Ray), before committing to a full calibration run.

Runs the exact ModelShardImpl methods the driver calls:
  load_model -> apply_quantization -> forward_shard (calibration) -> finalize
  -> export_status -> save_layers_to_disk (cloudpickle) -> export_shard
on layers [a:b] with real weights from the source checkpoint, then reports the
exported tensor layout, input_scale coverage and the ignore list the merge
would emit for these layers.

Pick a range that covers every layer type, e.g. 0:4 for Qwen3.5-MoE
(3x linear_attention + 1x full_attention) or 0:2 for Llama-class models.

Run:
  python3 export_smoke_test.py --source-model /path/to/BF16 --layers 0:4
"""
import argparse
import collections
import os
import re
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distrib_quant import (  # noqa: E402
    ModelShardImpl, build_ignore_list, detect_text_prefix, module_to_out_key, read_weight_map,
)


def main():
    p = argparse.ArgumentParser(
        description="Smoke-test load/quantize/calibrate/evict/export on a few real layers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source-model", "-s", required=True, help="BF16 source model directory.")
    p.add_argument("--layers", default="0:2", help="Layer range a:b to test.")
    p.add_argument("--samples", type=int, default=4, help="Calibration samples.")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--moe-calib-ratio", type=float, default=1.0)
    p.add_argument("--disable-quant", action="append", default=[])
    p.add_argument("--out-dir", default="/tmp/_smoke_test_export")
    args = p.parse_args()

    a, b = (int(x) for x in args.layers.split(":"))
    shard = ModelShardImpl(args.source_model, a, b, has_embed=True, has_head=False, name="smoke",
                           quant_opts={"disable": args.disable_quant,
                                       "moe_calib_ratio": args.moe_calib_ratio})

    def step(label, fn, *fa):
        t0 = time.time()
        print(f"[{label}] ...", flush=True)
        r = fn(*fa)
        print(f"[{label}] {r if r is not None else 'ok'}  ({time.time()-t0:.1f}s)", flush=True)
        return r

    step("load", shard.load_model)
    print(f"  layer types: {[shard.layer_types[i] for i in range(a, b)] if shard.layer_types else 'n/a'}")
    step("quantize", shard.apply_quantization)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.source_model)
    texts = [
        "The committee met on Tuesday to discuss the budget for the coming fiscal year. " * 20,
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n" * 10,
        "Sie öffnete die Tür langsam und blickte in den dunklen Flur, in dem es nach Regen roch. " * 15,
        "量子力学是描述微观粒子运动规律的物理学分支。" * 30,
    ]

    def calib():
        for i in range(args.samples):
            ids = tok(texts[i % len(texts)], truncation=True, max_length=args.seq_len,
                      return_tensors="pt").input_ids
            hidden, pos = shard.forward_shard(ids)
            assert hidden is not None and torch.isfinite(hidden.float()).all(), "non-finite hidden states"
        return f"{args.samples} samples, last hidden {list(hidden.shape)}"
    step("calibrate", calib)
    step("finalize", shard.finalize)
    step("status", shard.export_status)

    ckpt = args.out_dir + "_ckpt"
    step("evict", shard.save_layers_to_disk, ckpt)
    step("export", shard.export_shard, args.out_dir, ckpt)

    # ---- inspect what the merge would get ----
    from safetensors import safe_open
    tensors = {}
    for fn in sorted(os.listdir(args.out_dir)):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(args.out_dir, fn), framework="pt") as sf:
                for k in sf.keys():
                    tensors[k] = sf.get_tensor(k)
    tp = detect_text_prefix(read_weight_map(args.source_model))
    out = {module_to_out_key(k, tp): v for k, v in tensors.items()}

    pat = collections.Counter()
    for k, v in out.items():
        g = re.sub(r"\.experts\.\d+\.", ".experts.E.", re.sub(r"\.layers\.\d+\.", ".layers.N.", k))
        pat[(g, str(v.dtype))] += 1
    print("\nExported tensor patterns (count, dtype):")
    for (g, dt), n in sorted(pat.items()):
        print(f"  {n:6d}  {dt:16s} {g}")

    quant = [k[:-len(".weight_scale_2")] for k in out if k.endswith(".weight_scale_2")]
    no_in = [m for m in quant if m + ".input_scale" not in out]
    print(f"\nQuantized modules: {len(quant)}, without input_scale: {len(no_in)} {no_in[:3]}")
    if quant:
        vals = torch.stack([out[m + ".input_scale"].float().flatten()[0] for m in quant if m not in no_in])
        print(f"input_scale range: min={vals.min().item():.3e} max={vals.max().item():.3e} "
              f"ones={(vals == 1.0).sum().item()}")
    ignore = build_ignore_list(out)
    print(f"Ignore list for these layers ({len(ignore)}): {ignore[:12]}")

    with open(os.path.join(args.out_dir, "config.json")) as f:
        import json
        q = json.load(f).get("quantization_config", {})
    print(f"Template quantization_config keys: {sorted(q)}; quant_method={q.get('quant_method')} "
          f"quant_algo={q.get('quant_algo')}")
    groups = q.get("config_groups", {})
    for g, spec in groups.items():
        print(f"  {g}: input_activations={spec.get('input_activations')} targets={spec.get('targets')}")

    ok = bool(quant) and not no_in
    print("\nSUCCESS" if ok else "\nFAILED: no quantized modules or missing input_scale")
    shutil.rmtree(ckpt, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
