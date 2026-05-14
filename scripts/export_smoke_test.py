"""
Quick smoke test before re-running 30min calibration.

Model-agnostic. Builds a 1-layer template from the source model's config,
materializes one real decoder layer + tiny dummies, runs the same Phase-6
codepath (mtq.quantize + mte.export_hf_checkpoint) the driver uses.

Run:
  python3 export_smoke_test.py --source-model /path/to/BF16

If it prints SUCCESS + four files in /tmp/_smoke_test_export, the export
path is healthy for that model — safe to launch the full pipeline.
"""
import argparse
import os
import shutil

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

import modelopt.torch.export as mte
import modelopt.torch.quantization as mtq


def main():
    p = argparse.ArgumentParser(
        description="Smoke-test the NVFP4 export path on a 1-layer template of any HF model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source-model", "-s", required=True,
                   help="Path to BF16 source model directory.")
    p.add_argument("--out-dir", default="/tmp/_smoke_test_export",
                   help="Where to write the smoke-test export output.")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    cfg_t = AutoConfig.from_pretrained(args.source_model)
    cfg_t.num_hidden_layers = 1
    cfg_t.vocab_size = 2
    cfg_t.pad_token_id = None
    cfg_t.bos_token_id = None
    cfg_t.eos_token_id = None
    print(f"cfg: arch={cfg_t.architectures[0] if cfg_t.architectures else '?'}, "
          f"vocab={cfg_t.vocab_size}, pad={cfg_t.pad_token_id}, num_layers={cfg_t.num_hidden_layers}")

    print("Building model from cfg_t (meta)...")
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg_t, torch_dtype=torch.bfloat16)
    print("  OK")

    # Discover architecture-specific helper classes by introspecting the meta model.
    decoder_layer_cls = type(m.model.layers[0])
    rms_norm_cls = type(m.model.layers[0].input_layernorm)
    rotary_emb_cls = type(getattr(m.model, "rotary_emb", None))
    if rotary_emb_cls is type(None):
        rotary_emb_cls = type(getattr(m.model.layers[0].self_attn, "rotary_emb", None))
    print(f"  Detected: layer={decoder_layer_cls.__name__}, "
          f"rms_norm={rms_norm_cls.__name__}, rotary={rotary_emb_cls.__name__}")

    print("Materializing 1 random layer + dummies on cuda...")
    real_layer = decoder_layer_cls(cfg_t, layer_idx=0).to(args.device, dtype=torch.bfloat16)
    m.model.layers[0] = real_layer

    de = nn.Embedding(cfg_t.vocab_size, cfg_t.hidden_size).to(args.device, dtype=torch.bfloat16)
    with torch.no_grad():
        de.weight.zero_()
    m.model.embed_tokens = de

    dn = rms_norm_cls(cfg_t.hidden_size, eps=cfg_t.rms_norm_eps).to(args.device, dtype=torch.bfloat16)
    with torch.no_grad():
        dn.weight.zero_()
    m.model.norm = dn

    dl = nn.Linear(cfg_t.hidden_size, cfg_t.vocab_size, bias=False).to(args.device, dtype=torch.bfloat16)
    with torch.no_grad():
        dl.weight.zero_()
    m.lm_head = dl

    m.model.rotary_emb = rotary_emb_cls(cfg_t).to(args.device)
    print("  OK")

    print("Applying NVFP4 quantization to layer 0...")
    mtq.quantize(m.model.layers[0], mtq.NVFP4_DEFAULT_CFG, forward_loop=None)
    # Mirror the real driver's pre-export setup so this smoke test exercises the same code path:
    #   - finalize amax (so export sees realistic state)
    #   - disable_calib / enable_quant
    # DO NOT set _calibrator = None: modelopt's export uses set_quantizer_by_cfg_context which
    # AttributeErrors on a None calibrator. The driver previously did this and the smoke test
    # missed it because the test wasn't doing the same null-out. Don't reintroduce that gap.
    from modelopt.torch.quantization.nn import TensorQuantizer
    for mod in m.model.layers[0].modules():
        if isinstance(mod, TensorQuantizer):
            if hasattr(mod, "_amax") and mod._amax is None:
                mod._amax = torch.tensor(1.0, device=args.device)
            if hasattr(mod, "disable_calib"):
                mod.disable_calib()
            if hasattr(mod, "enable_quant"):
                mod.enable_quant()
    print("  OK")

    shutil.rmtree(args.out_dir, ignore_errors=True)
    print(f"Calling export_hf_checkpoint -> {args.out_dir}...")
    try:
        mte.export_hf_checkpoint(m, dtype=torch.bfloat16, export_dir=args.out_dir)
        print("  SUCCESS")
        print("Files produced:")
        for f in sorted(os.listdir(args.out_dir)):
            sz = os.path.getsize(f"{args.out_dir}/{f}")
            print(f"  {f}  ({sz} bytes)")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
