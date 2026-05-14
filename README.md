# distrib-nvfp4

Distributed NVFP4 quantization pipeline for 100B+ class LLMs on a 2-node NVIDIA DGX Spark (GB10) cluster.

This pipeline is what produced [`Kaleto/Anubis-Pro-105B-NVFP4`](https://huggingface.co/Kaleto/Anubis-Pro-105B-NVFP4) — to my knowledge the first publicly available NVFP4 quantization of a 100B+ class RP/storytelling model, made on a personal 2× DGX Spark workstation rather than datacenter GPUs.

The script is Ray-based, splits a Llama/Mistral-class model across two Spark nodes, runs calibration with hidden states routed between actors via RPC, and exports per-layer NVFP4 + final HF compressed-tensors merge.

## Why this exists

Quantizing 100B+ class models for the new NVIDIA DGX Spark workstation is not as turn-key as it sounds. The standard single-node `modelopt hf_ptq.py` workflow silently fails on GB10's 128 GB unified memory (the `accelerate` library misdetects unified memory as a 5.2 TB GPU and triggers an OOM-kill during shard loading). Patching it to work via `--low_memory_mode` is a known dead-end for ≥70B class models.

This pipeline shards the model across two Sparks via Ray, calibrates with hidden states passed between shards over the ConnectX-7 link, exports per-layer to bound peak memory, and merges into a single HF-compatible compressed-tensors NVFP4 directory.

## Hardware target

- **2× NVIDIA DGX Spark** (GB10 Grace+Blackwell, 128 GB unified memory each)
- **ConnectX-7 IB** between nodes (44 GB/s NCCL measured)
- **NFS cross-mount**: BF16 source on one node's NVMe, NVFP4 output on the other's

The driver itself is architecture-agnostic — it works for any HuggingFace model whose decoder layers expose a `RMSNorm`-style `input_layernorm` and a `RotaryEmbedding`-style positional encoding. Tested with Llama-3-derived (Anubis-Pro-105B) and Mistral-Large-derived (Behemoth-X-123B) bases. Qwen/Gemma/Cohere/Nemotron-NAS architectures may need small adapters.

## Quick start

Requirements on both nodes:
- `modelopt` 0.43.0+ (NVIDIA TensorRT-Model-Optimizer)
- `ray` 2.55.1
- `transformers` 4.57.6
- `safetensors`, `accelerate`, `torch` 2.11+
- All from the same Python environment, mirrored on both nodes (rsync is fine)

Ray cluster bring-up:
```bash
# On both nodes, in the env that has modelopt:
export GLOO_SOCKET_IFNAME=enp1s0f1np1     # or your IB interface
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export RAY_memory_monitor_refresh_ms=0
# Generous heartbeat tolerance for large model loads:
export RAY_health_check_period_ms=60000
export RAY_health_check_failure_threshold=10

# Head (DX10-01):
ray start --head --node-ip-address=<head-ip> --port=6379 --num-gpus=1

# Worker (DX10-02):
ssh worker "<same env exports>; ray start --address=<head-ip>:6379 --num-gpus=1"

ray status   # expect 2 nodes, 2 GPUs
```

Smoke-test the export path on a single layer (~1 min, validates new architectures before a full 30+ min run):
```bash
python3 scripts/export_smoke_test.py --source-model /path/to/<MODEL>-BF16
```

Full pipeline in tmux (Ray actors die if the driver exits, so keep the session alive):
```bash
tmux new -d -s quant 'scripts/run_quant.sh \
    --source-model /path/to/<MODEL>-BF16 \
    --output-dir   /path/to/<MODEL>-NVFP4 \
    --calib-size   256 \
    --temp-base    /tmp/_distrib_quant'
tail -F /tmp/distrib_quant.log
```

If Phase 6 (export) crashes after Phase 5.5 (eviction) has written the checkpoint, you can resume just Phase 6+7 without re-doing the 25 min of calibration:
```bash
scripts/run_quant.sh \
    --resume-from-checkpoint /tmp/_distrib_quant \
    --output-dir /path/to/<MODEL>-NVFP4
```

## Pipeline architecture

Driver runs on the head node, orchestrates two Ray actors:

- **shard0**: `embed_tokens` + layers `[0:split]`
- **shard1**: layers `[split:N]` + `norm` + `lm_head`

Phases:

| # | What |
|---|---|
| 1 | Build empty HF model on each actor (`init_empty_weights`), materialize only owned layers via streaming `safetensors.safe_open` (per-tensor, never caches a whole shard — required for ≥120 GB models on 128 GB UMA). |
| 2 | Wrap own layers in a `LocalQuantizable(nn.Module)`, call `mtq.quantize(wrapper, NVFP4_DEFAULT_CFG, forward_loop=None)`. `forward_loop=None` is critical — it inserts quantizers in calibration mode without modelopt running its own forward. |
| 3 | Driver-orchestrated calibration: shard0.forward_first → returns hidden states over Ray → shard1.forward_second. 128-256 samples × variable length from `cnn_dailymail` by default. |
| 4 | `load_calib_amax` + `disable_calib` + `enable_quant` on every TensorQuantizer per actor. |
| 5 | Diagnostic — print amax stats. Target: `nan=0 zero=0`. |
| 5.5 | **Evict layers to disk** — each actor `torch.save`s its quantized layers to a checkpoint dir, replaces in-memory refs with `nn.Identity()`. Drops UMA usage from ~95 % back to ~5 %. Files double as a resume checkpoint. |
| 6 | Per-layer streaming export — load layer from checkpoint dir, run `mte.export_hf_checkpoint`, save the per-layer NVFP4 safetensors, discard. Peak memory: ~1 layer + ~10 GB export overhead, regardless of model size. |
| 7 | Driver-side merge — collect per-actor files, rename shard1 layer indices, patch config.json (restore vocab_size, token IDs, set `input_activations.dynamic=true`), inject `input_scale=1.0` for every quantized Linear (modelopt 0.43 omits these; vLLM needs them present), copy tokenizer, write sharded `model-NNNNN-of-NNNNN.safetensors`. |

For more on Phase 5.5 / 6 and the export-side fixes that make vLLM actually serve the output, see `docs/debugging-notes.md`.

## Known production targets

| Model | Status |
|---|---|
| TheDrummer/Anubis-Pro-105B-v1 (Llama-3.3-based, 105B, 120 layers) | ✅ produced `Kaleto/Anubis-Pro-105B-NVFP4` |
| TheDrummer/Behemoth-X-123B-v2.2 (Mistral-Large-based, 88 layers) | 🟡 in validation as of this release |

Memory ceilings (per shard):
- ≤105B on 128 GB UMA — comfortable
- 105–123B class — memory-tight but works with streaming-load (this pipeline)
- >130B class — would need further chunking or a third node

## Acknowledgments

- **[Avarok-Cybersecurity](https://github.com/Avarok-Cybersecurity/dgx-vllm)** (`tbraun96`) for the MARLIN-backend port of NVFP4 GEMM that made NVFP4 actually competitive on Spark. The model this pipeline produces is intended to be served with that runtime stack.
- **[saricles](https://huggingface.co/saricles)** for setting the state-of-the-art bar on GB10-tuned NVFP4 recipes — the agentic-mix calibration and ignore-list documentation in their `-GB10`-suffixed releases is the reference for what a Spark-bandwidth-aware quantization looks like.
- **NVIDIA** for `modelopt` / TensorRT-Model-Optimizer (the upstream NVFP4 implementation) and the DGX Spark platform.
- **vLLM project** for compressed-tensors NVFP4 inference support.

## License

Apache 2.0 — see `LICENSE`.

This applies to the pipeline code in this repo only. Models produced by this pipeline inherit the license of their base model (e.g. Llama-3.3-derived models stay under the Llama 3.3 Community License).

## Status

This is "release v0.1 — works on my machine(s)". Single-author project, not yet load-tested by other Spark owners. Issues + PRs welcome.
