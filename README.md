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
- **Optional 3rd node** (e.g. RTX 3090 in a Proxmox VM over plain 2.5 GbE LAN) for N-shard mode used on 111B–123B class models — see `--shard-layers a,b,c` and the Phase 6.5 gather

The driver itself is architecture-agnostic — it works for any HuggingFace model whose decoder layers expose an `input_layernorm` (RMSNorm- or LayerNorm-style) and a rotary positional encoding. Tested with:

- **Llama-3-derived**: Anubis-Pro-105B, DeepSeek-R1-Distill-Llama-70B, Llama-3.3-70B-Instruct
- **Mistral-Large-derived**: Behemoth-X-123B-v2.2
- **Cohere2 / Command-A**: Fallen-Command-A-111B — in-tree handling for tied embeddings (no separate `lm_head.weight` in the checkpoint), `layer_norm_eps` attribute name, and dropping `generation_config.cache_implementation=hybrid` for the per-layer export template

- **Qwen3.5 / Qwen3.6 MoE** (`qwen3_5_moe`, hybrid Gated-DeltaNet + full attention, 256 experts, VL checkpoint): the text stack is loaded from the nested `text_config` and the `model.language_model.*` keys are mapped onto it; experts + shared expert go to NVFP4, all token-mixer projections, routers and `shared_expert_gate` stay BF16; vision tower and MTP head are copied verbatim in BF16. Needs transformers ≥ 5 (tested with 5.8.0).

Gemma / Nemotron-NAS untested.

## Quick start

Requirements on both nodes:
- `modelopt` 0.43.0+ (NVIDIA TensorRT-Model-Optimizer)
- `ray` 2.55.1
- `transformers` 4.57.6 for Llama / Mistral / Cohere2; ≥ 5.x (tested 5.8.0) for Qwen3.5/3.6 MoE
- `datasets` (calibration data, streamed)
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

Useful flags:
- `--shard-layers 40,40,8` — N-shard split (`'40'` = single shard)
- `--gpus-per-actor 0.5` — stack several shards on one node (e.g. a 35B test run with two shards on a single Spark)
- `--moe-calib-ratio 1.0` — MoE: fraction of experts every calibration token is routed to while collecting stats
- `--disable-quant '*self_attn*'` — extra modules to keep in BF16 (repeatable)
- `--dataset abisee/cnn_dailymail` / `--dataset /path/to/calib.jsonl` — rows with `article`, `text` or chat `messages`
- `--ssh-user` — user for the Phase-6.5 rsync from remote shard nodes

## Output format: W4A16 by default (`--act-format`)

The exported NVFP4 weights can be declared to vLLM in two ways:

| `--act-format` | config | vLLM path on GB10 (v0.31) | Qwen3.6-35B-A3B: top-K KLD vs BF16 | prefill / decode |
|---|---|---|---|---|
| `w4a16` (default) | `quant_algo: W4A16_NVFP4` (same ignore list) | Marlin FP4 GEMM, BF16 activations | **0.076** (median 0.0065) | 5969 tok/s / 38.7 tok/s |
| `w4a4` | plain `NVFP4` + ignore list | FlashInfer-Cutlass FP4 GEMM, activations quantized to FP4 at runtime | 0.110 (median 0.0122) | 6562 tok/s / 37.9 tok/s |

Dense models gain as much: Qwen3.5-9B-heretic-v2 measured 0.179 (W4A4) → **0.129** (W4A16), median 0.028 → 0.015. On a dense 111B (Fallen-Command) decode stays the same (3.3 tok/s), but prefill drops 654 → 228 tok/s, because every GEMM — not just the experts — goes through Marlin.

Same safetensors in both cases — only `quant_algo` in `config.json` / `hf_quant_config.json` differs (a per-module `MIXED_PRECISION` declaration measured the same: 0.076). NVIDIA's own release `nvidia/Qwen3.6-35B-A3B-NVFP4` (bit-identical expert weights, plus FP8 attention) uses W4A16 for the MLP and measured 0.100. In the W4A4 path the activation's global scale barely matters: calibrated `input_scale`, `input_scale = 1.0` and NVIDIA's scales all measured within noise (0.109–0.110) — the per-16-element FP4 rounding of activations dominates.

Measurement: 72 held-out sequences (news, wiki EN/DE, code, chat, RP chats, app prompts; 276k positions), `kld_eval.py` top-50 KLD with union + rest bucket, everything served by the vLLM 0.31 container.

## Sequential mode (`--sequential`)

Instead of holding whole layer slices in memory, the sequential mode walks the model one decoder layer at a time: load layer *i* in BF16 → insert quantizers → push every calibration sample through it (stats collection on) → export it → its outputs become the inputs of layer *i+1*. Peak memory per node is roughly one layer plus the calibration hidden states (128 × 1024 tokens × 3072 hidden ≈ 0.8 GB), so model size is no longer bounded by cluster memory — a 122B-class model runs on a single Spark.

With several workers (`--seq-workers`, default one per live GPU node) the **calibration samples** are split, not the layers: every worker runs the same layer on its share of the samples and keeps its hidden states node-local; the driver max-merges the per-quantizer amax (exact for max calibration) and rank 0 exports. Two Sparks therefore roughly halve the calibration time, and the result does not depend on the worker count. Every worker node needs the BF16 source under the same path (NFS mount or copy). Use one worker per GPU: two worker processes time-sharing one GPU were ~4× slower in a test (hundreds of tiny MoE expert kernels per layer, context switches between processes); one worker on one Spark calibrates as fast as the distributed mode (Qwen3.6-35B-A3B, 128 × ~720 tokens: ~53 s per layer).

The run is resumable: after each layer every worker persists its hidden states (`{temp_base}_seq_w{rank}/state.pt`) and rank 0 has the layer's export in `{temp_base}_seq_export/`. Restarting the same command (same model, samples, worker count, quant options) continues after the last completed layer; `--fresh` forces a restart.

```bash
python3 distrib_quant.py --sequential --source-model /models/Qwen3.5-122B-A10B \
    --output-dir /models/Qwen3.5-122B-A10B-NVFP4 --calib-size 256 --seq-len 2048
```

### Per-layer and per-expert error (`--layer-stats`)

A mean KLD over the whole model can hide single bad layers or experts. With `--layer-stats` the sequential mode measures, for every layer before it is exported, the quantization error on that layer's BF16 inputs (so errors of earlier layers do not accumulate): the whole layer (residual removed), attention and MLP in isolation, the MoE shared expert, and every routed expert together with the number of tokens routed to it. The error is the relative L2 distance to the same layer unquantized, using exactly the quantizers active at serve time (`--act-format w4a16`: weights only). One log line per layer; at the end a summary lists layers above 1.5× and experts above 2× the median, and everything goes to `<output-dir>/layer_stats.json`. Cost: one extra BF16 and one quantized forward of `--layer-stats-samples` (default 32) samples per layer, about 15 s per layer on Qwen3.6-35B-A3B. The exported model is bit-identical with and without it.

Qwen3.6-35B-A3B, W4A16, 64 mixed samples (news, wiki EN/DE, code, chat), all 10,240 experts routed: whole layer 0.030–0.110 (median 0.085), no layer above 1.5× median; experts median 0.141, worst 0.178 (1.27×), none above 2×, and the worst experts are not the rarely routed ones. With weight-only FP4 and a scale per 16 weights the error is nearly uniform; activation quantization (W4A4) is where calibration-dependent outliers would be expected.

## Pipeline architecture

Driver runs on the head node, orchestrates N Ray actors (2 by default, or N≥3 via `--shard-layers a,b,c`):

- **shard0**: `embed_tokens` + layers `[0:split[0]]`
- **shard[1..N-2]** (N-shard mode only): middle layer slices, no embed/head
- **shard[N-1]**: trailing layers + `norm` + `lm_head` — automatically placed on the smallest-VRAM node (e.g. an RTX 3090 takes the tail when paired with two Sparks)

Phases:

| # | What |
|---|---|
| 1 | Build empty HF model on each actor (`init_empty_weights`), materialize only owned layers via streaming `safetensors.safe_open` (per-tensor, never caches a whole shard — required for ≥120 GB models on 128 GB UMA). |
| 2 | Wrap own layers in a `LocalQuantizable(nn.Module)`, call `mtq.quantize(wrapper, cfg, forward_loop=None)` (NVFP4 + per-architecture BF16 patterns), then **re-open stats collection** with `enable_stats_collection`. `forward_loop=None` alone runs modelopt's weight-only calibration and closes collection again, so without the re-open no activation statistics are ever gathered. MoE blocks get `_moe_calib_experts_ratio` (`--moe-calib-ratio`, default 1.0 = every expert sees every calibration token). |
| 3 | Driver-orchestrated calibration: shard0 embeds + runs its layers → hidden states over Ray → next shard … Linear-attention layers get no causal mask. Aborts if the dataset yields fewer than `--calib-size` samples. |
| 4 | `finish_stats_collection` (+ MoE amax sync) — fails if any active quantizer ended up without amax. |
| 5 | Diagnostic — weight / input amax stats. Target: `zero=0 nan=0 none=0`. |
| 5.5 | **Evict layers to disk** — each actor `torch.save`s its quantized layers (cloudpickle) to a fresh checkpoint dir and replaces every in-memory reference (layers, embed, norm, lm_head) with `nn.Identity()`. Drops UMA usage from ~95 % back to ~5 %. Files double as a resume checkpoint; `meta.json` records shard index / count so a resume can verify completeness. |
| 6 | Per-layer streaming export — load layer from checkpoint dir, run `mte.export_hf_checkpoint` inside a 1-layer template, save the layer's tensors under their global index. Peak memory: ~1 layer + export overhead, regardless of model size. |
| 6.5 | Gather — `rsync -a --delete` of every remote shard's export dir to the driver. |
| 7 | Driver-side merge — verify every layer / file is present exactly once, rename to the source layout (e.g. `model.language_model.*`), copy non-text source tensors (vision, MTP) verbatim, write `config.json` = source config + `quantization_config` with an ignore list derived from the final tensors, copy generation config / tokenizer / chat template / processor files from the source. Loads the whole quantized model into driver RAM (≈ output size). |

For more on Phase 5.5 / 6 and the export-side fixes that make vLLM actually serve the output, see `docs/debugging-notes.md`.

## Known production targets

| Model | Mode | Result |
|---|---|---|
| TheDrummer/Anubis-Pro-105B-v1 (Llama-3.3, 105B, 120 layers) | 2-shard | ✅ [`Kaleto/Anubis-Pro-105B-NVFP4`](https://huggingface.co/Kaleto/Anubis-Pro-105B-NVFP4) |
| TheDrummer/Behemoth-X-123B-v2.2 (Mistral-Large, 88 layers) | 3-shard (41/41/6) | ✅ [`Kaleto/Behemoth-X-123B-v2.2-NVFP4`](https://huggingface.co/Kaleto/Behemoth-X-123B-v2.2-NVFP4) |
| deepseek-ai/DeepSeek-R1-Distill-Llama-70B (70B, 80 layers) | 2-shard | ✅ [`Kaleto/DeepSeek-R1-Distill-Llama-70B-NVFP4`](https://huggingface.co/Kaleto/DeepSeek-R1-Distill-Llama-70B-NVFP4) |
| meta-llama/Llama-3.3-70B-Instruct (70B, 80 layers) | 2-shard | ✅ [`Kaleto/Llama-3.3-70B-Instruct-NVFP4`](https://huggingface.co/Kaleto/Llama-3.3-70B-Instruct-NVFP4) |
| TheDrummer/Fallen-Command-A-111B-v1.1 (Cohere2 / Command-A, 64 layers) | 3-shard (30/30/4) | ✅ [`Kaleto/Fallen-Command-111B-NVFP4`](https://huggingface.co/Kaleto/Fallen-Command-111B-NVFP4) |

Note: the releases above were produced with v0.1, whose Phase 3 did not actually collect activation statistics (see Phase 2 above). Their NVFP4 weights are calibrated; their activation global scales are the `input_scale = 1.0` placeholder with `input_activations.dynamic = true`. Runs with the current code export calibrated `input_scale` values instead.

Memory ceilings (per shard):
- ≤105B on 128 GB UMA — comfortable in 2-shard mode
- 105–123B class — memory-tight but works in 2-shard with streaming-load; **3-shard via `--shard-layers a,b,c` is the comfortable path** and was used for Behemoth and Fallen-Command above
- \>130B class — would need either further chunking or a fourth node

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
