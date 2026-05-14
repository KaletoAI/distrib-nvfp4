# Debugging notes — modelopt 0.43 NVFP4 export + vLLM serving

This document is the long-form version of "things I had to fix to make the Anubis-Pro-105B-NVFP4 model actually serve correctly under vLLM". The pipeline in this repo applies every fix below automatically. If you build something analogous from scratch, or if the modelopt version evolves and breaks any of these workarounds, this is what to know.

## Symptom that brought everything to a halt: vLLM serves the model and returns `!!!!!!!!!!`

After the pipeline finishes, you set up `vllm serve` against the output dir, send a chat completion request, and get back:

```json
{"choices": [{"message": {"content": "!!!!!!!!!!!!!!!!!!!!"}}]}
```

The vLLM startup log shows `FlashInferCutlassNvFp4LinearKernel` being selected, no errors, no fallback warnings. It looks like everything is fine. It is not.

If you've never seen this before: it means the dequantization math is wrong somewhere. The model is producing uniform output (typically the token with the lowest token-id-as-string after the chat template), which is what happens when every logit comes out roughly equal — i.e. when the per-Linear `alpha` parameter (which dequantizes the FP4 weights) is built from garbage values.

There are six distinct fixes needed before the output is coherent. The first three are about getting Phase 6 (export) to run at all. The last three are about making vLLM's loader produce sensible math from the resulting files.

## Phase 6 — getting the export to complete

### Fix 1: `Padding_idx must be within num_embeddings`

modelopt's per-layer export uses a 1-layer template — it builds a `LlamaForCausalLM` (or whatever arch) with `num_hidden_layers=1` and `vocab_size` shrunk to a tiny value to keep the embed/lm_head dummies small. If you leave `pad_token_id` at its source-model value (e.g. 128004 for Llama 3.3) while shrinking `vocab_size`, `nn.Embedding.__init__` asserts:

```
AssertionError: Padding_idx must be within num_embeddings
```

**Fix:** Before `AutoModelForCausalLM.from_config(cfg_t, ...)`, also set `cfg_t.pad_token_id = cfg_t.bos_token_id = cfg_t.eos_token_id = None`.

### Fix 2: `indexSelectSmallIndex: srcIndex < srcSelectDimSize` (CUDA assertion)

The "shrink vocab_size to 1" trick conflicts with modelopt's internal `requantize_resmooth_fused_llm_layers`, which calls `_collect_shared_input_modules` with a `dummy_forward_fn` defined as:

```python
def llm_dummy_forward():
    fake_input = torch.ones([1, 2], dtype=torch.long).to(model.device)
    ...
    model(fake_input)
```

`torch.ones([1, 2], dtype=long)` produces token IDs `[1, 1]`. With `vocab_size=1`, only index 0 is valid, and the embedding lookup CUDA-asserts inside `Indexing.cu`.

**Fix:** `cfg_t.vocab_size = 2` (smallest value that satisfies the current modelopt dummy). Plus parameterize the dummy `nn.Embedding(...)` and `nn.Linear(..., out)` with `cfg_t.vocab_size` instead of hard-coding `1`.

### Fix 3: `AttributeError: 'NoneType' object has no attribute '_num_bits'`

If you try to be clever and `del`/`None`-out the `TensorQuantizer._calibrator` attributes before calling `mte.export_hf_checkpoint` (to free a few GB of memory), modelopt's internal `set_quantizer_by_cfg_context` context manager fails on `__exit__` because it tries `setattr(self._calibrator, key, ...)` on what is now `None`.

```
File "modelopt/torch/quantization/conversion.py", line 334, in set_quantizer_by_cfg_context
    module.set_from_modelopt_state(original_attributes[name], properties_only=True)
File "modelopt/torch/quantization/nn/modules/tensor_quantizer.py", line 1235, in set_from_modelopt_state
    setattr(self._calibrator, key, getattr(self, key))
AttributeError: 'NoneType' object has no attribute '_num_bits'
```

**Fix:** Don't clear `_calibrator` before `export_hf_checkpoint`. Drop the `_quant_wrapper` if you have one (to free a few GB), but leave the per-quantizer calibrator objects alone. Find memory savings somewhere else (the Phase-5.5 disk-eviction in this pipeline is the better answer).

## vLLM serving — getting the output to be coherent

### Fix 4: Phase-6 dummy `vocab_size=2` leaks into the merged `config.json`

Phase 6 wraps each layer in a 1-layer template with `cfg_t.vocab_size=2` and `pad_token_id=None`. When you then take that template's `config.json` (it has the `quantization_config` block, which is what you want) and use it as the basis for the merged final config, you also inherit `vocab_size=2` and `pad/bos/eos=None`.

vLLM then allocates `Embedding(num_embeddings=2)` for the model, tries to load the real `model.embed_tokens.weight` of shape `[128256, 8192]`, and asserts:

```
File "vllm/model_executor/layers/vocab_parallel_embedding.py", line 463, in weight_loader
    assert loaded_weight.shape[output_dim] == self.org_vocab_size
AssertionError
```

The actual safetensors are correct — only the config is wrong.

**Fix in `merge_exports`:** read `source_model/config.json`, restore `vocab_size`, `pad_token_id`, `bos_token_id`, `eos_token_id`, `architectures` from there, write into the merged config.

### Fix 5: `input_activations.dynamic: false` is a lie

modelopt 0.43's NVFP4 export writes `quantization_config.config_groups.group_0.input_activations.dynamic = false` into `hf_quant_config.json` and `config.json`. This tells vLLM "I have pre-computed static input scales for the activations, please use them at inference time".

modelopt 0.43 then **does not actually emit any `input_scale` keys** in the safetensors. The flag value is inconsistent with the safetensors content.

vLLM reads `dynamic=false`, registers a `PerTensorScaleParameter` named `input_scale` in each quantized Linear (uninitialized memory), looks for the `input_scale` key in the safetensors, doesn't find it, leaves the parameter uninitialized, and proceeds to use it as if it had been loaded — which is what causes the garbage output downstream.

**Fix in `merge_exports`:** patch `input_activations.dynamic = true` in the merged config groups. This tells vLLM to use per-token dynamic input quantization at inference, which doesn't need static `input_scale` keys.

### Fix 6: Even with `dynamic=true`, vLLM's loader still wants `input_scale` keys present

This one took the longest to find. Reading `vllm/model_executor/layers/quantization/modelopt.py`:

```python
def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    if (torch.unique(layer.input_scale).numel() != 1
        or torch.unique(layer.weight_scale_2).numel() != 1):
        logger.warning_once(...)

    # Rename ModelOpt checkpoint names to standardized names
    input_global_scale = layer.input_scale.max().to(torch.float32)
    layer.input_global_scale = Parameter(input_global_scale, requires_grad=False)
    del layer.input_scale

    weight_global_scale = layer.weight_scale_2.max().to(torch.float32)
    layer.weight_global_scale = Parameter(weight_global_scale, requires_grad=False)
    del layer.weight_scale_2

    # Pre-compute alpha and inverse for runtime quantization
    layer.alpha = Parameter(
        layer.input_global_scale * layer.weight_global_scale, requires_grad=False
    )
```

vLLM's loader unconditionally:

1. Registers `input_scale` as a parameter (it ends up uninitialized — `torch.empty(...)`).
2. Reads `layer.input_scale.max()` and assigns it to `input_global_scale`.
3. Multiplies it into `layer.alpha = input_global_scale * weight_global_scale`, which is the value the NVFP4 kernel uses as the per-Linear dequant constant.

If you don't have an `input_scale` key in your safetensors, the `Parameter` stays at its `torch.empty` initial values — random memory. `.max()` of that random memory is then multiplied into alpha. Wrong alpha → all FP4 weights dequantize to nonsensical magnitudes → uniform output → `!!!!!!!!`.

**Fix in `merge_exports`:** after merging, inject an `input_scale = torch.tensor(1.0, dtype=torch.float32)` (scalar) for every `<prefix>.weight_scale_2` key found. 1.0 is the identity scale — it's correct for dynamic input quantization (the NVFP4 kernel applies the runtime per-token scale itself; the static `input_scale` is essentially overridden by the runtime path).

For a 120-layer Llama with 7 quantized Linears per layer that's 840 keys, ~84 KB total. Tiny.

## Identifying the right "fast path" on Spark

A separate gotcha worth flagging: many NVFP4 quantization writeups (including older versions of this README) point at `FlashInferCutlassNvFp4LinearKernel` in the vLLM startup log as the marker for "you're on the NVFP4 fast path". This is misleading on DGX Spark / GB10.

The Spark community has converged on the MARLIN-backend port of NVFP4 GEMM (see [Avarok-Cybersecurity/dgx-vllm](https://github.com/Avarok-Cybersecurity/dgx-vllm)). Correct startup-log signals on Spark:

- For attention: `Using AttentionBackendEnum.FLASHINFER backend.`
- For MoE models (not relevant for the dense Llama/Mistral models this pipeline targets): `Using 'MARLIN' NvFp4 MoE backend`
- Env vars to set for the proper Spark stack: `VLLM_NVFP4_GEMM_BACKEND=marlin`, `VLLM_TEST_FORCE_FP8_MARLIN=1`, `VLLM_MARLIN_USE_ATOMIC_ADD=1`, and serve with `--attention-backend flashinfer`

Or just use the `avarok/dgx-vllm-nvfp4-kernel:v23` Docker image, which has these baked in.

## Memory shape on a 2-Spark cluster

For a Llama/Mistral-class model split across 2 nodes, expect these per-shard UMA peaks (Behemoth-X-123B figures):

| Phase | Per-shard peak | Notes |
|---|---|---|
| 1 (load) | ~123 GB | streaming `safe_open` keeps the spike to one tensor at a time (~600 MB max for gate_proj/up_proj), so the residual is just the materialized layers |
| 2-4 (quantize/calibrate/finalize) | ~123 GB | quantizer wrappers add <1 GB |
| 5.5 (eviction) | drops to ~5 GB | every layer goes to disk as `torch.save(layer.cpu())`, in-memory ref becomes `nn.Identity` |
| 6 (export) | ~13 GB | one layer loaded from disk + ~10 GB export overhead per iter |
| 7 (merge, on driver) | <10 GB | driver-side, not on actors |

Phase 1 is the tightest — if your model's per-shard size exceeds ~119 GB (the usable UMA on a 128 GB Spark after kernel/OS), you'll hit OOM even with streaming load. Above that ceiling you'd need a 3-way split or a different sharding strategy.

## Ray actor heartbeat watchdog

Large model loads can pin Python in C-level loops long enough that the GCS ↔ actor heartbeat times out and the actor is marked dead, mid-load:

```
ray.exceptions.ActorUnavailableError: ... keepalive watchdog timeout
rpc_code: 14
```

**Fix:** at Ray cluster startup, set generous timeouts:

```bash
export RAY_health_check_period_ms=60000
export RAY_health_check_timeout_ms=120000
export RAY_health_check_failure_threshold=10
export RAY_gcs_rpc_server_reconnect_timeout_s=300
```

Together these give ~10 min of unresponsiveness tolerance, which has covered the 88-layer Behemoth Phase-1 load with margin.
