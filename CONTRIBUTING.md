# Contributing

Issues and PRs welcome — this is a single-author release and I expect the community to find edge cases I haven't hit.

## Helpful contributions

- **New architecture adapters.** The driver works out-of-the-box for any HF arch whose decoder layer has `input_layernorm` (RMSNorm-style) and either a top-level `model.rotary_emb` or `layer.self_attn.rotary_emb`. Llama, Mistral, Qwen, Gemma should all be drop-in. Cohere2, Nemotron-NAS, custom_code architectures need investigation.
- **GB10-tuned ignore lists.** Currently uses `mtq.NVFP4_DEFAULT_CFG` with only `lm_head` in the ignore list. A more nuanced ignore list (à la `saricles/...-GB10` releases) would improve quality on Spark. See [saricles' GB10-tuned recipes](https://huggingface.co/saricles) for the reference.
- **Alternative calibration datasets.** Currently `cnn_dailymail`. RP/storytelling models would likely benefit from a domain-matched mix.
- **Re-validate against newer modelopt.** The six fix list in `docs/debugging-notes.md` is dated 2026-05 against modelopt 0.43.0. If a newer release fixes any of these upstream, the corresponding workaround in `scripts/distrib_quant.py` can be removed.

## Bug reports

Please include:

- Source model (HF repo or arch info), target hardware (Spark / B100 / RTX 5090 / etc.)
- Exact CLI invocation
- Phase that failed (1 / 2 / 3 / 4 / 5 / 5.5 / 6 / 7)
- Last ~50 lines of `/tmp/distrib_quant.log`
- If Phase 6 failed: was `--resume-from-checkpoint` attempted?

## PR style

Keep things small. The driver is ~900 lines and intentionally one-file. Don't refactor for its own sake — but if a fix is genuinely modular and the diff is clean, go ahead.

For non-trivial changes (new arch adapter, new pipeline phase, etc.) open an issue first to align on the approach.
