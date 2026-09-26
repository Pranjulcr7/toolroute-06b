# STATUS

_Last updated: 2026-09-26_

| Item | State |
|---|---|
| Dataset v1 (1,991 / 300 / 300), validation, leakage tests | implemented, locally tested |
| Keyword-router baseline (val, test once) + end-to-end (21 episodes) | measured |
| Metrics, prompt format, harness adapter | implemented, tested |
| Training / inference code (TRL 1.14, PEFT 0.21, Transformers 5.17) | pipeline-verified on CPU with a tiny random model (`scripts/cpu_pipeline_smoke.py`); this validates the code path only and says nothing about model quality |
| Base Qwen3-0.6B evaluation | **not run**: huggingface.co blocked in the build environment, no GPU |
| LoRA training | **not run**: needs a GPU → `notebooks/train_eval_colab.ipynb` |
| HF dataset / adapter publication | **not published** (dataset card ready in `cards/`; adapter card is a template) |
| GitHub repo | pushed to `main`; CI green (lint, tests, byte-for-byte dataset regeneration) |
| Phase 1 dependency | installs from GitHub at commit `4b8cef2` (verified from a fresh clone in a clean venv; no local paths) |

## Next steps
1. Run the notebook on a Colab T4. Record the base revision sha, the smoke-job time estimate, and all train/eval outputs; commit `results/`.
2. Fill `cards/MODEL_CARD.md` with the measured numbers only. Publish the dataset + adapter, then run the fresh-process load check.
3. If the adapter does not beat the base model on `investigate` / held-out structures / end-to-end, record that as the finding.
