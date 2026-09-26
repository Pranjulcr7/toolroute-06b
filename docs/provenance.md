# Provenance

- **Base model:** Qwen/Qwen3-0.6B (Apache-2.0). Chosen as a budget option, not as the newest or best model.
  Its `tokenizer_config.json` was read on 2026-09-26 (EOS `<|im_end|>`; the template inserts an empty
  `<think></think>` block for the final assistant turn and for `enable_thinking=False` generation). The exact
  revision sha is resolved and recorded by the notebook at training time.
- **Simulator and tool schemas:** imported from agent-reliability-lab at commit
  `4b8cef229bfd5d9db801f1483af164be4749636f`. The only change made to it for this project was an optional
  `pools`/`split` argument on `scenarios.build`, so this dataset uses its own entity names.
- **Data:** generated entirely by `src/toolroute/generate.py` (seed 0). No external datasets and no teacher models.
- **Original to this repository:** everything under `src/toolroute/`, `tests/`, `scripts/`, `notebooks/`, `cards/`, `docs/`.

## AI-assisted development

This project was developed with AI coding assistance (Claude Code) under the author's direction. Commits are
authored under the author's name without per-commit co-author trailers; this section is the disclosure.
