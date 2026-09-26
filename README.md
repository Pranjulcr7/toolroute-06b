# toolroute-06b

**Question:** can a small adapted model improve tool selection, argument correctness and appropriate
clarification/refusal on *unseen* incident scenarios?

This repo is the training and evaluation companion to
[agent-reliability-lab](https://github.com/Pranjulcr7/agent-reliability-lab). It imports that project's simulator
and tool schemas as a pinned dependency rather than copying them. The task is deliberately narrow. Given a request, the
eight tool schemas and prior tool observations, output **one** JSON decision:

```json
{"type": "tool_call", "name": "<tool>", "arguments": {...}}
{"type": "clarify", "question": "..."}
{"type": "decline", "reason": "..."}
{"type": "respond", "text": "..."}
```

It is not a general reasoning model, and no chain-of-thought is trained or produced.

> **Status (2026-09-26): the model is NOT trained yet.** No GPU was available in the build environment, and
> huggingface.co was unreachable from it, so the base Qwen checkpoint could not be downloaded or evaluated either.
> What exists and is tested: the dataset (v1), the non-LLM baseline, metrics, the end-to-end harness check,
> and the training/inference code, verified end to end on CPU with a *tiny random* model (a pipeline check, not a result).
> `notebooks/train_eval_colab.ipynb` runs the real training and evaluation on a GPU. No adapter or model card has been published.

## Dataset v1 (`data/v1`, synthetic, fictional)

| split | n | tool_call | clarify | decline | respond | held-out-structure examples |
|---|---|---|---|---|---|---|
| train | 1,991 | 1,372 | 140 | 260 | 219 | – |
| val | 300 | 210 | 21 | 39 | 30 | – |
| test | 300 | 209 | 21 | 39 | 31 | 64 |

- **Executable labels.** Each example's history is really executed on the arlab simulator. The gold next step comes
  from oracle code. Every gold tool call is schema-validated and executed against a replay of its own history;
  0 examples were rejected.
- **Leakage controls.** Request templates, service names (entity pools) and worlds/trajectories are disjoint across
  splits; all steps of one multi-step trajectory stay in one split. Duplicates are 0 within and across splits.
  The test split adds incident structures never seen in training, for example an injected instruction that asks for an
  action inside the grant scope. Checked by `tests/test_data.py`.
- Kinds: direct reads, remediation proposals, apply-after-authorized-proposal, decline (unauthorized proposal,
  unsupported/destructive request, injected "pre-approved" proposal), clarify (missing service, missing replica
  count), respond (answer from observations, no tool needed), and multi-step investigation (40%).

## Results so far (measured)

| model | split | exact match (95% CI) | tool choice | arguments | clarify F1 | decline F1 | unauthorized actions | multi-step kind | held-out structures |
|---|---|---|---|---|---|---|---|---|---|
| keyword router (non-LLM) | test | 0.800 (0.75–0.84) | 0.727 | 0.727 | 1.00 | 1.00 | 0 | 0.517 | 0.531 |
| Qwen3-0.6B (base) | – | not run | | | | | | | |
| Qwen3-0.6B + LoRA | – | not trained | | | | | | | |

End-to-end in the agent-reliability-lab harness (`recoverable` config, 21 held-out multi-step episodes): the keyword
router succeeds in **9/21**. It escalates correctly on unauthorized/unfixable incidents but never diagnoses or
remediates.

**What this already says about the dataset:** single-step kinds are nearly keyword-solvable (router exact match
0.97–1.0 on every single-step kind). The informative part is the multi-step investigation, where the next step
depends on reading observations. Any model gain should be judged mainly on `by_kind.investigate`,
`heldout_structure` and end-to-end success, not on overall exact match. See
[docs/engineering-notes.md](docs/engineering-notes.md).

## Reproduce

```bash
uv venv && uv pip install -e ".[dev]"
toolroute generate --out data/v1                       # deterministic; CI checks the committed files byte for byte
toolroute baseline --split test --out results/router-test.json
toolroute e2e --model router --instances 3 --out results/e2e-router.json
python scripts/cpu_pipeline_smoke.py                   # needs the [train] extra; tiny random model, pipeline check only
```

GPU (Colab T4 or better): open [`notebooks/train_eval_colab.ipynb` in Colab](https://colab.research.google.com/github/Pranjulcr7/toolroute-06b/blob/main/notebooks/train_eval_colab.ipynb). It pins the base revision, runs a smoke
job first, estimates the full run time, trains (LoRA r=16, all linear layers, 2 epochs, lr 2e-4 cosine, effective
batch 16, max length 2048, fp16 on T4 and bf16 where supported, QLoRA off unless memory requires it), selects the checkpoint by
lowest validation loss, and evaluates base and adapter once on test with identical greedy settings.

## Built vs. reused

- **Reused:** Qwen3-0.6B (Apache-2.0) as the base checkpoint; TRL `SFTTrainer`, PEFT LoRA, Transformers, Datasets;
  the simulator and tool schemas from agent-reliability-lab (pinned commit). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **Original:** the task format, oracle-labelled generator with leakage controls, metrics, keyword baseline,
  harness adapter, training wrapper with loss-mask/truncation checks, and notebook. See [docs/provenance.md](docs/provenance.md).

## Limitations

Template-generated English requests written by one author, who also wrote the keyword router; 3 remediation actions;
one runbook; 9 incident structures. Claims only hold inside this scope. Respond/clarify/decline *text* is not scored,
only the decision type. No constrained decoding has been tested; if it is added, it should apply to both base and adapter
and be reported as a separate factor.

Built with AI coding assistance; see [docs/provenance.md](docs/provenance.md#ai-assisted-development). License: Apache-2.0
for code and data. The base model keeps its own Apache-2.0 license.
