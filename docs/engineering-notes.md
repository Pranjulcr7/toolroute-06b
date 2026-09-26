# Engineering notes

## Key tradeoffs

- **One JSON decision per step instead of free-form chat.** Every output can be scored by parsing, schema
  validation and simulator execution, and syntax validity is reported separately from task success.
- **Prompt-completion SFT with the model's own template.** TRL tokenizes `prompt` with `add_generation_prompt=True`
  and `prompt + completion` with the same `chat_template_kwargs={"enable_thinking": False}`. Only the JSON target plus
  `<|im_end|>` is trained, and training and inference see the same empty think block. `verify_masking` asserts this on
  real tokenized examples before any training step. The notebook re-checks it with the real Qwen tokenizer.
- **Fail rather than truncate.** TRL silently truncates to `max_length`, which would cut off targets; `train.run`
  refuses to start if any example reaches the limit or was dropped.
- **Checkpoint by validation loss only.** Test is evaluated once, after selection. The notebook states it.
- **A strong dumb baseline on purpose.** The keyword router exposes how template-driven the single-step data is.
  Without it, a high overall exact match from a fine-tuned model would look better than it is.

## Three failure cases (keyword router, test split; replay with `results/router-test.json` rows)

1. **Multi-step diagnosis.** On `investigate` steps after the runbook read, the router responds "escalate" instead of
   `propose_remediation(rollback_deployment, …)`. It cannot connect "deployed 9 minutes ago + NullPointer" to a
   rollback. Exact match on `investigate` is 0.52.
2. **Never remediates end to end.** Across 21 held-out harness episodes the router succeeds on all 9 that should
   end in escalation (unauthorized, unfixable) and on 0 of the 12 that need a remediation (clean, injection). It reads
   five tools and then escalates, which the grader correctly scores as a failure when a fix was authorized and needed.
   The flapping "successes" also say little: escalating without trying the authorized fix is allowed by that grader.
3. **Answering vs. calling.** `respond` precision is 0.34: the fallback "respond" branch absorbs every request that
   no keyword matches.

## Reproduce one experiment (router on held-out test)

```bash
toolroute generate --out data/v1       # hashes must match data/v1/manifest.json
toolroute baseline --split test --out results/router-test.json
python -c "import json;s=json.load(open('results/router-test.json'))['summary'];print(s['exact_match'],s['by_kind']['investigate'])"
```
