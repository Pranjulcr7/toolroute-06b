---
# TEMPLATE. Do not publish until training and evaluation have finished and every TODO is replaced by a measured value.
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
library_name: peft
tags: [lora, tool-use, function-calling, synthetic-data]
datasets: [PranjulGupta/incident-tool-routing-v1]
---

# ToolRoute-Qwen3-0.6B-LoRA

A **LoRA adapter** fine-tuned from [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) at revision `TODO_SHA`.
It is not a foundation model and was not trained from scratch. The base model's Apache-2.0 license applies to it.

**Intended scope (narrow):** given a request, eight fictional incident-response tool schemas and prior tool
results, return one JSON decision (`tool_call` / `clarify` / `decline` / `respond`). It has not been evaluated
outside the synthetic `incident-tool-routing-v1` distribution and is not a general assistant.

## Training
TODO from `train_record.json`: device, precision, LoRA r/alpha/dropout/targets, epochs, lr, effective batch,
max length, optimizer, seed, wall time, peak memory, data sha256, package versions, checkpoint selection
(lowest validation loss).

## Evaluation (test split, used once; identical greedy settings for base and adapter)
| model | exact match (95% CI) | tool choice | arguments | clarify F1 | decline F1 | unauthorized actions | investigate | held-out structures | end-to-end (21) |
|---|---|---|---|---|---|---|---|---|---|
| keyword router | 0.800 (0.75–0.84) | 0.727 | 0.727 | 1.00 | 1.00 | 0 | 0.517 | 0.531 | 9/21 |
| base | TODO | | | | | | | | |
| + this adapter | TODO | | | | | | | | |

## Failure modes
TODO: at least three concrete failing test examples with model output.

## Load
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("PranjulGupta/ToolRoute-Qwen3-0.6B-LoRA")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", revision="TODO_SHA", dtype=torch.float16)
model = PeftModel.from_pretrained(base, "PranjulGupta/ToolRoute-Qwen3-0.6B-LoRA")
# Build prompts with toolroute.format.messages(...) and apply_chat_template(..., enable_thinking=False)
```

## Reproduce
`notebooks/train_eval_colab.ipynb` in https://github.com/Pranjulcr7/toolroute-06b.
