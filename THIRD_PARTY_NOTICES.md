# Third-party notices

No third-party source code is copied into this repository. Dependencies are imported:

| Component | Version | License | Use |
|---|---|---|---|
| [agent-reliability-lab](https://github.com/Pranjulcr7/agent-reliability-lab) | commit `4b8cef2` | Apache-2.0 (same author) | simulator, tool schemas, harness for end-to-end checks |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | revision recorded at training time | Apache-2.0 | base checkpoint, tokenizer, chat template (downloaded, not redistributed) |
| [TRL](https://github.com/huggingface/trl) | 1.14.0 | Apache-2.0 | `SFTTrainer` |
| [PEFT](https://github.com/huggingface/peft) | 0.21.0 | Apache-2.0 | LoRA |
| [Transformers](https://github.com/huggingface/transformers) | 5.17.0 | Apache-2.0 | model/tokenizer loading, generation |
| [Datasets](https://github.com/huggingface/datasets) | 5.0.1 | Apache-2.0 | dataset objects |
| [Accelerate](https://github.com/huggingface/accelerate) | 1.15.0 | Apache-2.0 | training backend |
| [PyTorch](https://pytorch.org) | 2.14.0 (tested on CPU) | BSD-3-Clause | tensors |
| [smolagents](https://github.com/huggingface/smolagents) (via arlab) | 1.26.0 | Apache-2.0 | agent runtime for end-to-end checks |

`src/toolroute/recoverable.yaml` is a copy of `configs/recoverable.yaml` from agent-reliability-lab, same author,
at the pinned commit.
