"""LoRA supervised fine-tuning with TRL's SFTTrainer (conversational prompt-completion, completion-only loss).

Checkpoint selection rule: the checkpoint with the lowest validation loss (load_best_model_at_end).
The test split is not touched here.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .generate import load

BASE_MODEL = "Qwen/Qwen3-0.6B"
# Resolved from the Hub on 2026-09-26; the notebook re-resolves and records the exact sha it used.
BASE_REVISION = "main"


@dataclass
class TrainConfig:
    base_model: str = BASE_MODEL
    revision: str = BASE_REVISION
    data_dir: str = "data/v1"
    out_dir: str = "runs/lora-v1"
    epochs: float = 2.0
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_length: int = 2048
    per_device_batch: int = 4
    grad_accum: int = 4
    eval_steps: int = 50
    seed: int = 0
    qlora: bool = False
    gradient_checkpointing: bool = True
    max_train_examples: int | None = None  # for smoke runs
    max_eval_examples: int | None = None


def to_prompt_completion(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"prompt": r["messages"], "completion": [{"role": "assistant", "content": r["completion"]}],
             "chat_template_kwargs": {"enable_thinking": False}} for r in rows]


def verify_masking(dataset, tokenizer, rows: list[dict[str, Any]], n: int = 5) -> dict[str, Any]:
    """Check on actual tokenized examples: only the target JSON + end-of-turn token are trained, EOS is present once."""
    eos = tokenizer.eos_token
    report = []
    for i in range(min(n, len(dataset))):
        item = dataset[i]
        ids = item["input_ids"]
        if "labels" in item:  # TRL >= 1.x folds completion_mask into labels (-100 = not trained)
            mask = [int(label != -100) for label in item["labels"]]
        else:
            mask = item["completion_mask"]
        trained = tokenizer.decode([t for t, m in zip(ids, mask, strict=True) if m])
        masked = tokenizer.decode([t for t, m in zip(ids, mask, strict=True) if not m])
        ok = (trained.strip().startswith(rows[i]["completion"]) and trained.count(eos) == 1
              and rows[i]["completion"] not in masked)
        report.append({"i": i, "ok": ok, "trained_text": trained, "prompt_tail": masked[-60:]})
    if not all(r["ok"] for r in report):
        raise AssertionError(f"loss-mask check failed: {json.dumps(report, indent=1)[:3000]}")
    return {"checked": len(report), "all_ok": True, "example": report[0]}


def _versions() -> dict[str, str]:
    out = {}
    for p in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "agent-reliability-lab"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = "missing"
    return out


def run(cfg: TrainConfig, model=None, tokenizer=None) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_rows = load(Path(cfg.data_dir) / "train.jsonl")[: cfg.max_train_examples]
    val_rows = load(Path(cfg.data_dir) / "val.jsonl")[: cfg.max_eval_examples]

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()  # e.g. T4 does not support bf16
    fp16 = cuda and not bf16
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, revision=cfg.revision)
    if model is None:
        kwargs: dict[str, Any] = {"revision": cfg.revision,
                                  "dtype": torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32}
        if cfg.qlora:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                               bnb_4bit_compute_dtype=kwargs["dtype"])
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)

    args = SFTConfig(
        output_dir=str(out / "checkpoints"), num_train_epochs=cfg.epochs, learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_batch, per_device_eval_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum, lr_scheduler_type="cosine",
        warmup_steps=0.05,  # a float < 1 is a fraction of total steps in transformers 5
        logging_steps=10, eval_strategy="steps", eval_steps=cfg.eval_steps, save_strategy="steps",
        save_steps=cfg.eval_steps, save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, max_length=cfg.max_length,
        bf16=bf16, fp16=fp16, gradient_checkpointing=cfg.gradient_checkpointing and cuda, seed=cfg.seed,
        completion_only_loss=True, report_to="none",
    )
    peft_config = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                             target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(to_prompt_completion(train_rows)),
                         eval_dataset=Dataset.from_list(to_prompt_completion(val_rows)), processing_class=tokenizer,
                         peft_config=peft_config)
    mask_report = verify_masking(trainer.train_dataset, tokenizer, train_rows)
    lengths = [len(x["input_ids"]) for x in trainer.train_dataset]
    if len(trainer.train_dataset) != len(train_rows) or max(lengths) >= cfg.max_length:
        raise ValueError(f"examples were truncated or dropped (max len {max(lengths)} vs limit {cfg.max_length}); "
                         "raise max_length rather than training on cut-off targets")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    wall = time.time() - t0
    adapter_dir = out / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    record = {
        "config": asdict(cfg), "versions": _versions(), "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "precision": "bf16" if bf16 else "fp16" if fp16 else "fp32", "qlora": cfg.qlora,
        "optimizer": args.optim, "effective_batch": cfg.per_device_batch * cfg.grad_accum,
        "train_examples": len(train_rows), "val_examples": len(val_rows),
        "seq_len": {"max": max(lengths), "mean": round(sum(lengths) / len(lengths), 1), "limit": cfg.max_length},
        "data_sha256": {s: sha(Path(cfg.data_dir) / f"{s}.jsonl") for s in ("train", "val")},
        "wall_time_s": round(wall, 1),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None,
        "train_loss": result.training_loss, "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric, "log_history": trainer.state.log_history,
        "loss_mask_check": mask_report, "checkpoint_selection": "lowest eval_loss on val",
    }
    (out / "train_record.json").write_text(json.dumps(record, indent=1, default=str))
    return record
