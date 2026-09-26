"""Batched greedy generation with identical settings for the base model and base + adapter."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

GEN = {"max_new_tokens": 160, "do_sample": False}


def load(base: str, revision: str, adapter: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32
    tok = AutoTokenizer.from_pretrained(adapter or base, revision=None if adapter else revision)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype=dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.to("cuda" if cuda else "cpu").eval()
    return model, tok


def generate(model, tok, examples: list[dict[str, Any]], batch_size: int = 8) -> tuple[dict[str, str], dict[str, Any]]:
    import torch

    prompts = [tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for ex in examples]
    out: dict[str, str] = {}
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(0, len(prompts), batch_size):
        batch = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**batch, **GEN, pad_token_id=tok.pad_token_id)
        for ex, g in zip(examples[i:i + batch_size], gen, strict=True):
            out[ex["id"]] = tok.decode(g[batch["input_ids"].shape[1]:], skip_special_tokens=True)
    stats = {"examples": len(examples), "wall_time_s": round(time.time() - t0, 1), "generation": GEN,
             "batch_size": batch_size, "device": torch.cuda.get_device_name(0) if cuda else "cpu",
             "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None}
    return out, stats


def save(preds: dict[str, str], stats: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"stats": stats, "predictions": preds}, indent=1))
