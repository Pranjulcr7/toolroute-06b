"""CPU smoke test of the training/inference pipeline with a TINY RANDOM model and a stand-in tokenizer.

Purpose: prove the code path (TRL prompt-completion formatting, loss-mask check, LoRA training, adapter save,
adapter reload, batched generation, scoring) runs with the pinned library versions. It measures nothing about
model quality and must never be reported as a training result. The real run uses Qwen/Qwen3-0.6B on a GPU.
"""

import json
import sys
import tempfile
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config

from toolroute import predict
from toolroute.generate import load
from toolroute.metrics import score
from toolroute.train import TrainConfig, run

# Mirrors the relevant behaviour of Qwen3's template: empty think block before the final assistant turn and
# after the generation prompt when enable_thinking is false.
TEMPLATE = (
    "{%- for m in messages %}{%- if m.role == 'assistant' and loop.last %}"
    "{{ '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + m.content + '<|im_end|>\\n' }}"
    "{%- else %}{{ '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{ '<think>\\n\\n</think>\\n\\n' }}{%- endif %}"
    "{%- endif %}"
)


def main(data_dir="data/v1", out_dir=None):
    out = Path(out_dir or tempfile.mkdtemp(prefix="toolroute-smoke-"))
    rows = load(Path(data_dir) / "train.jsonl")[:200]
    texts = [m["content"] for r in rows for m in r["messages"]] + [r["completion"] for r in rows]
    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    bpe.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=4000, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.chat_template = TEMPLATE
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=True)
    base_dir = out / "tiny-base"
    AutoModelForCausalLM.from_config(cfg).save_pretrained(base_dir)
    tok.save_pretrained(base_dir)

    record = run(TrainConfig(base_model=str(base_dir), revision=None, data_dir=data_dir, out_dir=str(out / "run"),
                             epochs=1, per_device_batch=4, grad_accum=1, eval_steps=4, max_train_examples=32,
                             max_eval_examples=8, max_length=2048),
                 model=AutoModelForCausalLM.from_pretrained(base_dir), tokenizer=tok)
    model, tok2 = predict.load(str(base_dir), None, adapter=str(out / "run" / "adapter"))
    val = load(Path(data_dir) / "val.jsonl")[:4]
    preds, stats = predict.generate(model, tok2, val, batch_size=4)
    agg, _ = score(val, preds)
    summary = {"loss_mask_check": record["loss_mask_check"]["all_ok"],
               "trained_text_example": record["loss_mask_check"]["example"]["trained_text"],
               "prompt_tail_example": record["loss_mask_check"]["example"]["prompt_tail"],
               "seq_len": record["seq_len"], "train_wall_time_s": record["wall_time_s"],
               "adapter_files": sorted(p.name for p in (out / "run" / "adapter").iterdir()),
               "reload_and_generate": stats["examples"], "parse_valid_of_random_model": agg["parse_valid"],
               "versions": record["versions"]}
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main(*sys.argv[1:])
