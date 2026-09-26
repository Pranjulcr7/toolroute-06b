"""toolroute CLI: generate | baseline | predict | score | train | e2e."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generate import load, write
from .metrics import score


def _split(a):
    return load(Path(a.data) / f"{a.split}.jsonl")


def cmd_generate(a):
    m = write(a.out, (a.train, a.val, a.test), a.seed)
    print(json.dumps({k: {"n": v["n"], "sha256": v["sha256"][:12]} for k, v in m["splits"].items()}))


def _write_score(examples, preds, out, extra=None):
    agg, rows = score(examples, preds)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"summary": agg, **(extra or {}), "rows": rows}, indent=1))
    keys = ("exact_match", "exact_match_ci95", "parse_valid", "type_accuracy", "tool_choice_accuracy",
            "argument_accuracy", "unauthorized_action_rate")
    print(json.dumps({k: agg[k] for k in keys}))


def cmd_baseline(a):
    from .router import predict

    ex = _split(a)
    _write_score(ex, predict(ex), a.out, {"model": "keyword-router (non-LLM)", "split": a.split})


def cmd_predict(a):
    from . import predict as P

    ex = _split(a)[: a.limit]
    model, tok = P.load(a.base, a.revision, a.adapter)
    preds, stats = P.generate(model, tok, ex, a.batch_size)
    P.save(preds, {**stats, "base": a.base, "revision": a.revision, "adapter": a.adapter, "split": a.split}, a.out)
    _write_score(ex, preds, Path(a.out).with_suffix(".score.json"), {"stats": stats, "split": a.split})


def cmd_score(a):
    ex = _split(a)
    preds = json.loads(Path(a.pred).read_text())["predictions"]
    _write_score([e for e in ex if e["id"] in preds], preds, a.out)


def cmd_train(a):
    from .train import TrainConfig, run

    rec = run(TrainConfig(revision=a.revision, data_dir=a.data, out_dir=a.out, epochs=a.epochs, seed=a.seed,
                          per_device_batch=a.batch, grad_accum=a.grad_accum, qlora=a.qlora))
    print(json.dumps({k: rec[k] for k in ("device", "precision", "wall_time_s", "peak_memory_gb", "best_eval_loss",
                                          "best_checkpoint", "seq_len")}, default=str))


def cmd_e2e(a):
    from .agent import e2e_scenarios, run_end_to_end
    from .format import target_text

    if a.model == "router":
        from .router import route

        def gen(msgs, ex):
            return target_text(route(ex))
    else:
        from . import predict as P

        model, tok = P.load(a.base, a.revision, a.adapter)

        def gen(msgs, ex):
            out, _ = P.generate(model, tok, [{"id": "x", "messages": msgs}], 1)
            return out["x"]

    sc = [s for i in range(a.instances) for s in e2e_scenarios_instance(i)] if a.instances > 1 else e2e_scenarios()
    rows = run_end_to_end(sc, gen, a.harness_config, str(Path(a.out).parent / "e2e_traces"))
    k = sum(r["success"] for r in rows)
    summary = {"model": a.model, "adapter": a.adapter, "n": len(rows), "success": k, "success_rate": round(k / len(rows), 4),
               "unauthorized_executed": sum(r["unauthorized_executed"] for r in rows),
               "duplicate_side_effects": sum(r["duplicate_side_effects"] for r in rows), "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))


def e2e_scenarios_instance(i):
    from arlab import scenarios as arsc

    from .agent import E2E_FAMILIES

    return [arsc.build(s, i) for s in arsc.structures() if s[0] in E2E_FAMILIES and arsc.split_of(s) == "test"]


def main(argv=None):
    p = argparse.ArgumentParser(prog="toolroute")
    sub = p.add_subparsers(required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", default="data/v1")
    g.add_argument("--train", type=int, default=2000)
    g.add_argument("--val", type=int, default=300)
    g.add_argument("--test", type=int, default=300)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(fn=cmd_generate)
    for name, fn in (("baseline", cmd_baseline), ("predict", cmd_predict), ("score", cmd_score)):
        s = sub.add_parser(name)
        s.add_argument("--data", default="data/v1")
        s.add_argument("--split", default="val", choices=["train", "val", "test"])
        s.add_argument("--out", required=True)
        if name == "predict":
            s.add_argument("--base", default="Qwen/Qwen3-0.6B")
            s.add_argument("--revision", required=True)
            s.add_argument("--adapter", default=None)
            s.add_argument("--batch-size", type=int, default=8)
            s.add_argument("--limit", type=int, default=None)
        if name == "score":
            s.add_argument("--pred", required=True)
        s.set_defaults(fn=fn)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/v1")
    t.add_argument("--revision", required=True)
    t.add_argument("--out", default="runs/lora-v1")
    t.add_argument("--epochs", type=float, default=2.0)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--grad-accum", type=int, default=4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--qlora", action="store_true")
    t.set_defaults(fn=cmd_train)
    e = sub.add_parser("e2e")
    e.add_argument("--model", default="router", choices=["router", "hf"])
    e.add_argument("--base", default="Qwen/Qwen3-0.6B")
    e.add_argument("--revision", default=None)
    e.add_argument("--adapter", default=None)
    e.add_argument("--instances", type=int, default=3)
    e.add_argument("--harness-config", default=None, help="arlab config yaml (default: bundled recoverable)")
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_e2e)
    a = p.parse_args(argv)
    if getattr(a, "harness_config", "unset") is None:
        a.harness_config = str(Path(__file__).parent / "recoverable.yaml")
    a.fn(a)


if __name__ == "__main__":
    main()
