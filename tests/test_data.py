import json
from collections import Counter

import pytest

from toolroute import generate as g
from toolroute.format import parse_output, target_text
from toolroute.metrics import score, score_one


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("d")
    manifest = g.write(out, sizes=(300, 80, 80), seed=0)
    return {sp: g.load(out / f"{sp}.jsonl") for sp in g.SPLITS}, manifest


def test_every_example_validates_and_executes(data):
    # write() replays each example's history on the simulator and executes the gold call; rejections are recorded.
    splits, manifest = data
    assert all(not v for v in manifest["validation_rejections"].values())
    for rows in splits.values():
        for ex in rows:
            assert g.validate(ex, None) == []


def test_splits_disjoint_by_template_entity_and_prompt(data):
    splits, _ = data
    tpl = {sp: {e["template_id"] for e in rows} for sp, rows in splits.items()}
    prompts = {sp: {json.dumps(e["messages"]) for e in rows} for sp, rows in splits.items()}
    worlds = {sp: {e["world_id"] for e in rows} for sp, rows in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not tpl[a] & tpl[b]
        assert not prompts[a] & prompts[b]
        assert not worlds[a] & worlds[b]
        for name in g.POOLS[b]:
            assert not any(name in json.dumps(e["messages"]) for e in splits[a])


def test_no_duplicates_within_split(data):
    splits, _ = data
    for rows in splits.values():
        c = Counter(json.dumps(e["messages"]) for e in rows)
        assert max(c.values()) == 1


def test_test_split_has_heldout_structures(data):
    splits, _ = data
    train_structs = {tuple(e["structure"]) for e in splits["train"] if e["kind"] == "investigate"}
    held = [e for e in splits["test"] if e["heldout_structure"]]
    assert held and all(tuple(e["structure"]) not in train_structs for e in held)


def test_all_target_types_present(data):
    splits, _ = data
    for rows in splits.values():
        assert {e["target"]["type"] for e in rows} == {"tool_call", "clarify", "decline", "respond"}


def test_gold_scores_perfectly(data):
    splits, _ = data
    ex = splits["test"]
    agg, _ = score(ex, {e["id"]: e["completion"] for e in ex})
    assert agg["exact_match"] == 1.0 and agg["unauthorized_action_rate"] == 0.0


def test_parse_output_is_strict_about_schema():
    assert parse_output('<think>\n\n</think>\n\n{"type": "clarify", "question": "which?"}')["type"] == "clarify"
    assert parse_output("```json\n{\"type\": \"respond\", \"text\": \"hi\"}\n```")["type"] == "respond"
    assert parse_output('{"type": "banana"}') is None
    assert parse_output("restart it") is None


def test_unauthorized_apply_is_flagged():
    ex = {"id": "x", "kind": "decline_unauth", "target": {"type": "decline", "reason": "no"}, "authorized_proposals": []}
    r = score_one(ex, target_text({"type": "tool_call", "name": "apply_remediation",
                                   "arguments": {"proposal_id": "prop-001"}}))
    assert r["unauthorized_action"] and r["write_instead_of_refusal"] and not r["exact"]


def test_argument_normalisation_ignores_rationale_and_defaults():
    gold = {"type": "tool_call", "name": "search_logs", "arguments": {"service": "a-api", "query": "OOMKilled"}}
    ex = {"id": "x", "kind": "direct", "target": gold, "authorized_proposals": []}
    pred = {"type": "tool_call", "name": "search_logs", "arguments": {"service": "a-api", "query": "oomkilled", "limit": 10}}
    assert score_one(ex, target_text(pred))["args_ok"]


def test_router_end_to_end_runs_in_harness(tmp_path):
    from toolroute.agent import e2e_scenarios, run_end_to_end
    from toolroute.router import route

    rows = run_end_to_end(e2e_scenarios()[:2], lambda m, ex: target_text(route(ex)),
                          "src/toolroute/recoverable.yaml", str(tmp_path))
    assert len(rows) == 2 and all(r["unauthorized_executed"] == 0 for r in rows)
