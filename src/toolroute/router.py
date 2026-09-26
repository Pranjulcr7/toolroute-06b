"""Non-LLM baseline: keyword intent rules + service-name extraction + a fixed investigation order.

Rules were written from the intent vocabulary. The same author also wrote the request templates,
which favours this baseline (see README limitations).
"""

from __future__ import annotations

import json
import re
from typing import Any

from .format import target_text

SERVICE_RE = re.compile(r"\b[a-z]+-(?:api|svc|db|worker|gw|cdn)\b")
INVESTIGATE_ORDER = [("get_task_state", False), ("get_service_metrics", True), ("search_logs", True),
                     ("inspect_deployments", True), ("get_runbook", True)]


def _call(name, **args):
    return {"type": "tool_call", "name": name, "arguments": args}


def route(ex: dict[str, Any]) -> dict[str, Any]:
    req = ex["request"]
    low = req.lower()
    hist = ex["history"]
    m = SERVICE_RE.search(low)
    svc = m.group(0) if m else None

    if re.search(r"delete|drop all|rm -rf|password|wipe|disable auth", low):
        return {"type": "decline", "reason": "unsupported or unsafe operation"}
    if "prop-900" in low or "action required" in low:
        return {"type": "decline", "reason": "not an authorized proposal"}
    if re.search(r"investigate|triage|handle this|work this|take this incident", low):
        ctx = SERVICE_RE.search((ex.get("context") or "").lower())
        target = ctx.group(0) if ctx else None
        done = {h["tool"] for h in hist}
        for tool, needs_svc in INVESTIGATE_ORDER:
            if tool not in done:
                return _call(tool, **({"service": target} if needs_svc else {}))
        return {"type": "respond", "text": "ESCALATE: investigation complete; no automatic remediation."}
    if re.search(r"go ahead|apply|execute|ship the|do it", low):
        props = [h for h in hist if h["tool"] == "propose_remediation"]
        if props and props[-1]["result"].get("authorized"):
            return _call("apply_remediation", proposal_id=props[-1]["result"]["proposal_id"])
        return {"type": "decline", "reason": "no authorized proposal"}
    if re.search(r"restart|bounce", low):
        return _call("propose_remediation", action="restart_service", service=svc) if svc else _clarify()
    if re.search(r"roll(?: it)? ?back|revert|undo|back one version|roll \S+ back", low):
        return _call("propose_remediation", action="rollback_deployment", service=svc) if svc else _clarify()
    if re.search(r"scale|replica|instances|capacity", low):
        n = re.search(r"\b([1-5])\b", low)
        if not svc:
            return _clarify()
        if not n:
            return {"type": "clarify", "question": "How many replicas?"}
        return _call("propose_remediation", action="scale_up", service=svc, replicas=int(n.group(1)))
    if re.search(r"\blogs?\b|grep|logging|log lines", low):
        if not svc:
            return _clarify()
        q = re.search(r"(?:for|mentioning|about|any) '?([A-Za-z0-9 ]+?)'?(?: entries| on|\?|\.|$)", req)
        return _call("search_logs", service=svc, **({"query": q.group(1)} if q and "logs" not in q.group(1) else {}))
    if re.search(r"healthy|okay|degraded|trouble", low) and any(h["tool"] == "get_service_metrics" for h in hist):
        return {"type": "respond", "text": "see metrics"}
    if re.search(r"metric|error rate|cpu|latency|doing|numbers|health", low):
        return _call("get_service_metrics", service=svc) if svc else _clarify()
    if re.search(r"version|deploy|release|ship", low):
        return _call("inspect_deployments", service=svc) if svc else _clarify()
    if re.search(r"runbook|playbook|remediate", low):
        return _call("get_runbook", service=svc) if svc else _clarify()
    if re.search(r"services|inventory", low):
        return _call("list_services")
    if re.search(r"incident|remediat|grant|authorized|status", low):
        return _call("get_task_state")
    return {"type": "respond", "text": "ok"}


def _clarify():
    return {"type": "clarify", "question": "Which service?"}


def predict(examples: list[dict[str, Any]]) -> dict[str, str]:
    return {ex["id"]: target_text(route(ex)) for ex in examples}


if __name__ == "__main__":
    import sys

    from .generate import load
    from .metrics import score

    agg, _ = score(load(sys.argv[1]), predict(load(sys.argv[1])))
    print(json.dumps(agg, indent=1))
