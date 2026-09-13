"""Shared, bounded validation for host-owned session plan metadata."""

from __future__ import annotations

import json
from typing import Any


MAX_PLAN_STEPS = 32
MAX_PLAN_CHARS = 32_000
PLAN_FIELD_LIMITS = {"step": 500, "status": 20, "acceptance": 2000, "evidence": 4000}
PLAN_STATUSES = {"pending", "in_progress", "completed"}


def normalize_plan(plan: Any, explanation: Any = "") -> dict[str, Any]:
    if not isinstance(plan, list) or len(plan) > MAX_PLAN_STEPS:
        raise ValueError(f"plan must be an array with at most {MAX_PLAN_STEPS} items")
    if not isinstance(explanation, str) or len(explanation) > 2000:
        raise ValueError("explanation must be a string of at most 2000 characters")
    clean = []
    for index, item in enumerate(plan, 1):
        if not isinstance(item, dict) or set(item) - PLAN_FIELD_LIMITS.keys():
            raise ValueError(f"plan item {index} contains invalid fields")
        values = {}
        for field, limit in PLAN_FIELD_LIMITS.items():
            value = item.get(field, "")
            if not isinstance(value, str) or len(value) > limit:
                raise ValueError(f"plan item {index} {field} must be a string of at most {limit} characters")
            value = value.strip()
            if value or field in ("step", "status"):
                values[field] = value
        if not values["step"] or values["status"] not in PLAN_STATUSES:
            raise ValueError(f"plan item {index} needs a nonempty step and valid status")
        if values["status"] == "completed" and values.get("acceptance") and not values.get("evidence"):
            raise ValueError(f"plan item {index} needs evidence for its acceptance criteria before completion")
        clean.append(values)
    active = sum(item["status"] == "in_progress" for item in clean)
    incomplete = any(item["status"] != "completed" for item in clean)
    if active > 1 or (incomplete and active != 1):
        raise ValueError("exactly one plan item must be in_progress until all are completed")
    payload = {"plan": clean, "explanation": explanation.strip() or None}
    if len(json.dumps(payload, ensure_ascii=False)) > MAX_PLAN_CHARS:
        raise ValueError(f"plan must fit within {MAX_PLAN_CHARS} serialized characters")
    return payload
