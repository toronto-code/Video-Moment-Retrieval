from __future__ import annotations

from dataclasses import asdict

from .types import Candidate, QueryPlan, component_identity, object_list, string_list


def assess_candidates(assessor, cache, plan: QueryPlan, candidates: list[Candidate], batch_size: int = 10):
    errors = []
    original = candidates
    # Enumeration can contain several bounded slices of one original record.
    # Assess that record once, then attach the result to every slice.
    candidates = list({c.record.id: c for c in candidates}.values())
    for offset in range(0, len(candidates), batch_size):
        batch = candidates[offset:offset+batch_size]
        by_id = {c.record.id: c for c in batch}
        def validate(rows):
            object_list(rows, "Assessments")
            for row in rows:
                string_list(row.get("evidence_ids", []), "Assessment evidence IDs")
                string_list(row.get("supported_link_ids", []), "Assessment link IDs")
            if len(rows) != len(batch) or {r["record_id"] for r in rows} != set(by_id):
                raise ValueError("Assessment must cover each candidate exactly once")
            for row in rows:
                c = by_id[row["record_id"]]
                if row["status"] not in {"likely", "contradicted", "unresolved"} or not row.get("reason"):
                    raise ValueError("Invalid assessment")
                evidence = {e.id for r in c.records for e in r.evidence}
                if not set(row.get("evidence_ids", [])).issubset(evidence):
                    raise ValueError("Assessment invented evidence")
                if row["status"] in {"likely", "contradicted"} and not row.get("evidence_ids"):
                    raise ValueError("Decisive assessment requires evidence")
                supported = {link["id"] for r in c.records for link in r.links
                             if link.get("kind") == "speaker_visual" and link["status"] == "supported"}
                if not set(row.get("supported_link_ids", [])).issubset(supported):
                    raise ValueError("Assessment invented a supported subject link")
        try:
            rows = cache.get("assessment", {"version": 1, "model": component_identity(assessor, "assessment"),
                "plan": asdict(plan), "candidates": [[r.to_dict() for r in c.records] for c in batch]},
                lambda: assessor.assess(plan, batch), validate)
            for row in rows:
                row = dict(row)
                if plan.requires_subject_link and row["status"] == "likely" and (plan.subject_link_kind == "visual" or not row.get("supported_link_ids")):
                    row.update(status="unresolved", reason="Same-person binding requires direct media confirmation")
                by_id[row["record_id"]].assessment = row
        except (RuntimeError, ValueError, KeyError, TypeError, OSError) as exc:
            errors.append(str(exc))
            for c in batch:
                c.assessment = {"status": "unresolved", "reason": f"Assessment failed: {exc}"}
    priority = {"likely": 0, "unresolved": 1, "contradicted": 2}
    # Do not delete unresolved or contradicted candidates. Original-media inspection
    # can correct extraction/assessment errors; ranked search merely changes order.
    assessments = {c.record.id: c.assessment for c in candidates}
    for candidate in original:
        candidate.assessment = dict(assessments[candidate.record.id])
    return sorted(original, key=lambda c: (priority[c.assessment["status"]], -c.score, c.record.id)), errors
