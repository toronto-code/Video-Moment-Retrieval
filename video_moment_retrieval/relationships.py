"""Validate model-proposed relationships without turning co-occurrence into identity."""
from __future__ import annotations

from .cache import digest
from .types import Evidence, Record, interval


def attach_visual_links(records: list[Record], proposals: list[dict], transcript: list[dict],
                        offset: float, duration: float, source: str, scope: str, has_audio: bool) -> None:
    known_people = {p for r in records for p in (r.subject, r.actor, r.recipient) if p}
    known_speakers = {s.get("speaker") for s in transcript if s.get("speaker")}
    for raw in proposals:
        a, b = float(raw["start"]), float(raw["end"])
        interval(a, b, duration)
        person = f"{scope}:{raw['person']}" if raw.get("person") else None
        speaker = raw.get("speaker")
        if person and person not in known_people:
            raise ValueError("Relationship refers to an unobserved visual subject")
        if speaker and speaker not in known_speakers:
            raise ValueError("Relationship refers to an unobserved transcript speaker")
        status = raw.get("status", "unresolved")
        if status not in {"supported", "hypothesis", "unresolved"}:
            raise ValueError("Invalid relationship status")
        method = raw.get("method", "temporal_overlap")
        # A model may claim support only for an affirmative audiovisual mechanism.
        # This remains model evidence, not a calibrated or human-confirmed identity.
        if status == "supported" and (not person or not speaker or not has_audio or
                method != "visible_synchronized_speech" or not raw.get("detail")):
            status = "unresolved"
        target = next((r for r in records if person in (r.subject, r.actor, r.recipient)), records[0])
        link_id = scope + ":link:" + digest(raw)[:16]
        evidence = [Evidence(link_id+":visual", "visual", offset+a, offset+b, source, raw.get("detail", ""))]
        if has_audio:
            evidence.append(Evidence(link_id+":audio", "audio", offset+a, offset+b, source, raw.get("detail", "")))
        target.evidence.extend(evidence)
        target.links.append({"id": link_id, "kind": "speaker_visual", "person": person, "speaker": speaker,
            "status": status, "method": method, "start": offset+a, "end": offset+b,
            "evidence_ids": [e.id for e in evidence], "reason": raw.get("detail", "")})


def attach_reconciliation(records: list[Record], links: list[dict]) -> None:
    by_id = {r.id: r for r in records}
    for raw in links:
        if raw.get("kind") not in {"event_continuation", "same_person"}:
            raise ValueError("Unknown reconciliation relationship")
        ids = raw.get("record_ids", [])
        if len(ids) != 2 or len(set(ids)) != 2 or not set(ids).issubset(by_id):
            raise ValueError("Reconciliation must reference two distinct existing observations")
        a, b = [by_id[id_] for id_ in ids]
        if a.video_id != b.video_id:
            raise ValueError("Reconciliation cannot cross videos")
        evidence = {e.id: e for r in (a, b) for e in r.evidence}
        cited = raw.get("evidence_ids", [])
        if not cited or not set(cited).issubset(evidence):
            raise ValueError("Reconciliation lacks valid supporting evidence")
        if not raw.get("reason", "").strip():
            raise ValueError("Reconciliation requires a reason")
        target = b
        known = {e.id for e in target.evidence}
        target.evidence.extend(evidence[id_] for id_ in cited if id_ not in known)
        # This pass reads descriptions, not raw media: it may propose hypotheses,
        # but cannot promote identities or actions to supported.
        target.links.append({**raw, "id": "reconcile:"+digest(raw)[:24], "status": "hypothesis",
                             "start": min(a.start, b.start), "end": max(a.end, b.end)})


def rolling_context(records: list[Record], end: float) -> dict:
    recent = [r for r in records if r.end >= end-5 and r.kind != "window"][-12:]
    return {"ending_at": end, "observations": [r.to_dict() for r in recent],
            "instruction": "Previous-window hypotheses only. Ground every new claim in current media."}
