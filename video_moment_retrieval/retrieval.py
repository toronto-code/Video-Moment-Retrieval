from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

from .store import Store
from .types import Candidate, Encoder, QueryPlan, Record


def overlap_ratio(a: Record, b: Record) -> float:
    if a.video_id != b.video_id:
        return 0
    intersection = max(0, min(a.end, b.end)-max(a.start, b.start))
    return intersection / max(a.end-a.start, b.end-b.start)


def deduplicate(candidates: list[Candidate], threshold: float = 0.7) -> list[Candidate]:
    """Suppress alternate windows around the same evidence without joining disjoint events.

    Broad windows cannot transitively glue separate short events together.
    Prefer precise observations to coverage windows inside each duplicate group.
    """
    result: list[Candidate] = []
    for candidate in candidates:
        match = next((c for c in result if overlap_ratio(c.record, candidate.record) >= threshold
                      and (c.record.kind == "window" or candidate.record.kind == "window"
                           or (c.record.kind == candidate.record.kind and c.record.subject == candidate.record.subject
                               and c.record.actor == candidate.record.actor and c.record.text == candidate.record.text))), None)
        if match:
            match.channels = sorted(set(match.channels + candidate.channels))
            if match.record.kind == "window" and candidate.record.kind != "window":
                match.members.append(match.record)
                match.record = candidate.record
            else:
                match.members.append(candidate.record)
        else:
            result.append(candidate)
    return result


def retrieve(store: Store, encoder: Encoder, plan: QueryPlan, budget: int = 30,
             policy: str = "combined", enumerate_all: bool = False) -> tuple[list[Candidate], dict]:
    if budget <= 0:
        raise ValueError("Candidate budget must be positive")
    if policy not in {"lexical", "dense", "structured", "combined"}:
        raise ValueError("Unknown retrieval policy")
    total = store.db.execute("SELECT count(*) FROM records").fetchone()[0]
    limit = max(total, 1) if enumerate_all else budget
    def channel(name):
        reader = Store(store.path, read_only=True)
        try:
            if name == "lexical":
                return reader.lexical(plan.keywords, limit)
            if name == "structured":
                return reader.structured(plan.constraints, limit)
            return reader.semantic(encoder.encode([plan.query])[0], encoder.identity, limit)
        finally:
            reader.close()
    names = ["lexical", "structured", "dense"] if policy == "combined" else [policy]
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        futures = {name: pool.submit(channel, name) for name in names}
        channels, errors = {}, {}
        for name, future in futures.items():
            try:
                channels[name] = future.result()
            except (RuntimeError, ValueError, OSError, sqlite3.Error) as exc:
                errors[name] = str(exc)
                channels[name] = []
    if len(errors) == len(names):
        raise RuntimeError(f"All retrieval channels failed: {errors}")
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    for name, results in channels.items():
        for rank, (id_, _) in enumerate(results, 1):
            scores[id_] = scores.get(id_, 0) + 1/(60+rank)
            sources.setdefault(id_, []).append(name)
    ordered = sorted(scores, key=lambda id_: (-scores[id_], id_))
    records = store.records(ordered)
    if enumerate_all:
        # Enumerate *all indexed evidence*, even records with zero lexical/structured hits.
        records += [r for r in store.records() if r.id not in scores]
    candidates = [Candidate(r, scores.get(r.id, 0), sources.get(r.id, ["coverage"])) for r in records]
    # Enumeration visits every record: a broad coverage window may contain an
    # unextracted event besides a shorter explicit observation.
    if not enumerate_all:
        candidates = deduplicate(candidates)
    before_budget = len(candidates)
    if not enumerate_all:
        candidates = candidates[:budget]  # same final candidate budget for every ablation
    return candidates, {"policy": policy, "record_count": total,
                        "channel_hits": {k: len(v) for k, v in channels.items()},
                        "merged_candidates": before_budget, "candidate_count": len(candidates),
                        "budget": budget, "enumerate_all": enumerate_all, "channel_errors": errors}
