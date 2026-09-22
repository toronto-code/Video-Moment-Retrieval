"""Evaluate retrieval and decisions separately against manually reviewed intervals."""
from __future__ import annotations

import statistics
import math
from dataclasses import replace

from .search import SearchConfig, SearchEngine
from .types import interval


def temporal_iou(a: dict, b: dict) -> float:
    if a["video_id"] != b["video_id"]:
        return 0.0
    intersection = max(0, min(a["end"], b["end"])-max(a["start"], b["start"]))
    union = max(a["end"], b["end"])-min(a["start"], b["start"])
    return intersection/union if union else 0.0


def matching(predictions: list[dict], truth: list[dict], threshold: float,
             onset_tolerance: float | None = None) -> list[tuple[int, int]]:
    """Maximum-cardinality one-to-one matching prevents duplicate true positives."""
    edges = {}
    for i, prediction in enumerate(predictions):
        candidates = []
        for j, actual in enumerate(truth):
            value = temporal_iou(prediction, actual)
            valid = (prediction["video_id"] == actual["video_id"] and
                     abs(prediction["start"]-actual["start"]) <= onset_tolerance) if onset_tolerance is not None else value >= threshold
            if actual.get("details"):
                valid = valid and all(prediction.get("details", {}).get(k) == v for k,v in actual["details"].items())
            if valid:
                candidates.append((j, value))
        edges[i] = [j for j, _ in sorted(candidates, key=lambda x: -x[1])]
    owner = {}
    def assign(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or assign(owner[j], seen):
                owner[j] = i
                return True
        return False
    for i in range(len(predictions)):
        assign(i, set())
    return [(i, j) for j, i in owner.items()]


def metrics(candidates: list[dict], results: list[dict], truth: list[dict],
            threshold: float = 0.5, onset_tolerance: float | None = None) -> dict:
    # Retrieval asks whether sufficient evidence reached the shortlist. A coverage
    # window may contain multiple distinct true events, unlike final predictions.
    def covers(candidate, actual):
        return candidate["video_id"] == actual["video_id"] and \
            max(0, min(candidate["end"], actual["end"])-max(candidate["start"], actual["start"])) / \
            (actual["end"]-actual["start"]) >= .8

    def has_details(candidate, actual):
        required = actual.get("details", {})
        if not required:
            return True
        return any(e["video_id"] == actual["video_id"] and e["start"] < actual["end"] and e["end"] > actual["start"]
                   and all(e.get("details", {}).get(k) == v for k, v in required.items())
                   for e in [candidate, *candidate.get("detail_evidence", [])])

    temporal = [any(covers(c, t) for c in candidates) for t in truth]
    covered = [any(covers(c, t) and has_details(c, t) for c in candidates) for t in truth]
    final_matches = matching(results, truth, threshold, onset_tolerance)
    onset_errors = [abs(results[i]["start"]-truth[j]["start"]) for i, j in final_matches]
    end_errors = [abs(results[i]["end"]-truth[j]["end"]) for i, j in final_matches]
    return {"candidate_recall": sum(covered)/len(truth) if truth else None,
            "candidate_temporal_recall": sum(temporal)/len(truth) if truth else None,
            "precision": len(final_matches)/len(results) if results else None,
            "recall": len(final_matches)/len(truth) if truth else None,
            "true_positives": len(final_matches), "false_positives": len(results)-len(final_matches),
            "false_negatives": len(truth)-len(final_matches), "abstained": not results,
            "no_match_correct": not results if not truth else None,
            "mean_onset_error_seconds": statistics.mean(onset_errors) if onset_errors else None,
            "mean_end_error_seconds": statistics.mean(end_errors) if end_errors else None}


def evaluate(engine: SearchEngine, labels: dict, config: SearchConfig,
             policies: list[str] | None = None, split: str = "test") -> dict:
    if bool(labels.get("synthetic")) != engine.synthetic:
        raise ValueError("Do not mix synthetic labels and real inference")
    if not engine.synthetic:
        actual = {r[0] for r in engine.store.db.execute("SELECT id FROM videos")}
        if set(labels.get("video_ids", [])) != actual:
            raise ValueError("Real evaluation labels must list exactly the indexed video_ids; use a held-out index")
    policies = policies or ["lexical", "dense", "structured", "combined", "combined+verification",
                            "combined+assessment", "combined+assessment+verification"]
    queries = [q for q in labels["queries"] if q.get("split", "test") == split]
    if not queries:
        raise ValueError("No labeled queries for selected split")
    if len({q["id"] for q in queries}) != len(queries):
        raise ValueError("Evaluation query IDs must be unique")
    for q in queries:
        for match in q["matches"]:
            video = engine.store.video(match["video_id"])
            interval(match["start"], match["end"], video["duration"])
        if not 0 < q.get("iou_threshold", 0.5) <= 1:
            raise ValueError("IoU threshold must be in (0,1]")
        tolerance = q.get("onset_tolerance_seconds")
        if tolerance is not None and (not math.isfinite(tolerance) or tolerance < 0):
            raise ValueError("Onset tolerance must be finite and nonnegative")
    reports = []
    for policy in policies:
        parts = policy.split("+")
        if parts[0] not in {"lexical", "dense", "structured", "combined"} or set(parts[1:]) - {"assessment", "verification"}:
            raise ValueError("Unknown evaluation policy")
        verify, assess = "verification" in parts, "assessment" in parts
        retrieval_policy = parts[0]
        results = []
        for q in queries:
            client = getattr(engine.verifier, "client", None)
            before = client.stats() if client else {"http_attempts": 0, "reported_cost_usd": 0}
            usage_start = len(client.usage) if client else 0
            cache_before = (engine.cache.hits, engine.cache.misses)
            response = engine.search(q["query"], replace(config, policy=retrieval_policy,
                verify=verify, verify_budget=config.candidate_budget, top_k=config.candidate_budget,
                enumerate_all=False, offset=0, snapshot=None, assess=assess))
            after = client.stats() if client else before
            scores = metrics(response["candidates"], response["results"], q["matches"],
                             q.get("iou_threshold", 0.5), q.get("onset_tolerance_seconds"))
            results.append({"id": q["id"], "query": q["query"], **scores,
                            "elapsed_seconds": response["elapsed_seconds"],
                            "http_attempts": after["http_attempts"]-before["http_attempts"],
                            "reported_cost_usd": after["reported_cost_usd"]-before["reported_cost_usd"],
                            "cost_complete": (after["http_attempts"]-before["http_attempts"] == len(client.usage[usage_start:])
                                and all(isinstance(r["usage"].get("cost"), (int,float)) for r in client.usage[usage_start:])) if client else True,
                            "cache_hits": engine.cache.hits-cache_before[0],
                            "cache_misses": engine.cache.misses-cache_before[1],
                            "unresolved": sum(i["status"] == "unresolved" for i in response["inspected"]),
                            "result_intervals": [{k:r[k] for k in ("video_id", "start", "end", "status")}
                                                 for r in response["results"]]})
        tp = sum(r["true_positives"] for r in results)
        fp = sum(r["false_positives"] for r in results)
        fn = sum(r["false_negatives"] for r in results)
        recalls = [r["candidate_recall"] for r in results if r["candidate_recall"] is not None]
        temporal_recalls = [r["candidate_temporal_recall"] for r in results if r["candidate_temporal_recall"] is not None]
        latencies = sorted(r["elapsed_seconds"] for r in results)
        reports.append({"policy": policy, "candidate_budget": config.candidate_budget,
                        "micro_precision": tp/(tp+fp) if tp+fp else None,
                        "micro_recall": tp/(tp+fn) if tp+fn else None,
                        "mean_candidate_recall": statistics.mean(recalls) if recalls else None,
                        "mean_candidate_temporal_recall": statistics.mean(temporal_recalls) if temporal_recalls else None,
                        "abstention_rate": sum(r["abstained"] for r in results)/len(results),
                        "p50_latency_seconds": statistics.median(latencies),
                        "p95_latency_seconds": latencies[max(0, math.ceil(.95*len(latencies))-1)],
                        "queries": results})
    return {"synthetic": engine.synthetic, "split": split, "query_count": len(queries),
            "note": "Synthetic contract tests are not evidence of model accuracy." if engine.synthetic else
                    "Scores apply only to this manually reviewed subset. Report sample size and limitations.",
            "policies": reports, "coverage": engine.store.coverage()}
