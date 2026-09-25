from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

from . import media
from .assessment import assess_candidates
from .cache import ArtifactCache, atomic_json, digest
from .retrieval import retrieve
from .store import Store
from .types import Candidate, Encoder, Planner, Verifier, Verdict, Verification, component_identity, validate_media_details


@dataclass
class SearchConfig:
    candidate_budget: int = 30
    verify_budget: int = 5
    top_k: int = 5
    max_clip_seconds: float = 30
    context_seconds: float = 3
    height: int = 720
    fps: int = 8
    policy: str = "combined"
    verify: bool = True
    enumerate_all: bool = False
    offset: int = 0
    snapshot: str | None = None
    assess: bool = True

    def __post_init__(self):
        if min(self.candidate_budget, self.verify_budget, self.top_k) <= 0 or self.offset < 0:
            raise ValueError("Positive budgets and nonnegative offset required")
        if not 0 < self.max_clip_seconds <= 120 or not 0 <= self.context_seconds <= 30:
            raise ValueError("Invalid clip/context budget")
        if not 64 <= self.height <= 2160 or not 1 <= self.fps <= 60:
            raise ValueError("Invalid media resolution/frame rate")


def bounds(candidate: Candidate, config: SearchConfig, duration: float) -> tuple[float, float]:
    a = candidate.target_start if candidate.target_start is not None else candidate.record.start
    b = candidate.target_end if candidate.target_end is not None else candidate.record.end
    pad = min(config.context_seconds, config.max_clip_seconds/4)
    # Ensure short budgets do not spend the entire clip on context preceding the event.
    start = max(0, a-pad)
    end = min(duration, b+pad, start+config.max_clip_seconds)
    return start, end


def verification_tasks(candidates: list[Candidate], config: SearchConfig) -> list[Candidate]:
    if not config.enumerate_all or not config.verify:
        return candidates
    tasks = []
    pad = min(config.context_seconds, config.max_clip_seconds/4)
    size = config.max_clip_seconds-2*pad
    for candidate in candidates:
        r = candidate.record
        for start, end in media.windows(r.end-r.start, size, 0):
            c = deepcopy(candidate)
            c.target_start, c.target_end = r.start+start, r.start+end
            tasks.append(c)
    return tasks


def candidate_summary(candidate: Candidate) -> dict:
    r = candidate.record
    # Keep separate observations separate: do not combine one plate's text with
    # another plate's attributes, or borrow a detail from a different time.
    evidence = []
    for observation in candidate.records:
        details = dict(observation.attributes)
        if observation.kind == "ocr":
            details.update({k: observation.metadata[k] for k in ("text", "bbox") if k in observation.metadata})
        if details:
            evidence.append({"record_id": observation.id, "video_id": observation.video_id,
                             "start": observation.start, "end": observation.end, "details": details})
    return {"record_id": r.id, "video_id": r.video_id, "start": r.start, "end": r.end,
            "score": candidate.score, "channels": candidate.channels, "detail_evidence": evidence}


def same_match(a: dict, b: dict) -> bool:
    if a["video_id"] != b["video_id"] or a.get("details", {}) != b.get("details", {}):
        return False
    overlap = max(0, min(a["end"], b["end"])-max(a["start"], b["start"]))
    # IoU, not overlap/min-duration: distinct actions inside one broad interval
    # must not be collapsed merely because one contains the other.
    union = max(a["end"], b["end"])-min(a["start"], b["start"])
    return overlap/union >= .8


class SearchEngine:
    def __init__(self, store: Store, cache: ArtifactCache, work: Path,
                 encoder: Encoder, planner: Planner, verifier: Verifier, synthetic: bool = False,
                 assessor=None):
        self.store, self.cache, self.work = store, cache, work
        self.encoder, self.planner, self.verifier = encoder, planner, verifier
        self.synthetic = synthetic
        self.assessor = assessor if assessor is not None else (planner if hasattr(planner, "assess") else None)

    def enrich(self, candidate: Candidate, config: SearchConfig) -> None:
        r = candidate.record
        start, end = bounds(candidate, config, self.store.video(r.video_id)["duration"])
        if getattr(candidate, "_context_bounds", None) == (r.video_id, start, end):
            return
        candidate._context_bounds = (r.video_id, start, end)
        known = {rec.id for rec in candidate.records}
        candidate.members.extend(rec for rec in self.store.context(r.video_id, start, end)
                                 if rec.id not in known)

    def prepare(self, plan, candidate: Candidate, config: SearchConfig) -> dict:
        r = candidate.record
        v = self.store.video(r.video_id)
        start, end = bounds(candidate, config, v["duration"])
        self.enrich(candidate, config)
        target_start = candidate.target_start if candidate.target_start is not None else r.start
        target_end = candidate.target_end if candidate.target_end is not None else r.end
        prepared = {"start": start, "end": end, "target_start": target_start, "target_end": target_end,
                    "whole_candidate_inspected": start <= target_start and end >= target_end,
                    "source": v["path"], "synthetic": self.synthetic, "enumerate_all": config.enumerate_all}
        if self.synthetic:
            return prepared
        self.source_state(v)
        if any(m in plan.modalities for m in ("audio", "transcript")) and not v["has_audio"]:
            prepared["unavailable"] = "Query requires audio but source has no audio stream"
            return prepared
        root = self.work / digest({"video": r.video_id, "start": start, "end": end,
                                   "height": config.height, "fps": config.fps})
        root.mkdir(parents=True, exist_ok=True)
        if "visual" in plan.modalities:
            prepared["clip"] = media.clip(v["path"], root / "clip.mp4", start, end, config.height, config.fps)
        if any(m in plan.modalities for m in ("audio", "transcript")):
            prepared["audio"] = media.audio(v["path"], root / "audio.wav", start, end)
        if "ocr" in plan.modalities:
            ocr_times = {rec.metadata["frame_time"] for rec in candidate.records
                         if rec.kind == "ocr" and "frame_time" in rec.metadata
                         and start <= rec.metadata["frame_time"] < end}
            # Inspect known detections first. Also spread samples through the target,
            # rather than inspecting only its start as the initial implementation did.
            times = sorted(ocr_times)[:6] or [start+(end-start)*f for f in (.1,.5,.9)]
            prepared["frames"] = [{"time": t, "path": media.frame(v["path"], root / f"frame-{t}.png", t)} for t in times]
            prepared["sampled_visual_only"] = True
        return prepared

    def source_state(self, video: dict):
        if self.synthetic:
            return None
        stat = Path(video["path"]).stat()
        if video["metadata"].get("source_size", stat.st_size) != stat.st_size or \
                video["metadata"].get("source_mtime_ns", stat.st_mtime_ns) != stat.st_mtime_ns:
            raise ValueError("Source media changed since indexing; reindex before verification")
        return [stat.st_size, stat.st_mtime_ns]

    @staticmethod
    def media_evidence(prepared: dict, candidate: Candidate) -> list[dict]:
        evidence = [asdict(e) for rec in candidate.records for e in rec.evidence]
        for asset, modality in (("clip", "visual"), ("audio", "audio")):
            if prepared.get(asset):
                evidence.append({"id": "media:"+modality, "modality": modality,
                    "start": prepared["start"], "end": prepared["end"], "source": prepared[asset],
                    "detail": "Direct verification input"})
        for i, frame in enumerate(prepared.get("frames", [])):
            evidence.append({"id": f"media:frame:{i}", "modality": "ocr", "start": frame["time"],
                "end": min(prepared["end"], frame["time"]+.01), "source": frame["path"],
                "detail": "Original-resolution verification frame"})
        return evidence

    def validate_verification(self, value: dict, plan, candidate: Candidate):
        prepared = value["inspection"]
        batch = Verification.from_dict(value["verification"])
        evidence = self.media_evidence(prepared, candidate)
        ids = {e["id"] for e in evidence}
        for verdict in batch.verdicts:
            verdict.validate(prepared["start"], prepared["end"], ids, plan.criteria)
            if verdict.status == "supported" and not self.synthetic:
                cited = set(verdict.evidence_ids)
                for criterion in verdict.criteria:
                    cited.update(criterion.get("evidence_ids", []))
                allowed = {"visual": {"media:visual"}, "audio": {"media:audio"},
                    "ocr": {e["id"] for e in evidence if e["id"].startswith("media:frame:")},
                    "transcript": {"media:audio"} | {e["id"] for e in evidence if e["modality"] == "transcript"}}
                if any(not cited.intersection(allowed[m]) for m in plan.modalities):
                    raise ValueError("Supported result lacks direct evidence for a required modality")
                validate_media_details(verdict, plan, ids,
                    {f"media:frame:{i}": f["time"] for i, f in enumerate(prepared.get("frames", []))})
        # A cached answer cannot refer to removed/replaced derived files.
        for path in [prepared.get("clip"), prepared.get("audio"), *[f["path"] for f in prepared.get("frames", [])]]:
            if path and not Path(path).is_file():
                raise ValueError("Verification media artifact missing")
        return batch

    def search(self, query: str, config: SearchConfig) -> dict:
        began = time.monotonic()
        if not query.strip():
            raise ValueError("Search query must not be empty")
        if not self.store.db.execute("SELECT 1 FROM records LIMIT 1").fetchone():
            raise ValueError("Index contains no searchable records; index a video before searching")
        plan = self.planner.plan(query)
        if plan.requires_subject_link and plan.subject_criterion not in plan.criteria:
            plan.criteria.append(plan.subject_criterion)
        sources = []
        for row in self.store.db.execute("SELECT id FROM videos ORDER BY id"):
            video = self.store.video(row[0])
            try:
                source_state = self.source_state(video)
            except (ValueError, OSError) as exc:
                source_state = {"invalid": str(exc)}
            sources.append([video["id"], video["path"], source_state])
        snapshot = digest({"version": 4, "revision": self.store.get_meta("index_revision"),
            "plan": asdict(plan), "encoder": self.encoder.identity, "sources": sources,
            "config": {k: v for k, v in asdict(config).items() if k not in {"snapshot", "offset"}},
            "assessor": component_identity(self.assessor, "assessment"),
            "verifier": component_identity(self.verifier, "verification")})
        if config.snapshot and config.snapshot != snapshot:
            raise ValueError("Index, plan, or verification settings changed; restart enumeration")
        state_path = self.cache.root / "enumeration" / (snapshot + ".json")
        count = config.verify_budget if config.verify else config.top_k
        assessment_errors = []
        state = {"next_offset": 0, "results": [], "incomplete_tasks": 0, "operational_errors": [], "pages": {}}
        candidates = []
        if config.enumerate_all and config.offset:
            if config.snapshot is None or not state_path.exists():
                raise ValueError("Continuation requires previous snapshot and saved enumeration state")
            state = json.loads(state_path.read_text())
            if str(config.offset) in state["pages"]:
                cached = state["pages"][str(config.offset)]
                for inspected in cached["inspected"]:
                    assets = inspected.get("inspection", {})
                    for path in [assets.get("clip"), assets.get("audio"), *[f["path"] for f in assets.get("frames", [])]]:
                        if path and not Path(path).is_file():
                            raise ValueError("Enumeration media was removed; restart from offset zero")
                return {**cached, "coverage": self.store.coverage(),
                        "accumulated_results": state["results"][:cached["pagination"]["accumulated_count"]]}
            if config.offset != state["next_offset"]:
                raise ValueError("Enumeration offset would skip unexamined tasks")
            retrieval = state["retrieval"]
        else:
            candidates, retrieval = retrieve(self.store, self.encoder, plan, config.candidate_budget,
                                             config.policy, config.enumerate_all)
            if config.enumerate_all:
                # Freeze a lightweight queue once. Continuations never rerun retrieval,
                # enrich the corpus, or serialize all observations into a snapshot hash.
                tasks = verification_tasks(candidates, config)
                state.update(candidate_count=len(candidates), retrieval=retrieval, tasks=[
                    {"record_id": c.record.id, "score": c.score, "channels": c.channels,
                     "target_start": c.target_start, "target_end": c.target_end} for c in tasks])
            else:
                for candidate in candidates:
                    self.enrich(candidate, config)
                if config.assess and self.assessor:
                    candidates, assessment_errors = assess_candidates(self.assessor, self.cache, plan, candidates)
                elif config.assess:
                    assessment_errors = ["No assessment adapter configured"]
        if config.enumerate_all:
            rows = state["tasks"][config.offset:config.offset + count]
            records = {r.id: r for r in self.store.records([r["record_id"] for r in rows])}
            page = [Candidate(records[r["record_id"]], r["score"], r["channels"],
                              target_start=r["target_start"], target_end=r["target_end"]) for r in rows]
            total_tasks, candidate_count = len(state["tasks"]), state["candidate_count"]
            for candidate in page:
                self.enrich(candidate, config)
            if config.assess and self.assessor:
                _, assessment_errors = assess_candidates(self.assessor, self.cache, plan, page)
            elif config.assess:
                assessment_errors = ["No assessment adapter configured"]
        else:
            page = candidates[config.offset:config.offset + count]
            total_tasks = candidate_count = len(candidates)
        errors = [*state["operational_errors"],
            *[{"stage": "retrieval", "channel": k, "error": v} for k, v in retrieval.get("channel_errors", {}).items()],
            *[{"stage": "assessment", "error": e} for e in assessment_errors]]
        inspected, found = [], []
        incomplete = 0
        for candidate in page:
            r = candidate.record
            video = self.store.video(r.video_id)
            self.enrich(candidate, config)
            base = {"record_id": r.id, "video_id": r.video_id, "start": r.start, "end": r.end,
                    "description": r.text, "rank_score": candidate.score, "channels": candidate.channels,
                    "source": video["path"], "assessment": candidate.assessment,
                    "evidence": [asdict(e) for rec in candidate.records for e in rec.evidence]}
            if not config.verify:
                found.append({**base, "status": "candidate", "reason": "Not verified against media"})
                continue
            try:
                key = {"version": 3, "plan": asdict(plan), "source_state": self.source_state(video),
                    "records": [rec.to_dict() for rec in candidate.records],
                    "verifier": component_identity(self.verifier, "verification"),
                    "source": r.video_id, "target": [candidate.target_start, candidate.target_end],
                    "config": [config.max_clip_seconds, config.context_seconds, config.height, config.fps, config.enumerate_all]}
                def compute():
                    prepared = self.prepare(plan, candidate, config)
                    if prepared.get("unavailable"):
                        batch = Verification([Verdict("unresolved", prepared["unavailable"])], False)
                    else:
                        answer = self.verifier.verify(plan, candidate, prepared)
                        batch = Verification([answer], not config.enumerate_all) if isinstance(answer, Verdict) else answer
                    return {"verification": batch.to_dict(), "inspection": prepared}
                value = self.cache.get("verification", key, compute,
                    lambda v: self.validate_verification(v, plan, candidate))
                batch = Verification.from_dict(value["verification"])
                prepared = value["inspection"]
                if not batch.complete or not prepared["whole_candidate_inspected"] or any(v.status == "unresolved" for v in batch.verdicts):
                    incomplete += 1
                for verdict in batch.verdicts:
                    item = {**base, **asdict(verdict), "inspection": prepared,
                            "evidence": self.media_evidence(prepared, candidate), "inspection_complete": batch.complete}
                    inspected.append(item)
                    if verdict.status == "supported":
                        found.append(item)
            except (RuntimeError, ValueError, KeyError, TypeError, OSError) as exc:
                incomplete += 1
                error = {"stage": "verification", "record_id": r.id, "error": str(exc)}
                errors.append(error)
                inspected.append({**base, "status": "unresolved", "operational_error": error,
                                  "reason": f"Verification failed: {exc}"})
        unique = []
        seen = state["results"] if config.enumerate_all else []
        for item in found:
            if not any(same_match(item, old) for old in [*seen, *unique]):
                unique.append(item)
        next_offset = config.offset + len(page)
        outcome = ("degraded" if found or seen else "failed") if errors else "complete"
        decision = ("unavailable" if errors and not found else "candidates" if not config.verify
                    else "matches" if found else "rejected_candidates" if inspected and not incomplete else "abstained")
        response = {"query": query, "plan": asdict(plan), "synthetic": self.synthetic,
            "results": unique if config.enumerate_all else unique[:config.top_k], "inspected": inspected,
            "candidates": [candidate_summary(c) for c in (page if config.enumerate_all else candidates)],
            "retrieval": retrieval, "assessment_errors": assessment_errors, "coverage": self.store.coverage(),
            "operational_errors": errors, "outcome": outcome, "decision": decision,
            "pagination": {"offset": config.offset, "next_offset": next_offset if next_offset < total_tasks else None,
                "snapshot": snapshot, "candidate_count": candidate_count, "verification_task_count": total_tasks,
                "candidate_set_exhausted": next_offset >= total_tasks,
                "incomplete_tasks": state["incomplete_tasks"] + incomplete,
                "all_tasks_verified": config.verify and not errors and next_offset >= total_tasks and state["incomplete_tasks"] + incomplete == 0},
            "completeness": "Enumeration covers indexed evidence in bounded slices; source sampling, failed ingestion, and missed detections still limit real-world recall.",
            "elapsed_seconds": time.monotonic() - began}
        if config.enumerate_all:
            state["results"].extend(unique)
            state["next_offset"] = next_offset
            state["incomplete_tasks"] += incomplete
            state["operational_errors"] = errors
            response["accumulated_results"] = list(state["results"])
            response["pagination"]["accumulated_count"] = len(state["results"])
            state["pages"][str(config.offset)] = {k: v for k, v in response.items() if k not in {"coverage", "accumulated_results"}}
            atomic_json(state_path, state)
        return response
