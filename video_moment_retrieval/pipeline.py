from __future__ import annotations

import json
import math
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

from . import media
from .budget import StageBudgets
from .cache import ArtifactCache, atomic_json, digest, file_hash
from .store import Store
from .relationships import attach_reconciliation, attach_visual_links, rolling_context
from .speech import attach_speakers, WebRTCSpeechDetector
from .types import Encoder, Evidence, Extractor, OCR, Record, Transcriber, interval, component_identity, object_value, object_list

SCHEMA_VERSION = 2


@dataclass
class IndexConfig:
    window_seconds: float = 20
    overlap_seconds: float = 5
    height: int = 720
    fps: int = 8
    ocr_every_seconds: float = 10
    max_seconds: float | None = None
    enable_ocr: bool = True
    rolling_context: bool = True
    reconcile_events: bool = True

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.window_seconds, self.overlap_seconds, self.ocr_every_seconds)):
            raise ValueError("Index intervals must be finite")
        if self.window_seconds <= 0 or not 0 <= self.overlap_seconds < self.window_seconds or self.ocr_every_seconds <= 0:
            raise ValueError("Invalid index window, overlap, or OCR interval")
        if self.max_seconds is not None and (not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError("max_seconds must be finite and positive")
        if not 64 <= self.height <= 2160 or not 1 <= self.fps <= 60:
            raise ValueError("Invalid media resolution/frame rate")


def validate_segments(segments: list[dict], duration: float) -> None:
    """Validate and normalize timestamps, including optional aligned words, in place."""
    if not isinstance(segments, list):
        raise ValueError("Transcript must be a list")
    object_list(segments, "Transcript segments")
    for segment in segments:
        segment["start"], segment["end"] = float(segment["start"]), float(segment["end"])
        interval(segment["start"], segment["end"], duration)
        if not isinstance(segment["text"], str) or not segment["text"].strip():
            raise ValueError("Transcript segment requires text")
        words = segment.get("words", [])
        if not isinstance(words, list):
            raise ValueError("Transcript words must be a list")
        object_list(words, "Transcript words")
        for word in words:
            # Forced aligners may retain unaligned words without either timestamp.
            for key in ("start", "end"):
                if word.get(key) is not None:
                    word[key] = float(word[key])
                    if not math.isfinite(word[key]) or not segment["start"] <= word[key] <= segment["end"] + .001:
                        raise ValueError("Word timestamp outside its transcript segment")
            if word.get("start") is not None and word.get("end") is not None and word["end"] < word["start"]:
                raise ValueError("Word ends before it starts")


def transcript_slice(segments: list[dict], start: float, end: float, offset: float = 0) -> list[dict]:
    """Clip segments and timed words, then shift both onto the requested timeline."""
    result = []
    for segment in segments:
        if segment["start"] >= end or segment["end"] <= start:
            continue
        item = deepcopy(segment)
        item.update(start=max(start, segment["start"])+offset, end=min(end, segment["end"])+offset)
        if "words" in item:
            words = []
            for word in item["words"]:
                a, b = word.get("start"), word.get("end")
                if (a is not None and a >= end) or (b is not None and b <= start):
                    continue
                # Untimed words cannot be assigned to a cropped portion reliably.
                if a is None and b is None and (segment["start"] < start or segment["end"] > end):
                    continue
                for key in ("start", "end"):
                    if word.get(key) is not None:
                        word[key] = max(start, min(end, word[key]))+offset
                words.append(word)
            item["words"] = words
        result.append(item)
    return result


def validate_extraction(data: dict, duration: float) -> None:
    object_value(data, "Extraction")
    if not isinstance(data.get("summary"), str) or not data["summary"].strip():
        raise ValueError("Coverage window requires a nonempty summary")
    if not isinstance(data.get("observations"), list):
        raise ValueError("Missing observations list")
    object_list(data["observations"], "Observations")
    object_list(data.get("links", []), "Visual links")
    for obs in data["observations"]:
        object_value(obs.get("attributes", {}), "Observation attributes")
        obs["start"], obs["end"] = float(obs["start"]), float(obs["end"])
        interval(obs["start"], obs["end"], duration)
        if obs["kind"] not in {"action", "appearance", "context", "audio"}:
            raise ValueError("Unsupported observation type")
        if not isinstance(obs.get("text"), str) or not obs["text"].strip():
            raise ValueError("Empty observation")
        if not all(isinstance(v, str) for v in obs.get("attributes", {}).values()):
            raise ValueError("Attributes must be strings")
        if obs.get("status", "hypothesis") not in {"hypothesis", "unresolved"}:
            raise ValueError("Initial extraction must retain hypothesis or unresolved status")


def visual_records(data: dict, video_id: str, index: int, start: float, end: float,
                   source: str, transcript: list[dict], has_audio: bool) -> list[Record]:
    validate_extraction(data, end-start)
    scope = f"{video_id}:window:{index}"
    records = [Record(scope, video_id, "window", start, end, data["summary"],
        [Evidence(scope+":e", "visual", start, end, source, "coverage-window description")])]
    for i, obs in enumerate(data["observations"]):
        speaker = obs.get("speaker")
        if speaker is not None and (not isinstance(speaker, str) or speaker not in {s.get("speaker") for s in transcript}):
            raise ValueError("Observation speaker must reference the supplied transcript")
        a, b = obs["start"]+start, obs["end"]+start
        id_ = f"{scope}:event:{i}"
        local = lambda name: f"{scope}:{obs[name]}" if obs.get(name) else None
        records.append(Record(id_, video_id, obs["kind"], a, b, obs["text"],
            [Evidence(id_+":e", "audio" if obs["kind"] == "audio" else "visual", a, b, source,
                      obs.get("evidence_detail", ""))], subject=local("subject"), actor=local("actor"),
            recipient=local("recipient"), speaker=speaker, attributes=obs.get("attributes", {}), status=obs.get("status", "hypothesis")))
    attach_visual_links(records, data.get("links", []), transcript, start, end-start, source, scope, has_audio)
    return records


def validate_ocr(data: dict) -> None:
    object_value(data, "OCR output")
    object_list(data.get("detections"), "OCR detections")
    if not isinstance(data.get("detections"), list) or len(data["detections"]) > 20:
        raise ValueError("Invalid OCR detection list")
    for detection in data["detections"]:
        index = detection["frame_index"]
        if not isinstance(index, int) or not 0 <= index < len(data["frames"]):
            raise ValueError("OCR references a nonexistent frame")
        box = detection["bbox"]
        if len(box) != 4 or not all(math.isfinite(x) and 0 <= x <= 1 for x in box) or not (box[0] < box[2] and box[1] < box[3]):
            raise ValueError("Invalid OCR region")
        if detection["legibility"] not in {"readable", "partial", "unreadable"}:
            raise ValueError("Invalid OCR legibility")
        if detection.get("text") is not None and not isinstance(detection["text"], str):
            raise ValueError("OCR text must be a string or null")
        if detection["legibility"] == "readable" and not detection.get("text"):
            raise ValueError("Readable OCR must contain text")
        if not Path(detection["crop_path"]).is_file():
            raise ValueError("OCR crop artifact missing")


def deduplicate_observations(records: list[Record]) -> list[Record]:
    """Conservatively remove duplicates. Do not invent cross-window person identity.

    Identical overlapping observations can share provenance. Different wording or
    subjects remain separate for retrieval-time grouping and media verification.
    """
    result: list[Record] = []
    referenced = {id_ for r in records for link in r.links for id_ in link.get("record_ids", [])}
    for record in sorted(records, key=lambda r: (r.start, r.end, r.id)):
        match = next((r for r in result if r.kind == record.kind and r.kind != "window"
                      and r.id not in referenced and record.id not in referenced and not r.links and not record.links
                      and r.text == record.text and r.subject == record.subject
                      and r.actor == record.actor and r.recipient == record.recipient
                      and r.speaker == record.speaker and r.status == record.status
                      and r.attributes == record.attributes and r.end > record.start
                      and r.start < record.end), None)
        if match:
            match.start = min(match.start, record.start)
            match.end = max(match.end, record.end)
            known = {e.id for e in match.evidence}
            match.evidence.extend(e for e in record.evidence if e.id not in known)
        else:
            result.append(record)
    return result


class Indexer:
    def __init__(self, store: Store, cache: ArtifactCache, work: Path,
                 extractor: Extractor, transcriber: Transcriber, ocr: OCR, encoder: Encoder,
                 progress: Callable[[str], None] = lambda _: None, speech_detector=None, reconciler=None):
        self.store, self.cache, self.work = store, cache, work
        self.extractor, self.transcriber, self.ocr, self.encoder = extractor, transcriber, ocr, encoder
        self.progress = progress
        self.speech_detector = speech_detector
        self.reconciler = reconciler if reconciler is not None else (extractor if hasattr(extractor, "reconcile") else None)

    def index(self, path: str, config: IndexConfig, transcript_path: str | None = None) -> dict:
        attempt_id = uuid.uuid4().hex
        attempt_root = self.work.parent / "index-attempts"
        attempt_root.mkdir(parents=True, exist_ok=True)
        staging_path = attempt_root / (attempt_id + ".sqlite")
        report_path = attempt_root / (attempt_id + ".json")
        self.store.record_attempt(attempt_id, None, "running", str(report_path), str(staging_path))
        staging = Store(staging_path)
        worker = Indexer(staging, self.cache, self.work, self.extractor, self.transcriber, self.ocr,
                         self.encoder, self.progress, self.speech_detector, self.reconciler)
        try:
            report = worker._index(path, config, transcript_path)
            video_id = report["video_id"]
            previous = self.store.db.execute("SELECT 1 FROM records WHERE video_id=? LIMIT 1", (video_id,)).fetchone()
            publish = not report["errors"] or not previous
            if publish:
                self.store.publish_video(staging, video_id)
            status = "published" if not report["errors"] else ("published_partial" if publish else "retained_previous")
            report.update(publication=status, attempt_id=attempt_id, attempt_report=str(report_path),
                          attempt_coverage=staging.coverage())
            atomic_json(report_path, report)
            self.store.record_attempt(attempt_id, video_id, status, str(report_path), str(staging_path))
            return report
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            atomic_json(report_path, {"status": status, "error_type": type(exc).__name__, "coverage": staging.coverage()})
            self.store.record_attempt(attempt_id, None, status, str(report_path), str(staging_path))
            raise
        finally:
            if hasattr(worker, '_budgets'):
                worker._budgets.restore()
            staging.close()

    def _index(self, path: str, config: IndexConfig, transcript_path: str | None = None) -> dict:
        started = time.monotonic()
        source = str(Path(path).resolve())
        info = media.probe(source)
        duration = info["duration"]
        stop = min(duration, config.max_seconds) if config.max_seconds is not None else duration
        spans = media.windows(stop, config.window_seconds, config.overlap_seconds)
        stages = []
        if info["has_audio"] and not transcript_path:
            stages.append(("transcript", self.transcriber, 25))
        stages.append(("visual", self.extractor, 35))
        if config.reconcile_events and self.reconciler and len(spans) > 1:
            stages.append(("reconciliation", self.reconciler, 5))
        if config.enable_ocr:
            stages.append(("ocr", self.ocr, 25))
        stages.append(("embedding", self.encoder, 10))
        self._budgets = StageBudgets(stages)
        if not math.isfinite(config.ocr_every_seconds) or config.ocr_every_seconds <= 0:
            raise ValueError("OCR interval must be finite and positive")
        video_id = file_hash(source)
        root = self.work / video_id
        root.mkdir(parents=True, exist_ok=True)
        stat = Path(source).stat()
        self.store.add_video(video_id, source, duration, info["has_audio"], {**info, "synthetic": False,
            "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns})
        self.store.clear_coverage(video_id)
        records: list[Record] = []
        errors = []
        identity = {"source": video_id, "schema": SCHEMA_VERSION}

        def failed(stage, a, b, exc):
            message = f"{type(exc).__name__}: {exc}"
            errors.append({"stage": stage, "start": a, "end": b, "error": message})
            self.store.mark(video_id, stage, a, b, "failed", message)
            self.progress(f"{stage} {a:.1f}–{b:.1f}s failed: {message}")

        # Stage boundaries are persisted before work, so interruption is visible as pending.
        for stage in ("transcript", "visual", "vad", "audio", "ocr", "reconciliation", "embedding"):
            if stop < duration:
                self.store.mark(video_id, stage, stop, duration, "skipped", "outside requested indexing range")
        asr_spans = [(0, stop)] if transcript_path or getattr(self.transcriber, "full_track", False) else media.windows(stop, config.window_seconds, 0)
        scheduled = {"visual": spans, "transcript": asr_spans if info["has_audio"] or transcript_path else [(0, stop)],
                     "vad": [(0, stop)], "audio": [(0, stop)], "embedding": [(0, stop)],
                     "ocr": media.windows(stop, config.ocr_every_seconds, 0) if config.enable_ocr else [(0, stop)]}
        for stage, tasks in scheduled.items():
            for a, b in tasks:
                self.store.mark(video_id, stage, a, b, "pending")
        if config.reconcile_events and self.reconciler and len(spans) > 1:
            for left, right in zip(spans, spans[1:]):
                self.store.mark(video_id, "reconciliation", left[0], right[1], "pending")
        for a, b in spans:
            self.store.mark(video_id, "visual", a, b, "pending")
        transcript: list[dict] = []
        if transcript_path:
            supplied = json.loads(Path(transcript_path).read_text())
            validate_segments(supplied, duration)
            transcript = transcript_slice(supplied, 0, stop)
            self.store.mark(video_id, "transcript", 0, stop, "complete", "user-supplied original-timeline alignment")
        elif info["has_audio"]:
            self._budgets.enter("transcript")
            for a, b in asr_spans:
                self.store.mark(video_id, "transcript", a, b, "pending")
                self.progress(f"transcribe {a:.1f}–{b:.1f}s")
                try:
                    def transcribe():
                        if hasattr(self.transcriber, "ensure_budget"):
                            self.transcriber.ensure_budget()
                        path = media.audio(source, root / f"asr-{a}.wav", a, b)
                        return self.transcriber.transcribe(path, b-a)
                    segments = self.cache.get("transcript", {**identity, "model": component_identity(self.transcriber, "transcript"),
                        "start": a, "end": b}, transcribe, lambda s: validate_segments(s, b-a))
                    shifted = transcript_slice(segments, 0, b-a, offset=a)
                    for s in shifted:
                        # Never assume speaker labels are stable across independent calls.
                        s["speaker"] = f"asr:{a}:{s['speaker']}" if s.get("speaker") else None
                        for word in s.get("words", []):
                            if word.get("speaker"):
                                word["speaker"] = f"asr:{a}:{word['speaker']}"
                    transcript.extend(shifted)
                    self.store.mark(video_id, "transcript", a, b, "complete", getattr(self.transcriber, "alignment", "model-estimated timestamps; not forced alignment"))
                except (RuntimeError, ValueError, KeyError, TypeError, OSError) as exc:
                    failed("transcript", a, b, exc)
        else:
            self.store.mark(video_id, "transcript", 0, stop, "not_applicable", "no audio stream")
        for i, s in enumerate(transcript):
            id_ = f"{video_id}:speech:{i}"
            records.append(Record(id_, video_id, "speech", s["start"], s["end"], s["text"],
                [Evidence(id_+":e", "transcript", s["start"], s["end"], source, "ASR or supplied transcript")],
                speaker=s.get("speaker"), metadata={"alignment": "supplied" if transcript_path else getattr(self.transcriber, "alignment", "model_estimated"),
                                                   "words": s.get("words", [])}))

        self._budgets.enter("visual")
        previous_context = None
        window_groups = []
        for i, (a, b) in enumerate(spans):
            self.progress(f"visual {a:.1f}–{b:.1f}s")
            relevant = transcript_slice(transcript, a, b, offset=-a)
            try:
                def extract():
                    if hasattr(self.extractor, "ensure_budget"):
                        self.extractor.ensure_budget()
                    path = media.clip(source, root / f"visual-{a}.mp4", a, b, config.height, config.fps)
                    return self.extractor.extract(path, b-a, relevant, context=previous_context if config.rolling_context else None)
                observations = self.cache.get("visual", {**identity, "model": component_identity(self.extractor, "visual"),
                    "start": a, "end": b, "height": config.height, "fps": config.fps,
                    "transcript": relevant, "context": previous_context if config.rolling_context else None}, extract,
                    lambda d: visual_records(d, video_id, i, a, b, source, relevant, info["has_audio"]))
                window_records = visual_records(observations, video_id, i, a, b, source, relevant, info["has_audio"])
                # Publish a stage's records only after its entire response validates.
                records.extend(window_records)
                window_groups.append(window_records)
                previous_context = rolling_context(window_records, b)
                self.store.mark(video_id, "visual", a, b, "complete")
            except (RuntimeError, ValueError, KeyError, TypeError, OSError) as exc:
                failed("visual", a, b, exc)
                previous_context = None

        if info["has_audio"]:
            self.store.mark(video_id, "audio", 0, stop, "pending")
            try:
                def dsp():
                    path = media.audio(source, root / "dsp.wav", 0, stop)
                    detector = self.speech_detector or WebRTCSpeechDetector()
                    regions = self.cache.get("vad", {**identity, "end": stop, "detector": detector.identity},
                                             lambda: detector.detect(path))
                    for region in regions:
                        interval(region["start"], region["end"], stop)
                    self.store.mark(video_id, "vad", 0, stop, "complete", detector.identity)
                    speaker_regions = attach_speakers(regions, transcript)
                    return self.cache.get("dsp", {**identity, "end": stop, "regions": speaker_regions,
                        "algorithm": "independent-vad-local-dbfs-v2"}, lambda: media.energy_observations(path, speaker_regions))
                # VAD and energy artifacts are independent of ASR success; transcript only supplies optional speakers.
                changes = dsp()
                for i, change in enumerate(changes):
                    id_ = f"{video_id}:energy:{i}"
                    a, b = change["start"], change["end"]
                    records.append(Record(id_, video_id, "audio", a, b,
                        "Speech-region energy increase; possible raised voice, not confirmed shouting",
                        [Evidence(id_+":e", "audio", a, b, source, "measured PCM energy in dBFS")],
                        speaker=change["speaker"], attributes={"behavior": "raised_voice_candidate"},
                        metadata=change))
                self.store.mark(video_id, "audio", 0, stop, "complete", "energy proposals inside independently detected speech")
            except (RuntimeError, ValueError, OSError) as exc:
                self.store.mark(video_id, "vad", 0, stop, "failed", str(exc))
                failed("audio", 0, stop, exc)
        else:
            self.store.mark(video_id, "vad", 0, stop, "not_applicable", "no audio stream")
            self.store.mark(video_id, "audio", 0, stop, "not_applicable", "no audio stream")

        if config.reconcile_events and self.reconciler and len(spans) > 1:
            self._budgets.enter("reconciliation")
            groups = {group[0].start: group for group in window_groups}
            for left_span, right_span in zip(spans, spans[1:]):
                a, b = left_span[0], right_span[1]
                if left_span[0] not in groups or right_span[0] not in groups:
                    self.store.mark(video_id, "reconciliation", a, b, "skipped", "missing neighboring visual extraction")
                    continue
                pair = [*groups[left_span[0]], *groups[right_span[0]]]
                try:
                    data = self.cache.get("reconciliation", {**identity, "model": component_identity(self.reconciler, "reconciliation"),
                        "records": [r.to_dict() for r in pair]}, lambda: self.reconciler.reconcile(pair),
                        lambda links: attach_reconciliation(deepcopy(pair), links))
                    attach_reconciliation(pair, data)
                    self.store.mark(video_id, "reconciliation", a, b, "complete", "evidence-linked hypotheses; no forced identity merge")
                except (RuntimeError, ValueError, KeyError, TypeError, OSError) as exc:
                    failed("reconciliation", a, b, exc)
        else:
            self.store.mark(video_id, "reconciliation", 0, stop, "not_applicable" if len(spans) <= 1 else "skipped",
                            "one window or reconciliation disabled/unavailable")

        if config.enable_ocr:
            self._budgets.enter("ocr")
            for i, (a, b) in enumerate(media.windows(stop, config.ocr_every_seconds, 0)):
                self.store.mark(video_id, "ocr", a, b, "pending")
                self.progress(f"OCR sample {a:.1f}–{b:.1f}s")
                try:
                    def read():
                        if hasattr(self.ocr, "ensure_budget"):
                            self.ocr.ensure_budget()
                        times = sorted({max(a, min(b-0.01, a+delta)) for delta in (0.1, 0.5, 1.0)})
                        frames = [{"time": t, "path": media.frame(source, root / f"ocr-{t}.png", t)} for t in times]
                        # Detect on all nearby frames: whole-frame blur is not plate/crop blur.
                        detections = object_list(self.ocr.read_frames(frames), "OCR detections")
                        if len(detections) > 20:
                            raise ValueError("OCR detection count exceeds per-window budget")
                        for j, detection in enumerate(detections):
                            f = frames[detection["frame_index"]]
                            crop_id = digest({"frame": f["time"], "bbox": detection["bbox"]})[:24]
                            crop_path = media.crop(f["path"], root / f"crop-{crop_id}.png", detection["bbox"])
                            detection["crop_path"] = crop_path
                            detection["crop_sharpness"] = media.sharpness(crop_path)
                        # Inspect sharp crops first, retaining other independently detected
                        # regions rather than assuming they depict the same plate.
                        detections.sort(key=lambda d: d["crop_sharpness"], reverse=True)
                        for detection in detections:
                            crop_path = detection["crop_path"]
                            refined = object_value(self.ocr.read_crop(crop_path), "OCR crop reading")
                            detection["coarse_reading"] = detection.get("text")
                            detection.update(refined)
                            detection["crop_path"] = crop_path
                        return {"frames": frames, "detections": detections}
                    data = self.cache.get("ocr", {**identity, "model": component_identity(self.ocr, "ocr"), "start": a,
                        "end": b, "sampling": "three-nearby-crop-sharpness-v3"}, read, validate_ocr)
                    for j, detection in enumerate(data["detections"]):
                        f = data["frames"][detection["frame_index"]]
                        box = detection["bbox"]
                        if len(box) != 4 or not all(math.isfinite(x) and 0 <= x <= 1 for x in box) or not (box[0] < box[2] and box[1] < box[3]):
                            raise ValueError("Invalid OCR region")
                        if detection["legibility"] not in {"readable", "partial", "unreadable"}:
                            raise ValueError("Invalid OCR legibility")
                        text = detection.get("text") or "unreadable"
                        id_ = f"{video_id}:ocr:{i}:{j}"
                        t = f["time"]
                        records.append(Record(id_, video_id, "ocr", t, min(stop, t+0.01),
                            f"{detection['text_type']}: {text}",
                            [Evidence(id_+":e", "ocr", t, min(stop, t+0.01), detection["crop_path"], detection.get("detail", ""))],
                            attributes={"text_type": detection["text_type"], "legibility": detection["legibility"]},
                            status="unresolved" if detection["legibility"] == "unreadable" else "hypothesis",
                            metadata={"bbox": box, "text": detection.get("text"), "frame_time": t,
                                      "original_frame": f["path"], "coarse_reading": detection.get("coarse_reading")}))
                    self.store.mark(video_id, "ocr", a, b, "complete", "sampled frames only, not continuous text detection")
                except (RuntimeError, ValueError, KeyError, TypeError, IndexError, OSError) as exc:
                    failed("ocr", a, b, exc)
        else:
            self.store.mark(video_id, "ocr", 0, stop, "skipped", "disabled by configuration")

        self._budgets.enter("embedding")
        records = deduplicate_observations(records)
        atomic_json(root / "observations.json", [r.to_dict() for r in records])
        self.store.mark(video_id, "embedding", 0, stop, "pending")
        try:
            vectors = self.encoder.encode([r.text + " " + json.dumps(r.attributes, sort_keys=True) for r in records])
            self.store.replace_records(video_id, records, vectors, self.encoder.identity)
            self.store.mark(video_id, "embedding", 0, stop, "complete")
        except (RuntimeError, ValueError, KeyError, OSError) as exc:
            failed("embedding", 0, stop, exc)
            # Embedding failure must not discard successfully extracted lexical/structured evidence.
            self.store.replace_records(video_id, records, None, self.encoder.identity)
        report = {"video_id": video_id, "source": source, "duration_seconds": duration,
                  "requested_seconds": stop, "records": len(records), "errors": errors,
                  "config": asdict(config), "elapsed_seconds": time.monotonic()-started,
                  "cache_hits": self.cache.hits, "cache_misses": self.cache.misses,
                  "reserved_requests_by_stage": self._budgets.allocations}
        atomic_json(root / "index-report.json", report)
        return report
