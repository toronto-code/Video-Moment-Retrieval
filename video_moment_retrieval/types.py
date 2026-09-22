"""Provider-independent contracts. Times are seconds on the original media timeline."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


def component_identity(provider, stage: str) -> str:
    return getattr(provider, "identities", {}).get(stage, getattr(provider, "identity", type(provider).__name__))


def interval(start: float, end: float, duration: float | None = None) -> None:
    if not all(math.isfinite(x) for x in (start, end)) or start < 0 or end <= start:
        raise ValueError(f"Invalid interval [{start}, {end}]")
    if duration is not None and end > duration + 0.001:
        raise ValueError(f"Interval end {end} exceeds media duration {duration}")


@dataclass
class Evidence:
    id: str
    modality: str
    start: float
    end: float
    source: str
    detail: str = ""

    def __post_init__(self) -> None:
        interval(self.start, self.end)
        if self.modality not in {"visual", "audio", "transcript", "ocr", "synthetic"}:
            raise ValueError("Unknown evidence modality")


@dataclass
class Record:
    id: str
    video_id: str
    kind: str
    start: float
    end: float
    text: str
    evidence: list[Evidence]
    subject: str | None = None
    actor: str | None = None
    recipient: str | None = None
    speaker: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)
    status: str = "hypothesis"
    links: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        interval(self.start, self.end)
        if self.kind not in {"window", "action", "appearance", "speech", "audio", "ocr", "context"}:
            raise ValueError("Unknown record kind")
        if not self.id or not self.video_id or not self.text.strip() or not self.evidence:
            raise ValueError("Records require IDs, text, and evidence")
        if self.status not in {"supported", "hypothesis", "unresolved"}:
            raise ValueError("Unknown observation status")
        for link in self.links:
            if link.get("status") not in {"supported", "hypothesis", "unresolved"}:
                raise ValueError("Links need explicit evidence status")
            interval(float(link["start"]), float(link["end"]))
            if link["status"] == "supported" and not link.get("evidence_ids"):
                raise ValueError("Supported links require evidence")
            if not set(link.get("evidence_ids", [])).issubset({e.id for e in self.evidence}):
                raise ValueError("Unknown link evidence")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Record:
        data = dict(data)
        data["evidence"] = [Evidence(**e) for e in data["evidence"]]
        return cls(**data)


@dataclass
class QueryPlan:
    query: str
    keywords: list[str]
    constraints: dict[str, str] = field(default_factory=dict)
    modalities: list[str] = field(default_factory=lambda: ["visual"])
    criteria: list[str] = field(default_factory=list)
    requires_subject_link: bool = False

    def __post_init__(self) -> None:
        if not self.query.strip() or not self.modalities:
            raise ValueError("Query and modalities are required")
        if set(self.modalities) - {"visual", "audio", "transcript", "ocr"}:
            raise ValueError("Unknown query modality")
        if not all(isinstance(x, str) for x in self.keywords + self.criteria):
            raise ValueError("Keywords and criteria must be strings")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in self.constraints.items()):
            raise ValueError("Constraints must map strings to strings")
        if not isinstance(self.requires_subject_link, bool):
            raise ValueError("requires_subject_link must be a boolean")
        self.criteria = list(dict.fromkeys(self.criteria))


@dataclass
class Candidate:
    record: Record
    score: float
    channels: list[str]
    members: list[Record] = field(default_factory=list)
    target_start: float | None = None
    target_end: float | None = None
    assessment: dict[str, Any] = field(default_factory=dict)

    @property
    def records(self) -> list[Record]:
        return [self.record, *self.members]


@dataclass
class Verdict:
    status: str  # supported, rejected, unresolved; never a calibrated probability
    reason: str
    start: float | None = None
    end: float | None = None
    evidence_ids: list[str] = field(default_factory=list)
    criteria: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def validate(self, start: float, end: float, evidence_ids: set[str], required: list[str]) -> None:
        if self.status not in {"supported", "rejected", "unresolved"}:
            raise ValueError("Invalid verification status")
        if not self.reason.strip() or not set(self.evidence_ids).issubset(evidence_ids):
            raise ValueError("Verification needs a reason and valid evidence references")
        names = [c.get("criterion") for c in self.criteria]
        if len(names) != len(set(names)):
            raise ValueError("Each criterion must have one unambiguous assessment")
        for c in self.criteria:
            if c.get("status") not in {"supported", "rejected", "unresolved"} or not set(c.get("evidence_ids", [])).issubset(evidence_ids):
                raise ValueError("Invalid criterion evidence/status")
        if self.status == "supported":
            if self.start is None or self.end is None or not self.evidence_ids:
                raise ValueError("Supported results need localized evidence")
            interval(self.start, self.end, end)
            if self.start < start:
                raise ValueError("Verification begins outside inspected media")
            supported = {c.get("criterion") for c in self.criteria
                         if c.get("status") == "supported" and c.get("evidence_ids")
                         and set(c["evidence_ids"]).issubset(evidence_ids)}
            if not set(required).issubset(supported):
                raise ValueError("Every query criterion needs supporting evidence")


@dataclass
class Verification:
    verdicts: list[Verdict]
    complete: bool = False

    def to_dict(self) -> dict:
        return {"verdicts": [asdict(v) for v in self.verdicts], "complete": self.complete}

    @classmethod
    def from_dict(cls, data: dict) -> Verification:
        if not isinstance(data.get("complete"), bool) or not isinstance(data.get("verdicts"), list):
            raise ValueError("Verification requires verdicts and explicit completeness")
        if not 1 <= len(data["verdicts"]) <= 20:
            raise ValueError("Verification must contain 1–20 verdicts")
        return cls([Verdict(**v) for v in data["verdicts"]], data["complete"])


class Encoder(Protocol):
    identity: str
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class Planner(Protocol):
    def plan(self, query: str) -> QueryPlan: ...


class Extractor(Protocol):
    identity: str
    def extract(self, clip: str, duration: float, transcript: list[dict], context: dict | None = None) -> dict: ...


class Transcriber(Protocol):
    identity: str
    def transcribe(self, audio: str, duration: float) -> list[dict]: ...


class OCR(Protocol):
    identity: str
    def read_frames(self, frames: list[dict]) -> list[dict]: ...
    def read_crop(self, path: str) -> dict: ...


class Verifier(Protocol):
    def verify(self, plan: QueryPlan, candidate: Candidate, media: dict) -> Verification | Verdict: ...


class Assessor(Protocol):
    identity: str
    def assess(self, plan: QueryPlan, candidates: list[Candidate]) -> list[dict]: ...


class Reconciler(Protocol):
    identity: str
    def reconcile(self, records: list[Record]) -> list[dict]: ...


class SpeechDetector(Protocol):
    identity: str
    def detect(self, audio: str) -> list[dict]: ...
