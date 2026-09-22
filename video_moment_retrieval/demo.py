"""Synthetic contract fixtures, NOT a video-understanding benchmark.

The toy encoder and scripted verifier let contributors test orchestration without
credits. Verification reads authored assertions, not media. Never use on real data.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .cache import atomic_json
from .store import Store, normalize_vector
from .types import Candidate, Evidence, QueryPlan, Record, Verdict, Verification


class DemoEncoder:
    identity = "synthetic:toy-hash-synonyms-v1"
    def encode(self, texts: list[str]) -> list[list[float]]:
        synonyms = {"handcuffed": "cuffs", "handcuffs": "cuffs", "wrists": "cuffs",
                    "shouting": "voice", "shouts": "voice", "raised": "voice",
                    "recites": "miranda", "rights": "miranda", "vehicle": "car"}
        vectors = []
        for text in texts:
            vector = [0.0]*64
            for token in re.findall(r"\w+", text.lower()) or ["empty"]:
                token = synonyms.get(token, token)
                index = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 64
                vector[index] += 1
            vectors.append(normalize_vector(vector))
        return vectors


class DemoBackend:
    identity = "synthetic:scripted-verifier-v1"
    def plan(self, query: str) -> QueryPlan:
        q = query.lower()
        criteria, modalities, constraints = [], [], {}
        if any(w in q for w in ("handcuff", "wrists")):
            criteria.append("application_of_cuffs")
            modalities.append("visual")
            constraints["action"] = "applies handcuffs"
        if "miranda" in q or "rights" in q:
            criteria.append("recites_rights")
            modalities.append("transcript")
        if "red" in q:
            criteria.append("red_shirt")
            modalities.append("visual")
            constraints["clothing"] = "red shirt"
        if "voice" in q or "shout" in q:
            criteria.append("raised_voice")
            modalities.append("audio")
            constraints["behavior"] = "raised_voice_candidate"
        linked = "red_shirt" in criteria and "raised_voice" in criteria
        if linked:
            criteria.append("same_person")
        if "pulled over" in q or "traffic stop" in q:
            criteria.append("traffic_stop")
            modalities.append("visual")
        if "night" in q:
            criteria.append("night")
            constraints["lighting"] = "night"
            modalities.append("visual")
        if "plate" in q:
            criteria.append("plate_visible")
            modalities.append("ocr")
            if "says" in q or "readable" in q or "read " in q:
                criteria.append("plate_readable")
        if not criteria:
            criteria = [query]
        return QueryPlan(query, re.findall(r"\w+", query), constraints,
                         sorted(set(modalities)) or ["visual"], criteria, linked)

    def assess(self, plan, candidates):
        return [{"record_id": c.record.id, "status": "unresolved",
                 "reason": "Synthetic baseline has no inferred relationship proof", "evidence_ids": [],
                 "supported_link_ids": []} for c in candidates]

    def verify(self, plan: QueryPlan, candidate: Candidate, media: dict) -> Verification:
        # Only assertions on the actual candidate may satisfy criteria. Nearby other
        # people/events cannot be combined into an imaginary compound match.
        r = candidate.record
        assertions = set(r.metadata.get("synthetic_assertions", []))
        if "same_person" in assertions:
            assertions.add("Same person satisfies the appearance and speech conditions")
        ids = [e.id for e in r.evidence]
        criteria = [{"criterion": c, "status": "supported" if c in assertions else "rejected",
                     "evidence_ids": ids if c in assertions else []} for c in plan.criteria]
        if set(plan.criteria).issubset(assertions):
            return Verification([Verdict("supported", "Synthetic fixture assertions satisfy all criteria; no media inference",
                           r.start, r.end, ids, criteria)], True)
        return Verification([Verdict("rejected", "Synthetic fixture lacks: " + ", ".join(set(plan.criteria)-assertions), criteria=criteria)], True)


def build_demo(store: Store, output: Path) -> dict:
    if store.get_meta("encoder") not in {None, DemoEncoder.identity}:
        raise ValueError("Use a separate data directory for the synthetic demo")
    examples = [
        ("cuffs_apply", "action", "Officer secures the person's wrists behind their back, applying handcuffs", ["application_of_cuffs"], {"action": "applies handcuffs"}),
        ("cuffs_state", "action", "Person already handcuffed sits quietly; officer does not apply cuffs", [], {"action": "wears handcuffs"}),
        ("rights_recited", "speech", "Officer recites Miranda rights: You have the right to remain silent", ["recites_rights"], {}),
        ("rights_discussed", "speech", "Officer asks whether Miranda rights were read earlier; no recitation", [], {}),
        ("red_shouts", "audio", "The person in a red shirt raises their voice while speaking", ["red_shirt", "raised_voice", "same_person"], {"clothing": "red shirt", "behavior": "raised_voice_candidate"}),
        ("red_other_shouts", "audio", "Person A wears a red shirt silently while person B raises their voice", ["red_shirt", "raised_voice"], {"clothing": "red shirt", "behavior": "raised_voice_candidate"}),
        ("siren", "audio", "Loud siren energy spike near a quiet person in a red shirt", ["red_shirt"], {"clothing": "red shirt", "behavior": "raised_voice_candidate"}),
        ("night_stop", "action", "Officer signals a moving vehicle to pull over at night; driver stops", ["traffic_stop", "night"], {"lighting": "night", "action": "traffic stop"}),
        ("night_parked", "context", "Parked car at night; no vehicle is pulled over", ["night"], {"lighting": "night"}),
        ("plate_clear", "ocr", "License plate visible and readable: DEMO123", ["plate_visible", "plate_readable"], {"text_type": "license_plate", "legibility": "readable"}),
        ("plate_blurred", "ocr", "License plate visible but unreadable due to motion blur", ["plate_visible"], {"text_type": "license_plate", "legibility": "unreadable"}),
    ]
    records = []
    video_id = "synthetic-fixture"
    store.add_video(video_id, "synthetic://authored-scenarios-no-video", 330, True, {"synthetic": True})
    for i, (id_, kind, text, assertions, attributes) in enumerate(examples):
        a, b = i*30+5, i*30+12
        e = Evidence(id_+":e", "synthetic", a, b, "synthetic://"+id_, "Authored contract fixture")
        records.append(Record(id_, video_id, kind, a, b, text, [e], attributes=attributes,
                              metadata={"synthetic_assertions": assertions}))
    encoder = DemoEncoder()
    store.replace_records(video_id, records, encoder.encode([r.text for r in records]), encoder.identity)
    for stage in ("visual", "transcript", "audio", "ocr", "embedding"):
        store.mark(video_id, stage, 0, 330, "complete", "synthetic fixtures; no media processed")
    cases = [
        ("handcuffs", "Find moments where someone is being handcuffed", ["cuffs_apply"]),
        ("rights", "Find an officer reading Miranda rights", ["rights_recited"]),
        ("red_voice", "Find the person in a red shirt raising their voice", ["red_shouts"]),
        ("night", "Find a vehicle being pulled over at night", ["night_stop"]),
        ("plate", "Find readable license plates", ["plate_clear"]),
        ("absence", "Find a purple helicopter landing", []),
    ]
    by_id = {r.id: r for r in records}
    labels = {"synthetic": True, "description": "Contract fixtures, not model accuracy evaluation",
              "queries": [{"id": id_, "query": q, "split": "test", "iou_threshold": 0.5,
                           "matches": [{"video_id": video_id, "start": by_id[r].start, "end": by_id[r].end} for r in ids]}
                          for id_, q, ids in cases]}
    atomic_json(output, labels)
    return {"synthetic": True, "records": len(records), "labels": str(output),
            "warning": "Toy embeddings and scripted verification; these scores measure orchestration only."}
