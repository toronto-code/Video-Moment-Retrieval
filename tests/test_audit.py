import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from video_moment_retrieval.assessment import assess_candidates
from video_moment_retrieval.cache import ArtifactCache, atomic_json
from video_moment_retrieval.demo import DemoBackend, DemoEncoder
from video_moment_retrieval.evaluation import metrics
from video_moment_retrieval.relationships import attach_reconciliation, attach_visual_links
from video_moment_retrieval.retrieval import retrieve
from video_moment_retrieval.search import SearchConfig, SearchEngine, bounds, verification_tasks
from video_moment_retrieval.speech import attach_speakers
from video_moment_retrieval.store import Store
from video_moment_retrieval.types import Candidate, Evidence, QueryPlan, Record, Verdict, Verification


def rec(id_, start, end, text="event", kind="window"):
    return Record(id_, "v", kind, start, end, text, [Evidence(id_+":e", "synthetic", start, end, "fixture")])


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "index.sqlite")
        self.store.add_video("v", "synthetic://test", 120, True)
        self.cache = ArtifactCache(self.root / "cache")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def engine(self, records, verifier=None):
        encoder = DemoEncoder()
        self.store.replace_records("v", records, encoder.encode([r.text for r in records]), encoder.identity)
        return SearchEngine(self.store, self.cache, self.root, encoder, DemoBackend(), verifier or DemoBackend(), True)

    def test_all_mode_slices_cover_long_record_including_tail(self):
        candidate = Candidate(rec("long", 10, 100), 1, [])
        cfg = SearchConfig(max_clip_seconds=20, context_seconds=3, enumerate_all=True)
        tasks = verification_tasks([candidate], cfg)
        self.assertGreater(len(tasks), 1)
        self.assertEqual(tasks[0].target_start, 10)
        self.assertEqual(tasks[-1].target_end, 100)
        for a, b in zip(tasks, tasks[1:]):
            self.assertEqual(a.target_end, b.target_start)
        for task in tasks:
            start, end = bounds(task, cfg, 120)
            self.assertLessEqual(end-start, 20)
            self.assertLessEqual(start, task.target_start)
            self.assertGreaterEqual(end, task.target_end)

    def test_tail_match_and_multiple_matches_are_returned(self):
        class Verifier:
            identity = "multi-tail"
            def verify(self, plan, candidate, media):
                found = []
                for start, end in [(4,5),(8,9),(58,59)]:
                    if start >= media["start"] and end <= media["end"]:
                        found.append(Verdict("supported", "fixture", start, end, ["long:e"],
                            [{"criterion": c, "status": "supported", "evidence_ids": ["long:e"]} for c in plan.criteria]))
                return Verification(found or [Verdict("rejected", "none here")], True)
        engine = self.engine([rec("long", 0, 60)], Verifier())
        response = engine.search("test", SearchConfig(enumerate_all=True, verify_budget=20, max_clip_seconds=20))
        self.assertEqual([(r["start"],r["end"]) for r in response["results"]], [(4,5),(8,9),(58,59)])
        self.assertTrue(response["pagination"]["all_tasks_verified"])

    def test_across_page_duplicate_results_are_suppressed(self):
        class V:
            identity = "duplicate"
            def verify(self, plan, candidate, media):
                ids = [candidate.record.evidence[0].id]
                return Verification([Verdict("supported", "same event", 6, 8, ids,
                    [{"criterion": c, "status": "supported", "evidence_ids": ids} for c in plan.criteria])], True)
        engine = self.engine([rec("a", 0, 10), rec("b", 1, 11)], V())
        first = engine.search("test", SearchConfig(enumerate_all=True, verify_budget=1))
        second = engine.search("test", SearchConfig(enumerate_all=True, verify_budget=1,
            offset=1, snapshot=first["pagination"]["snapshot"]))
        self.assertEqual(len(first["results"]), 1)
        self.assertEqual(second["results"], [])
        self.assertEqual(len(second["accumulated_results"]), 1)
        self.assertEqual(second, engine.search("test", SearchConfig(enumerate_all=True, verify_budget=1,
            offset=1, snapshot=first["pagination"]["snapshot"])))

    def test_changed_inspection_geometry_rejects_cursor(self):
        engine = self.engine([rec("a", 0, 60)])
        first = engine.search("test", SearchConfig(enumerate_all=True, verify_budget=1))
        with self.assertRaisesRegex(ValueError, "settings changed"):
            engine.search("test", SearchConfig(enumerate_all=True, verify_budget=1, max_clip_seconds=10,
                offset=1, snapshot=first["pagination"]["snapshot"]))

    def test_cached_invalid_verification_is_recomputed_not_trusted(self):
        engine = self.engine([rec("a", 1, 3)])
        engine.search("test", SearchConfig(assess=False))
        path = next((self.cache.root / "verification").glob("*.json"))
        data = json.loads(path.read_text())
        data["verification"]["verdicts"] = [{"status":"supported","reason":"invented", "start":999,"end":1000,
            "evidence_ids":["imaginary"],"criteria":[]}]
        atomic_json(path, data)
        self.assertEqual(engine.search("test", SearchConfig(assess=False))["results"], [])

    def test_relationship_is_stored_and_overlap_cannot_be_supported(self):
        r = rec("person", 1, 3, kind="appearance")
        r.subject = "w:A"
        attach_visual_links([r], [{"person":"A","speaker":"S","start":1,"end":2,
            "status":"supported","method":"temporal_overlap","detail":"both present"}],
            [{"speaker":"S"}], 0, 10, "fixture", "w", True)
        self.assertEqual(r.links[0]["status"], "unresolved")
        self.engine([r])
        self.assertEqual(self.store.relationships("v", 0, 10)[0]["person"], "w:A")

    def test_reconciliation_cannot_invent_or_promote_identity(self):
        a,b=rec("a", 1, 5),rec("b", 4, 9)
        attach_reconciliation([a,b], [{"kind":"event_continuation","record_ids":["a","b"],
            "evidence_ids":["a:e","b:e"],"reason":"continuous action","status":"supported"}])
        self.assertEqual(b.links[0]["status"], "hypothesis")
        self.assertEqual({e.id for e in b.evidence}, {"a:e","b:e"})
        with self.assertRaises(ValueError):
            attach_reconciliation([a,b], [{"kind":"same_person","record_ids":["a","missing"],
                "evidence_ids":["a:e"],"reason":"guess"}])

    def test_assessment_reorders_but_does_not_discard_unknowns(self):
        class Assessor:
            identity = "fixture-assessor"
            def assess(self, plan, candidates):
                return [{"record_id": c.record.id, "status": "likely" if c.record.id=="b" else "unresolved",
                    "reason":"fixture", "evidence_ids":[c.record.evidence[0].id]} for c in candidates]
        a,b=Candidate(rec("a",1,2),2,[]),Candidate(rec("b",3,4),1,[])
        ordered, errors=assess_candidates(Assessor(),self.cache,QueryPlan("test",["test"]),[a,b])
        self.assertEqual([c.record.id for c in ordered],["b","a"])
        self.assertEqual(errors,[])
        ordered,_=assess_candidates(Assessor(),self.cache,
            QueryPlan("test",["test"],requires_subject_link=True),[a,b])
        self.assertTrue(all(c.assessment["status"]=="unresolved" for c in ordered))

    def test_embedding_failure_does_not_block_lexical_publication(self):
        self.store.replace_records("v", [rec("a",1,2,"red shirt")], None, "broken")
        self.assertEqual(len(self.store.lexical(["red"],10)),1)
        self.assertEqual(self.store.semantic([1],"broken",10),[])

    def test_retrieval_channels_execute_concurrently(self):
        engine=self.engine([rec("a",1,2,"red shirt")])
        barrier=threading.Barrier(3,timeout=3)
        def wait(*args):
            barrier.wait()
            return [("a",1)]
        with patch.object(Store,"lexical",side_effect=wait), patch.object(Store,"structured",side_effect=wait), patch.object(Store,"semantic",side_effect=wait):
            candidates,_=retrieve(self.store,DemoEncoder(),QueryPlan("red",["red"],{"clothing":"red"}))
        self.assertEqual(len(candidates),1)

    def test_vad_speaker_attachment_preserves_unknown_and_overlap(self):
        result=attach_speakers([{"start":0,"end":5}], [{"start":1,"end":3,"speaker":"A"},
                                                       {"start":2,"end":4,"speaker":"B"}])
        self.assertEqual([(r["start"],r["end"],r["speaker"]) for r in result],
            [(0,1,None),(1,2,"A"),(2,3,None),(3,4,"B"),(4,5,None)])

    def test_one_coverage_window_can_retrieve_two_events(self):
        candidates=[{"video_id":"v","start":0,"end":20}]
        truth=[{"video_id":"v","start":2,"end":3},{"video_id":"v","start":17,"end":18}]
        scores=metrics(candidates,[],truth)
        self.assertEqual(scores["candidate_recall"],1)
        self.assertEqual(scores["recall"],0)

    def test_correct_timestamp_wrong_plate_is_not_a_true_positive(self):
        truth=[{"video_id":"v","start":2,"end":3,"details":{"text":"ABC123"}}]
        wrong=[{"video_id":"v","start":2,"end":3,"details":{"text":"ABC128"}}]
        self.assertEqual(metrics(wrong,wrong,truth)["precision"],0)

    def test_conflicting_duplicate_criteria_are_invalid(self):
        verdict=Verdict("supported","conflicting",1,2,["e"],[
            {"criterion":"applies cuffs","status":"supported","evidence_ids":["e"]},
            {"criterion":"applies cuffs","status":"rejected","evidence_ids":["e"]}])
        with self.assertRaisesRegex(ValueError,"unambiguous"):
            verdict.validate(0,3,{"e"},["applies cuffs"])

    def test_verifier_cannot_claim_same_person_without_binding(self):
        engine=self.engine([rec("a",1,3)])
        engine.synthetic=False
        plan=QueryPlan("red shirt shouts",["red"],modalities=["visual","audio"],criteria=["same person"],requires_subject_link=True)
        clip=self.root/"clip.mp4"; clip.write_bytes(b"test")
        audio=self.root/"audio.wav"; audio.write_bytes(b"test")
        verdict=Verdict("supported","co-occurrence",1,2,["media:visual","media:audio"],[
            {"criterion":"same person","status":"supported","evidence_ids":["media:visual","media:audio"]}])
        value={"verification":Verification([verdict],True).to_dict(),
               "inspection":{"start":0,"end":4,"clip":str(clip),"audio":str(audio)}}
        with self.assertRaisesRegex(ValueError,"affirmative"):
            engine.validate_verification(value,plan,Candidate(rec("a",1,3),1,[]))
