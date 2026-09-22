import tempfile
import unittest
from pathlib import Path

from video_moment_retrieval.cache import ArtifactCache
from video_moment_retrieval.demo import DemoBackend, DemoEncoder, build_demo
from video_moment_retrieval.evaluation import matching, metrics
from video_moment_retrieval.retrieval import deduplicate, retrieve
from video_moment_retrieval.search import SearchConfig, SearchEngine
from video_moment_retrieval.store import Store
from video_moment_retrieval.types import Candidate, QueryPlan, Verdict
from .test_core import record


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "index.sqlite")
        build_demo(self.store, self.root / "labels.json")
        self.backend = DemoBackend()
        self.engine = SearchEngine(self.store, ArtifactCache(self.root / "cache"), self.root,
                                   DemoEncoder(), self.backend, self.backend, synthetic=True)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_combined_runs_semantic_even_when_structured_nonempty(self):
        plan = self.backend.plan("Find someone being handcuffed")
        candidates, info = retrieve(self.store, DemoEncoder(), plan, 10)
        self.assertGreater(info["channel_hits"]["structured"], 0)
        self.assertGreater(info["channel_hits"]["dense"], 0)
        self.assertLessEqual(len(candidates), 10)

    def test_complete_query_rejects_wrong_person_and_siren(self):
        response = self.engine.search("Find a person in a red shirt raising their voice",
                                      SearchConfig(verify_budget=30))
        self.assertEqual([r["record_id"] for r in response["results"]], ["red_shouts"])
        rejected = {r["record_id"] for r in response["inspected"] if r["status"] == "rejected"}
        self.assertTrue({"red_other_shouts", "siren"}.issubset(rejected))

    def test_unverified_results_are_not_supported(self):
        response = self.engine.search("handcuffed", SearchConfig(verify=False))
        self.assertTrue(all(r["status"] == "candidate" for r in response["results"]))
        self.assertEqual(response["inspected"], [])

    def test_exhaustive_pagination_does_not_truncate_at_ranked_budget(self):
        response = self.engine.search("handcuffed", SearchConfig(candidate_budget=1, verify_budget=2, enumerate_all=True))
        self.assertEqual(response["pagination"]["candidate_count"], 11)
        self.assertEqual(response["pagination"]["next_offset"], 2)
        next_page = self.engine.search("handcuffed", SearchConfig(candidate_budget=1, verify_budget=2,
            enumerate_all=True, offset=2, snapshot=response["pagination"]["snapshot"]))
        self.assertFalse({i["record_id"] for i in response["inspected"]} & {i["record_id"] for i in next_page["inspected"]})
        with self.assertRaises(ValueError):
            self.engine.search("handcuffed", SearchConfig(snapshot="changed"))

    def test_verification_error_does_not_promote_candidate(self):
        class BadVerifier:
            identity = "bad"
            def verify(self, plan, candidate, media):
                return Verdict("supported", "invented", 0, 10000, ["fake"])
        self.engine.verifier = BadVerifier()
        response = self.engine.search("handcuffed", SearchConfig())
        self.assertEqual(response["results"], [])
        self.assertTrue(all(r["status"] == "unresolved" for r in response["inspected"]))

    def test_verification_budget_caps_calls_and_cache_reuses(self):
        class Counting(DemoBackend):
            def __init__(self): self.calls = 0
            def verify(self, plan, candidate, media):
                self.calls += 1
                return super().verify(plan, candidate, media)
        verifier = Counting()
        self.engine.verifier = verifier
        self.engine.search("handcuffed", SearchConfig(verify_budget=2))
        self.assertEqual(verifier.calls, 2)
        self.engine.search("handcuffed", SearchConfig(verify_budget=2))
        self.assertEqual(verifier.calls, 2)

    def test_no_match_abstains(self):
        response = self.engine.search("purple helicopter landing", SearchConfig(verify_budget=30))
        self.assertEqual(response["results"], [])

    def test_duplicate_windows_do_not_transitively_merge_separate_actions(self):
        a = record("a", start=1, end=3)
        b = record("b", start=8, end=10)
        window = record("window", start=0, end=20)
        window.kind = "window"
        groups = deduplicate([Candidate(a, 1, []), Candidate(window, .9, []), Candidate(b, .8, [])])
        self.assertEqual(len(groups), 3)  # broad coverage window remains independently searchable


class EvaluationTests(unittest.TestCase):
    def test_duplicate_predictions_do_not_inflate_recall(self):
        interval = {"video_id": "v", "start": 1, "end": 3}
        result = metrics([interval, interval], [interval, interval], [interval])
        self.assertEqual(result["candidate_recall"], 1)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1)

    def test_abstention_not_reported_as_perfect_precision(self):
        result = metrics([], [], [{"video_id": "v", "start": 1, "end": 3}])
        self.assertIsNone(result["precision"])
        self.assertEqual(result["recall"], 0)
        self.assertTrue(result["abstained"])

    def test_maximum_matching_not_greedy(self):
        truth = [{"video_id": "v", "start": 0, "end": 3}, {"video_id": "v", "start": 3, "end": 6}]
        pred = [{"video_id": "v", "start": 0, "end": 6}, {"video_id": "v", "start": 0, "end": 3}]
        self.assertEqual(len(matching(pred, truth, .5)), 2)

    def test_onset_metric_differs_from_overlap(self):
        truth = [{"video_id": "v", "start": 5, "end": 20}]
        pred = [{"video_id": "v", "start": 5.2, "end": 6}]
        self.assertEqual(metrics(pred, pred, truth)["recall"], 0)
        self.assertEqual(metrics(pred, pred, truth, onset_tolerance=.5)["recall"], 1)
