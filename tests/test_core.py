import tempfile
import unittest
from pathlib import Path

from video_moment_retrieval.cache import ArtifactCache
from video_moment_retrieval.store import Store
from video_moment_retrieval.types import Evidence, Record, Verdict


def record(id_="r1", text="officer secures wrists", start=2, end=5, **kwargs):
    return Record(id_, "v1", "action", start, end, text,
                  [Evidence(id_+"e", "visual", start, end, "fixture")], **kwargs)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "index.sqlite")
        self.store.add_video("v1", "fixture.mp4", 30, True)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_transactional_replace_and_no_duplicate_fts(self):
        self.store.replace_records("v1", [record()], [[1, 0]], "encoder")
        self.store.replace_records("v1", [record(text="new observation")], [[0, 1]], "encoder")
        self.assertEqual(self.store.lexical(["wrists"], 10), [])
        self.assertEqual(len(self.store.lexical(["observation"], 10)), 1)
        with self.assertRaises(ValueError):
            self.store.replace_records("v1", [record(end=31)], [[1, 0]], "encoder")
        self.assertEqual(self.store.records()[0].text, "new observation")

    def test_lexical_user_input_does_not_become_sql_or_fts_syntax(self):
        self.store.replace_records("v1", [record()], [[1, 0]], "encoder")
        self.assertEqual(len(self.store.lexical(['wrists" OR *; DROP TABLE records'], 10)), 1)
        self.assertEqual(len(self.store.records()), 1)

    def test_semantic_rejects_wrong_encoder_and_dimension(self):
        self.store.replace_records("v1", [record()], [[1, 0]], "encoder")
        for vector, encoder in [([1, 0], "wrong"), ([1], "encoder")]:
            with self.assertRaises(ValueError):
                self.store.semantic(vector, encoder, 10)

    def test_structured_is_candidate_union_not_uncertain_hard_filter(self):
        self.store.replace_records("v1", [record(attributes={"lighting": "night"})], [[1]], "encoder")
        self.assertEqual(len(self.store.structured({"lighting": "night", "action": "arrest"}, 5)), 1)

    def test_overlapping_coverage_is_not_double_counted(self):
        self.store.mark("v1", "visual", 0, 20, "complete")
        self.store.mark("v1", "visual", 15, 25, "complete")
        self.store.mark("v1", "visual", 25, 30, "failed", "timeout")
        stage = self.store.coverage()["videos"][0]["stages"]["visual"]
        self.assertEqual(stage["processed_seconds"], 25)
        self.assertIn("failed", stage["statuses"])

    def test_cache_invalidates_on_upstream_change_and_retries_failure(self):
        cache = ArtifactCache(Path(self.tmp.name) / "cache")
        calls = []
        def compute():
            calls.append(1)
            return {"ok": True}
        for upstream in ["a", "a", "b"]:
            cache.get("extract", {"upstream": upstream}, compute)
        self.assertEqual(len(calls), 2)
        with self.assertRaises(RuntimeError):
            cache.get("fail", {}, lambda: (_ for _ in ()).throw(RuntimeError("failed")))
        self.assertEqual(cache.get("fail", {}, compute), {"ok": True})

    def test_supported_verdict_requires_all_criteria_and_real_evidence(self):
        v = Verdict("supported", "yes", 2, 5, ["e"], [])
        with self.assertRaises(ValueError):
            v.validate(0, 10, {"e"}, ["applies cuffs"])
        v.criteria = [{"criterion": "applies cuffs", "status": "supported", "evidence_ids": ["e"]}]
        v.validate(0, 10, {"e"}, ["applies cuffs"])
        v.end = 11
        with self.assertRaises(ValueError):
            v.validate(0, 10, {"e"}, ["applies cuffs"])

    def test_supported_link_cannot_reference_nonexistent_evidence(self):
        with self.assertRaises(ValueError):
            record(links=[{"status": "supported", "start": 2, "end": 5, "evidence_ids": ["invented"]}])


if __name__ == "__main__":
    unittest.main()
