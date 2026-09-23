import io
import sqlite3
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from video_moment_retrieval.download import download, entries
from video_moment_retrieval.evaluation import metrics
from video_moment_retrieval.pipeline import transcript_slice, validate_segments, visual_records
from video_moment_retrieval.search import candidate_summary
from video_moment_retrieval.store import Store
from video_moment_retrieval.types import Candidate, Evidence, Record


class ReviewFixTests(unittest.TestCase):
    def test_speaker_survives_materialization_storage_and_structured_search(self):
        data = {"summary": "speech", "observations": [{"start": "1", "end": "2",
            "kind": "audio", "text": "raised voice", "speaker": "asr:0:A"}]}
        records = visual_records(data, "v", 1, 10, 15, "video", [{"speaker": "asr:0:A"}], True)
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp)/"index.sqlite")
            try:
                store.add_video("v", "video", 20, True)
                store.replace_records("v", records, None, "fixture")
                matches = store.structured({"speaker": "asr:0:A"}, 10)
                self.assertEqual([r.speaker for r in store.records([m[0] for m in matches])], ["asr:0:A"])
                self.assertEqual((records[1].start, records[1].end), (11, 12))
            finally:
                store.close()
        with self.assertRaisesRegex(ValueError, "speaker"):
            visual_records(data, "v", 1, 10, 15, "video", [], True)

    def test_normalized_words_shift_and_clip_without_mutating_cached_segments(self):
        segments = [{"start": "0.5", "end": "2.5", "text": "one two",
            "words": [{"word": "one", "start": "0.5", "end": "1.0"},
                      {"word": "two", "start": "2", "end": "2.5"}, {"word": "unaligned"}]}]
        validate_segments(segments, 3)
        original = deepcopy(segments)
        shifted = transcript_slice(segments, 0, 3, 20)
        self.assertEqual(shifted[0]["words"][1]["start"], 22)
        self.assertEqual(shifted[0]["words"][2], {"word": "unaligned"})
        local = transcript_slice(shifted, 21, 22.2, -21)
        self.assertEqual(len(local[0]["words"]), 1)
        self.assertEqual(local[0]["words"][0]["word"], "two")
        self.assertEqual(local[0]["words"][0]["start"], 1)
        self.assertAlmostEqual(local[0]["words"][0]["end"], 1.2)
        self.assertEqual(segments, original)

    def test_invalid_nested_word_times_are_rejected(self):
        for word in ({"start": "nan"}, {"start": 4}, {"start": 2, "end": 1}):
            with self.subTest(word=word), self.assertRaises(ValueError):
                validate_segments([{"start": 0, "end": 3, "text": "word", "words": [word]}], 3)

    def test_read_only_store_never_creates_directories_and_rejects_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp)/"absent"/"index.sqlite"
            with self.assertRaises(sqlite3.OperationalError):
                Store(missing, read_only=True)
            self.assertFalse(missing.parent.exists())
            path = Path(tmp)/"index.sqlite"
            Store(path).close()
            with patch.object(Path, "mkdir", side_effect=AssertionError("read-only mkdir")):
                store = Store(path, read_only=True)
            try:
                with self.assertRaises(sqlite3.OperationalError):
                    store.set_meta("key", "value")
            finally:
                store.close()

    def test_detail_recall_distinguishes_missing_wrong_correct_and_wrong_time(self):
        truth = [{"video_id": "v", "start": 2, "end": 3, "details": {"text": "ABC123"}}]
        candidate = {"video_id": "v", "start": 0, "end": 20}
        for details, expected in (({}, 0), ({"text": "ABC128"}, 0), ({"text": "ABC123"}, 1)):
            result = metrics([{**candidate, "details": details}], [], truth)
            self.assertEqual(result["candidate_recall"], expected)
            self.assertEqual(result["candidate_temporal_recall"], 1)
        observation = {"video_id": "v", "start": 12, "end": 13, "details": {"text": "ABC123"}}
        self.assertEqual(metrics([{**candidate, "detail_evidence": [observation]}], [], truth)["candidate_recall"], 0)

    def test_candidate_summary_exposes_ocr_details_with_their_own_intervals(self):
        window = Record("w", "v", "window", 0, 20, "street", [Evidence("e", "visual", 0, 20, "video")])
        ocr = Record("o", "v", "ocr", 2.5, 2.51, "plate", [Evidence("oe", "ocr", 2.5, 2.51, "crop")],
                     metadata={"text": "ABC123"}, attributes={"legibility": "readable"})
        candidate = candidate_summary(Candidate(window, 1, [], members=[ocr]))
        truth = [{"video_id": "v", "start": 2, "end": 3, "details": {"text": "ABC123"}}]
        self.assertEqual(metrics([candidate], [], truth)["candidate_recall"], 1)


class DownloadFixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.script = self.root/"download.sh"
        self.url = "https://storage.googleapis.com/bucket/video?signature=SECRET"
        self.script.write_text(f'curl -o video.mp4 "{self.url}"\n')
        self.target = self.root/"video.mp4"

    def response(self, body=b"video bytes", length=None):
        response = io.BytesIO(body)
        response.status = 200
        response.headers = {"Content-Length": str(len(body) if length is None else length), "Content-Type": "video/mp4"}
        return response

    def fetch(self):
        return download(str(self.script), str(self.root), ["video.mp4"])

    def test_duplicate_output_names_are_rejected_including_case_collisions(self):
        for name in ("video.mp4", "VIDEO.mp4"):
            self.script.write_text(f'curl -o video.mp4 "{self.url}"\ncurl -o {name} "{self.url}/other"\n')
            with self.assertRaisesRegex(ValueError, "Duplicate download filename"):
                entries(self.script)

    def test_existing_file_is_replaced_then_reused_only_with_matching_checksum(self):
        self.target.write_bytes(b"old truncated download")
        with patch("urllib.request.urlopen", return_value=self.response()) as request, \
                patch("video_moment_retrieval.download.probe", return_value={}) as probe:
            self.fetch()
            self.assertEqual(request.call_count, 1)
            probe.assert_called_once()
        self.assertEqual(self.target.read_bytes(), b"video bytes")
        receipt = self.target.with_suffix(".mp4.download.json")
        self.assertNotIn("SECRET", receipt.read_text())
        with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
            self.fetch()
        self.target.write_bytes(b"wrong bytes")  # Same length; a size-only check would miss it.
        with patch("urllib.request.urlopen", return_value=self.response()) as request, \
                patch("video_moment_retrieval.download.probe", return_value={}):
            self.fetch()
            self.assertEqual(request.call_count, 1)

    def test_incomplete_transfer_preserves_existing_target_and_removes_partial(self):
        self.target.write_bytes(b"existing file")
        with patch("urllib.request.urlopen", return_value=self.response(length=100)), \
                self.assertRaisesRegex(RuntimeError, "Incomplete download"):
            self.fetch()
        self.assertEqual(self.target.read_bytes(), b"existing file")
        self.assertFalse(self.target.with_suffix(".mp4.partial").exists())
        self.assertFalse(self.target.with_suffix(".mp4.download.json").exists())

    def test_nonvideo_response_is_not_published(self):
        with patch("urllib.request.urlopen", return_value=self.response(b"<html>error</html>")), \
                patch("video_moment_retrieval.download.probe", side_effect=ValueError("Input has no video stream")), \
                self.assertRaisesRegex(ValueError, "no video stream"):
            self.fetch()
        self.assertFalse(self.target.exists())
