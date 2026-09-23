import json
import math
import shutil
import tempfile
import unittest
import wave
import array
from pathlib import Path
from unittest.mock import patch

from video_moment_retrieval import media
from video_moment_retrieval.cache import ArtifactCache
from video_moment_retrieval.demo import DemoEncoder
from video_moment_retrieval.pipeline import IndexConfig, Indexer
from video_moment_retrieval.search import SearchConfig, SearchEngine
from video_moment_retrieval.store import Store
from video_moment_retrieval.types import Candidate, QueryPlan, Verdict
from .test_core import record


class FakeMediaBackend:
    identity = "test-observer-v1"
    def __init__(self):
        self.extract_calls = 0
        self.transcribe_calls = 0
    def transcribe(self, path, duration):
        self.transcribe_calls += 1
        return [{"start": .1, "end": min(duration, .7), "text": "test speech", "speaker": None}]
    def extract(self, path, duration, transcript, context=None):
        self.extract_calls += 1
        return {"summary": "Synthetic test image, not body-camera footage", "observations": [
            {"kind": "appearance", "start": .1, "end": min(duration, .6),
             "text": "test color", "subject": "person_1", "attributes": {"clothing": "red shirt"}}]}
    def read_frames(self, frames):
        return [{"frame_index": 0, "text": None, "text_type": "license_plate", "bbox": [.1,.1,.5,.5],
                 "legibility": "unreadable", "detail": "synthetic OCR test"}]
    def read_crop(self, path):
        return {"text": None, "legibility": "unreadable", "detail": "synthetic crop test"}


class TestSpeechDetector:
    identity = "fixture-vad"
    def detect(self, audio):
        return [{"start": 0, "end": .8, "speaker": None}]


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
class MediaPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.video = Path(cls.fixture.name) / "fixture.mp4"
        media.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10",
                   "-f", "lavfi", "-i", "sine=frequency=400:sample_rate=16000", "-t", "3",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(cls.video)])

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "index.sqlite")
        self.backend = FakeMediaBackend()
        self.cache = ArtifactCache(self.root / "cache")
        self.indexer = Indexer(self.store, self.cache, self.root / "work", self.backend,
                               self.backend, self.backend, DemoEncoder(), speech_detector=TestSpeechDetector())

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_real_decode_clip_audio_and_timestamp_offsets(self):
        info = media.probe(self.video)
        self.assertTrue(info["has_audio"])
        self.assertAlmostEqual(info["duration"], 3, delta=.1)
        report = self.indexer.index(str(self.video), IndexConfig(window_seconds=2, overlap_seconds=.5, enable_ocr=False))
        self.assertEqual(report["errors"], [])
        records = self.store.records()
        windows = [r for r in records if r.kind == "window"]
        self.assertEqual([(r.start, r.end) for r in windows], [(0, 2), (1.5, 3)])
        appearance = [r for r in records if r.kind == "appearance"]
        self.assertAlmostEqual(appearance[1].start, 1.6)
        self.assertNotEqual(appearance[0].subject, appearance[1].subject)
        self.assertEqual({r.status for r in appearance}, {"hypothesis"})
        self.assertFalse(any(r.kind == "audio" for r in records))  # constant sine is not raised voice

    def test_reindex_reuses_models_without_duplicate_records(self):
        cfg = IndexConfig(window_seconds=3, overlap_seconds=0, enable_ocr=False)
        self.indexer.index(str(self.video), cfg)
        count = len(self.store.records())
        self.indexer.index(str(self.video), cfg)
        self.assertEqual((self.backend.extract_calls, self.backend.transcribe_calls), (1, 1))
        self.assertEqual(len(self.store.records()), count)

    def test_chunk_words_use_original_timeline_in_storage_and_local_timeline_for_visuals(self):
        seen = []
        def extract(path, duration, transcript, context=None):
            seen.append(transcript)
            return {"summary": "speech", "observations": []}
        segments = [{"start": "0.1", "end": "0.7", "text": "spoken word", "speaker": "A",
                     "words": [{"word": "spoken", "start": "0.2", "end": "0.6", "speaker": "A"}]}]
        with patch.object(self.backend, "transcribe", return_value=segments) as transcribe, \
                patch.object(self.backend, "extract", side_effect=extract):
            config = IndexConfig(window_seconds=2, overlap_seconds=.5, enable_ocr=False)
            for _ in range(2):  # Cache hits must not apply the offset twice.
                report = self.indexer.index(str(self.video), config)
                self.assertEqual(report["errors"], [])
                speech = sorted((r for r in self.store.records() if r.kind == "speech"), key=lambda r: r.start)
                self.assertAlmostEqual(speech[1].start, 2.1)
                self.assertAlmostEqual(speech[1].metadata["words"][0]["start"], 2.2)
                self.assertAlmostEqual(speech[1].metadata["words"][0]["end"], 2.6)
                self.assertEqual(speech[1].metadata["words"][0]["speaker"], speech[1].speaker)
            self.assertEqual(transcribe.call_count, 2)
        self.assertAlmostEqual(seen[1][0]["words"][0]["start"], .7)
        self.assertAlmostEqual(seen[1][0]["words"][0]["end"], 1.1)

    def test_sidecar_string_times_and_words_are_clipped_to_indexed_range(self):
        sidecar = self.root/"transcript.json"
        sidecar.write_text(json.dumps([{"start": "0.1", "end": "2", "text": "one two",
            "words": [{"word": "one", "start": ".2", "end": ".9"},
                      {"word": "two", "start": "1.5", "end": "2"}]}]))
        report = self.indexer.index(str(self.video), IndexConfig(max_seconds=1, enable_ocr=False), str(sidecar))
        self.assertEqual(report["errors"], [])
        speech = next(r for r in self.store.records() if r.kind == "speech")
        self.assertEqual(speech.end, 1)
        self.assertEqual(speech.metadata["words"], [{"word": "one", "start": .2, "end": .9}])

    def test_failed_visual_stage_is_reported_and_retry_recovers(self):
        cfg = IndexConfig(window_seconds=3, overlap_seconds=0, enable_ocr=False)
        with patch.object(self.backend, "extract", side_effect=RuntimeError("transient")):
            result = self.indexer.index(str(self.video), cfg)
        self.assertEqual(result["errors"][0]["stage"], "visual")
        self.assertIn("failed", self.store.coverage()["videos"][0]["stages"]["visual"]["statuses"])
        retry = self.indexer.index(str(self.video), cfg)
        self.assertEqual(retry["errors"], [])
        self.assertEqual(self.backend.transcribe_calls, 1)

    def test_ocr_detects_unreadable_plate_independently_and_keeps_original_frame(self):
        cfg = IndexConfig(window_seconds=3, overlap_seconds=0, enable_ocr=True, ocr_every_seconds=3)
        result = self.indexer.index(str(self.video), cfg)
        self.assertEqual(result["errors"], [])
        ocr = next(r for r in self.store.records() if r.kind == "ocr")
        self.assertEqual(ocr.status, "unresolved")
        self.assertIsNone(ocr.metadata["text"])
        self.assertTrue(Path(ocr.evidence[0].source).exists())

    def test_partial_index_reports_unprocessed_tail(self):
        result = self.indexer.index(str(self.video), IndexConfig(max_seconds=1, enable_ocr=False))
        stages = self.store.coverage()["videos"][0]["stages"]
        self.assertEqual(stages["visual"]["processed_seconds"], 1)
        self.assertIn("skipped", stages["visual"]["statuses"])

    def test_prepare_caps_clip_and_preserves_audio_for_prosody(self):
        info = media.probe(self.video)
        self.store.add_video("v1", str(self.video), info["duration"], True)
        r = record(start=1, end=2)
        engine = SearchEngine(self.store, self.cache, self.root, DemoEncoder(), self.backend, self.backend)
        prepared = engine.prepare(QueryPlan("raises voice", ["voice"], modalities=["audio"]),
                                  Candidate(r, 1, []), SearchConfig(max_clip_seconds=1))
        self.assertLessEqual(prepared["end"]-prepared["start"], 1)
        self.assertTrue(Path(prepared["audio"]).exists())
        self.assertNotIn("clip", prepared)
        self.assertFalse(prepared["whole_candidate_inspected"])

    def test_silent_video_cannot_verify_audio_query(self):
        self.store.add_video("v1", str(self.video), 3, False)
        engine = SearchEngine(self.store, self.cache, self.root, DemoEncoder(), self.backend, self.backend)
        prepared = engine.prepare(QueryPlan("voice", ["voice"], modalities=["audio"]),
                                  Candidate(record(start=1, end=2), 1, []), SearchConfig())
        self.assertIn("unavailable", prepared)

    def test_rolling_context_changes_invalidate_following_window(self):
        seen = []
        def extract(path, duration, transcript, context=None):
            seen.append(context)
            return {"summary": "first" if context is None else "later", "observations": [
                {"kind":"action","start":.1,"end":duration-.1,"text":"initial ending state" if context is None else "later action"}]}
        config=IndexConfig(window_seconds=2,overlap_seconds=.5,enable_ocr=False)
        with patch.object(self.backend,"extract",side_effect=extract):
            first=self.indexer.index(str(self.video),config)
            self.assertEqual(first["errors"],[])
            self.assertIsNone(seen[0])
            self.assertEqual(seen[1]["observations"][0]["text"],"initial ending state")
            for p in (self.cache.root/"visual").glob("*.json"):
                data=json.loads(p.read_text())
                if data["summary"]=="first":
                    data["observations"][0]["text"]="corrected ending state"
                    p.write_text(json.dumps(data))
            self.indexer.index(str(self.video),config)
        self.assertEqual(len(seen),3)
        self.assertEqual(seen[-1]["observations"][0]["text"],"corrected ending state")

    def test_reconciliation_pass_connects_different_descriptions(self):
        class Reconciler:
            identity="fixture-reconcile"
            def reconcile(self,records):
                observations=[r for r in records if r.kind=="appearance"]
                return [{"kind":"same_person","record_ids":[r.id for r in observations],
                         "evidence_ids":[r.evidence[0].id for r in observations],"reason":"test continuation"}]
        self.indexer.reconciler=Reconciler()
        result=self.indexer.index(str(self.video),IndexConfig(window_seconds=2,overlap_seconds=.5,enable_ocr=False))
        self.assertEqual(result["errors"],[])
        links=self.store.relationships(result["video_id"],0,3)
        self.assertEqual(len(links),1)
        self.assertEqual(links[0]["status"],"hypothesis")
        self.assertTrue(set(links[0]["record_ids"]).issubset({r.id for r in self.store.records()}))

    def test_vad_runs_even_when_transcription_fails(self):
        with patch.object(self.backend,"transcribe",side_effect=RuntimeError("ASR unavailable")):
            result=self.indexer.index(str(self.video),IndexConfig(enable_ocr=False))
        self.assertEqual(result["errors"][0]["stage"],"transcript")
        stages=self.store.coverage()["videos"][0]["stages"]
        self.assertEqual(stages["vad"]["statuses"],["complete"])
        self.assertEqual(stages["audio"]["statuses"],["complete"])

    def test_embedding_failure_preserves_other_indexes(self):
        with patch.object(self.indexer.encoder,"encode",side_effect=RuntimeError("embedding unavailable")):
            result=self.indexer.index(str(self.video),IndexConfig(enable_ocr=False))
        self.assertTrue(any(e["stage"]=="embedding" for e in result["errors"]))
        self.assertGreater(len(self.store.lexical(["test"],10)),0)
        self.assertEqual(self.store.coverage()["videos"][0]["stages"]["embedding"]["statuses"],["failed"])

    def test_completed_pipeline_leaves_no_pending_coverage(self):
        self.indexer.index(str(self.video),IndexConfig(window_seconds=2,overlap_seconds=.5,enable_ocr=False))
        self.assertFalse(any(row["status"]=="pending" for row in self.store.coverage()["videos"][0]["intervals"]))

    def test_full_track_alignment_adapter_is_called_once(self):
        self.backend.full_track=True
        self.backend.alignment="forced_word_alignment"
        self.indexer.index(str(self.video),IndexConfig(window_seconds=1,overlap_seconds=.25,enable_ocr=False))
        self.assertEqual(self.backend.transcribe_calls,1)
        speech=next(r for r in self.store.records() if r.kind=="speech")
        self.assertEqual(speech.metadata["alignment"],"forced_word_alignment")

    def test_delayed_audio_is_not_shifted_earlier_on_extraction(self):
        source=self.root/"delayed.mp4"
        media.run(["ffmpeg","-v","error","-y","-f","lavfi","-i","color=black:size=160x120:rate=10:duration=3",
            "-itsoffset","1","-f","lavfi","-i","sine=frequency=400:sample_rate=16000:duration=1",
            "-t","3","-c:v","libx264","-c:a","aac",str(source)])
        path=media.audio(str(source),self.root/"aligned.wav",0,3)
        with wave.open(path,"rb") as wav:
            self.assertAlmostEqual(wav.getnframes()/wav.getframerate(),3,places=2)
            early=array.array("h",wav.readframes(8000))
            wav.setpos(20000)
            voiced=array.array("h",wav.readframes(2000))
        self.assertLess(max(abs(v) for v in early),10)
        self.assertGreater(max(abs(v) for v in voiced),100)


class WindowTests(unittest.TestCase):
    def test_windows_have_full_coverage_without_redundant_tail(self):
        self.assertEqual(media.windows(40, 20, 5), [(0,20),(15,35),(30,40)])
        self.assertEqual(media.windows(20, 20, 5), [(0,20)])
        for size, overlap in [(0,0),(20,20),(20,-1),(float('inf'),0)]:
            with self.assertRaises(ValueError):
                media.windows(10, size, overlap)
