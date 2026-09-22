import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from video_moment_retrieval.cache import ArtifactCache
from video_moment_retrieval.download import entries
from video_moment_retrieval.providers import OpenRouterBackend, OpenRouterClient, OpenRouterEncoder, data_part
from video_moment_retrieval.types import Candidate, QueryPlan
from .test_core import record


class ProviderTests(unittest.TestCase):
    def test_retry_budget_and_no_secret_in_errors(self):
        c = OpenRouterClient("TOP_SECRET", max_requests=2, retries=1)
        error = urllib.error.HTTPError("https://example", 503, "bad", {}, None)
        with patch("urllib.request.urlopen", side_effect=error), patch("time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "HTTP 503") as exc:
                c.request("embeddings", {"model": "m"})
            self.assertNotIn("TOP_SECRET", str(exc.exception))
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                c.request("embeddings", {"model": "m"})
        self.assertEqual(c.attempts, 2)

    def test_auth_errors_not_retried(self):
        c = OpenRouterClient("secret", retries=3)
        error = urllib.error.HTTPError("https://example", 401, "bad", {}, None)
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(RuntimeError):
                c.request("embeddings", {"model": "m"})
        self.assertEqual(c.attempts, 1)

    def test_truncated_json_is_not_accepted(self):
        c = OpenRouterClient("secret")
        with patch.object(c, "request", return_value={"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}):
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                c.chat("m", "prompt")

    def test_embeddings_order_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = OpenRouterClient("secret")
            encoder = OpenRouterEncoder(c, ArtifactCache(tmp), "m")
            with patch.object(c, "request", return_value={"data": [
                    {"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [2, 0]}]}) as req:
                self.assertEqual(encoder.encode(["a", "b"]), [[1, 0], [0, 1]])
                encoder.encode(["a", "b"])
                self.assertEqual(req.call_count, 1)

    def test_audio_verification_receives_audio_and_offsets_local_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audio.wav"
            path.write_bytes(b"fake WAV bytes for request contract")
            c = OpenRouterClient("secret")
            backend = OpenRouterBackend(c, ArtifactCache(tmp))
            plan = QueryPlan("raised voice", ["voice"], modalities=["audio"], criteria=["speech becomes raised"])
            response = {"status": "supported", "reason": "speech", "start": 1, "end": 2,
                "evidence_ids": ["media:audio"], "criteria": [{"criterion": "speech becomes raised",
                "status": "supported", "evidence_ids": ["media:audio"]}]}
            with patch.object(c, "chat", return_value=response) as call:
                verdict = backend.verify(plan, Candidate(record(), 1, []),
                                         {"start": 100, "end": 110, "audio": str(path)})
            self.assertEqual((verdict.verdicts[0].start, verdict.verdicts[0].end), (101, 102))
            self.assertEqual(call.call_args.args[2][0]["type"], "input_audio")

    def test_download_script_is_parsed_not_executed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "download.sh"
            p.write_text('echo malicious\ncurl -L -o "video_1.mp4" "https://storage.googleapis.com/bucket/video?signature=secret"\n')
            self.assertEqual(entries(p)[0][0], "video_1.mp4")
            p.write_text('curl -o "../../escape.mp4" "https://storage.googleapis.com/x"')
            with self.assertRaises(ValueError):
                entries(p)

    def test_supported_audio_verdict_cannot_cite_only_visual_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "audio.wav"
            clip = Path(tmp) / "clip.mp4"
            audio.write_bytes(b"test")
            clip.write_bytes(b"test")
            c = OpenRouterClient("secret")
            backend = OpenRouterBackend(c, ArtifactCache(tmp))
            response = {"status": "supported", "reason": "unsupported inference", "start": 1, "end": 2,
                "evidence_ids": ["media:visual"], "criteria": [{"criterion": "speech",
                "status": "supported", "evidence_ids": ["media:visual"]}]}
            with patch.object(c, "chat", return_value=response), self.assertRaisesRegex(ValueError, "every required modality"):
                backend.verify(QueryPlan("shouts", ["shouts"], modalities=["audio", "visual"], criteria=["speech"]),
                    Candidate(record(), 1, []), {"start": 0, "end": 10, "audio": str(audio), "clip": str(clip)})

    def test_component_identity_does_not_reindex_visuals_on_audio_model_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            client=OpenRouterClient("secret")
            with patch.dict("os.environ",{"VIDEO_SEARCH_AUDIO_MODEL":"audio-one"}):
                first=OpenRouterBackend(client,ArtifactCache(tmp))
            with patch.dict("os.environ",{"VIDEO_SEARCH_AUDIO_MODEL":"audio-two"}):
                second=OpenRouterBackend(client,ArtifactCache(tmp))
            self.assertEqual(first.identities["visual"],second.identities["visual"])
            self.assertEqual(first.identities["assessment"],second.identities["assessment"])
            self.assertNotEqual(first.identities["transcript"],second.identities["transcript"])
            self.assertNotEqual(first.identities["verification"],second.identities["verification"])
