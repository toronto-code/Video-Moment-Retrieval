import array
import math
import sys
import tempfile
import unittest
import wave
import importlib.util
from pathlib import Path

from video_moment_retrieval.media import energy_observations
from video_moment_retrieval.speech import WebRTCSpeechDetector


class AudioTests(unittest.TestCase):
    def fixture(self, path):
        values = array.array("h")
        for i in range(16000*3):
            amplitude = .02 if i < 16000*2 else .5
            values.append(int(32767*amplitude*math.sin(2*math.pi*400*i/16000)))
        if sys.byteorder != "little":
            values.byteswap()
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes(values.tobytes())

    def test_energy_change_is_only_speech_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "energy.wav"
            self.fixture(p)
            self.assertEqual(energy_observations(str(p), []), [])
            changes = energy_observations(str(p), [{"start": 0, "end": 3, "speaker": None}])
            self.assertGreater(len(changes), 0)
            self.assertTrue(all(c["behavior"] == "raised_voice_candidate" and
                                c["baseline_status"] == "unresolved" for c in changes))
            self.assertGreater(changes[0]["delta_db"], 6)

    def test_different_speaker_does_not_inherit_quiet_speaker_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "energy.wav"
            self.fixture(p)
            changes = energy_observations(str(p), [{"start": 0, "end": 2, "speaker": "A"},
                                                    {"start": 2, "end": 3, "speaker": "B"}])
            self.assertEqual(changes, [])

    @unittest.skipUnless(importlib.util.find_spec("webrtcvad"), "Install the speech extra for native VAD tests")
    def test_native_vad_rejects_silence_and_handles_partial_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"silence.wav"
            with wave.open(str(p),"wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(16000)
                out.writeframes(b"\0\0"*16017)
            self.assertEqual(WebRTCSpeechDetector().detect(str(p)),[])
