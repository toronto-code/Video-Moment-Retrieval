import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from video_moment_retrieval.speech import WhisperXTranscriber


class AlignmentAdapterTests(unittest.TestCase):
    def test_adapter_runs_asr_then_forced_alignment_and_preserves_words(self):
        model=SimpleNamespace(transcribe=Mock(return_value={"language":"en","segments":[{"start":0,"end":1,"text":"hello"}]}))
        aligned={"segments":[{"start":.2,"end":.7,"text":"hello",
                              "words":[{"word":"hello","start":.2,"end":.7}]}]}
        module=SimpleNamespace(load_model=Mock(return_value=model),load_audio=Mock(return_value="waveform"),
            load_align_model=Mock(return_value=("aligner",{})),align=Mock(return_value=aligned))
        with patch.dict(sys.modules,{"whisperx":module}),patch("video_moment_retrieval.speech.version",return_value="test"):
            adapter=WhisperXTranscriber()
            segments=adapter.transcribe("audio.wav",1)
        self.assertEqual(segments[0]["words"][0]["start"],.2)
        self.assertEqual(segments[0]["alignment"],"forced_word_alignment")
        module.align.assert_called_once()
        self.assertTrue(adapter.full_track)
