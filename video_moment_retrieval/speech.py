"""Independent VAD and optional locally aligned transcription adapters."""
from __future__ import annotations

import wave
from importlib.metadata import PackageNotFoundError, version


class WebRTCSpeechDetector:
    def __init__(self, mode: int = 2):
        if mode not in range(4):
            raise ValueError("VAD mode must be 0–3")
        self.mode = mode
        try:
            import webrtcvad
        except ImportError:
            raise RuntimeError("Install speech dependencies: python -m pip install '.[speech]'. No transcript-as-VAD fallback is used.") from None
        self.vad = webrtcvad.Vad(mode)
        for distribution in ("webrtcvad-wheels", "webrtcvad"):
            try:
                release = version(distribution)
                break
            except PackageNotFoundError:
                continue
        else:
            release = getattr(webrtcvad, "__version__", None)
            if not isinstance(release, str) or not release:
                raise RuntimeError("Cannot determine WebRTC VAD version; reinstall the speech extra")
        self.identity = f"webrtcvad:{release}:mode={mode}:20ms-v1"

    def detect(self, audio: str) -> list[dict]:
        with wave.open(audio, "rb") as wav:
            rate = wav.getframerate()
            if wav.getsampwidth() != 2 or wav.getnchannels() != 1 or rate not in {8000, 16000, 32000, 48000}:
                raise ValueError("VAD requires 16-bit mono PCM at a supported rate")
            samples = rate//50
            duration = wav.getnframes()/rate
            intervals = []
            i = 0
            while raw := wav.readframes(samples):
                # Pad the final short frame for classification, but retain its real end time.
                raw = raw.ljust(samples*2, b"\0")
                if self.vad.is_speech(raw, rate):
                    start, end = i*.02, min((i+1)*.02, duration)
                    if intervals and start - intervals[-1]["end"] <= .12:
                        intervals[-1]["end"] = end
                    else:
                        intervals.append({"start": start, "end": end, "speaker": None,
                                          "speech_status": "vad_candidate"})
                i += 1
        return intervals


def attach_speakers(regions: list[dict], transcript: list[dict]) -> list[dict]:
    """Split VAD regions on diarization boundaries. Never assign one speaker across overlap."""
    result = []
    for region in regions:
        boundaries = {region["start"], region["end"]}
        for s in transcript:
            if s["start"] < region["end"] and s["end"] > region["start"]:
                boundaries.update((max(region["start"], s["start"]), min(region["end"], s["end"])))
        ordered = sorted(boundaries)
        for start, end in zip(ordered, ordered[1:]):
            active = [s for s in transcript if s["start"] < end and s["end"] > start]
            speakers = {s.get("speaker") for s in active}
            speaker = next(iter(speakers)) if len(speakers) == 1 and len(active) == 1 else None
            result.append({**region, "start": start, "end": end, "speaker": speaker,
                           "speaker_status": "diarization_hypothesis" if speaker else "unresolved"})
    return result


class WhisperXTranscriber:
    """One full-track ASR/alignment pass; diarization is optional and explicitly labeled."""
    full_track = True
    alignment = "forced_word_alignment"

    def __init__(self, model: str = "small.en", device: str = "cpu", diarize: bool = False,
                 hf_token: str | None = None):
        try:
            import whisperx
        except ImportError:
            raise RuntimeError("Install the optional alignment extra in a compatible Python environment: pip install '.[alignment]'") from None
        self.wx = whisperx
        self.device, self.diarize, self.hf_token = device, diarize, hf_token
        if diarize and not hf_token:
            raise ValueError("WhisperX diarization requires HF_TOKEN and access to its speaker model")
        self.identity = f"whisperx:{version('whisperx')}:{model}:{device}:diarize={diarize}"
        self.model = whisperx.load_model(model, device, compute_type="int8" if device == "cpu" else "float16")

    def transcribe(self, audio: str, duration: float) -> list[dict]:
        waveform = self.wx.load_audio(audio)
        result = self.model.transcribe(waveform, batch_size=4)
        model, metadata = self.wx.load_align_model(language_code=result["language"], device=self.device)
        aligned = self.wx.align(result["segments"], model, metadata, waveform, self.device,
                                return_char_alignments=False)
        if self.diarize:
            from whisperx.diarize import DiarizationPipeline
            turns = DiarizationPipeline(token=self.hf_token, device=self.device)(waveform)
            aligned = self.wx.assign_word_speakers(turns, aligned)
        return [{"start": s["start"], "end": s["end"], "text": s["text"],
                 "speaker": s.get("speaker"), "words": s.get("words", []),
                 "alignment": self.alignment, "speaker_status": "diarization_hypothesis"}
                for s in aligned["segments"] if s.get("text", "").strip()]
