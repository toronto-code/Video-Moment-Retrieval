"""Local media operations. Subprocess arguments never go through a shell."""
from __future__ import annotations

import array
import json
import math
import statistics
import subprocess
import sys
import wave
from pathlib import Path

from .types import interval


def run(args: list[str], timeout: float = 120) -> bytes:
    try:
        p = subprocess.run(args, capture_output=True, timeout=timeout, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required program not found: {args[0]}") from exc
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{args[0]} exceeded its {timeout}-second timeout") from None
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{args[0]} failed: {exc.stderr.decode(errors='replace')[-1500:]}") from exc
    return p.stdout


def probe(path: str | Path) -> dict:
    data = json.loads(run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]))
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if not video:
        raise ValueError("Input has no video stream")
    duration = float(data["format"]["duration"])
    interval(0, duration)
    return {"duration": duration, "has_audio": any(s["codec_type"] == "audio" for s in data["streams"]),
            "width": video["width"], "height": video["height"]}


def windows(duration: float, size: float, overlap: float) -> list[tuple[float, float]]:
    if size <= 0 or overlap < 0 or overlap >= size or not all(map(math.isfinite, (duration, size, overlap))):
        raise ValueError("Window size must be finite and positive, with 0 <= overlap < size")
    interval(0, duration)
    result = []
    start = 0.0
    while start < duration:
        end = min(start + size, duration)
        # Preserve positive sub-microsecond intervals and the exact duration bound.
        result.append((start, end))
        if end == duration:
            break
        next_start = len(result) * (float(size) - overlap)
        if next_start <= start:
            raise ValueError("Window step is below timestamp precision")
        start = next_start
    return result


def clip(source: str, target: Path, start: float, end: float, height: int = 720, fps: int = 8) -> str:
    interval(start, end)
    if height < 64 or height > 2160 or fps < 1 or fps > 60:
        raise ValueError("Invalid video encoding bounds")
    target.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", source,
         "-t", str(end-start), "-map", "0:v:0", "-map", "0:a:0?",
         "-vf", f"scale=-2:'min({height},ih)',fps={fps}", "-c:v", "libx264", "-preset", "fast",
         "-crf", "25", "-c:a", "aac", "-ac", "1", "-ar", "16000", "-movflags", "+faststart", str(target)])
    return str(target)


def audio(source: str, target: Path, start: float, end: float) -> str:
    interval(start, end)
    target.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", source,
         "-t", str(end-start), "-vn", "-af", f"aresample=async=1:first_pts=0,apad=whole_dur={end-start}",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)])
    return str(target)


def frame(source: str, target: Path, time: float) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-v", "error", "-y", "-ss", str(time), "-i", source,
         "-frames:v", "1", "-update", "1", str(target)])
    if not target.exists() or not target.stat().st_size:
        raise RuntimeError("Frame extraction produced no image")
    return str(target)


def sharpness(path: str) -> float:
    # Ranking heuristic only; preserve multiple crops/readings instead of declaring the sharpest correct.
    width = height = 64
    pixels = run(["ffmpeg", "-v", "error", "-i", path, "-vf", "scale=64:64,format=gray",
                  "-frames:v", "1", "-f", "rawvideo", "-"])
    if len(pixels) != width * height:
        raise ValueError("Unexpected grayscale frame shape")
    lap = [4*pixels[y*width+x] - pixels[y*width+x-1] - pixels[y*width+x+1]
           - pixels[(y-1)*width+x] - pixels[(y+1)*width+x]
           for y in range(1, height-1) for x in range(1, width-1)]
    return statistics.pvariance(lap)


def crop(source: str, target: Path, box: list[float]) -> str:
    if len(box) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in box):
        raise ValueError("Invalid normalized crop")
    left, top, right, bottom = box
    if not (left < right and top < bottom):
        raise ValueError("Empty crop")
    target.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-v", "error", "-y", "-i", source, "-vf",
         f"crop=w=iw*{right-left}:h=ih*{bottom-top}:x=iw*{left}:y=ih*{top}",
         "-frames:v", "1", "-update", "1", str(target)])
    return str(target)


def energy_observations(path: str, speech: list[dict], offset: float = 0,
                        hop: float = 0.25, threshold_db: float = 6) -> list[dict]:
    """Speech regions come from an independent detector, with optional diarization.
    Measurements are in dBFS, not sound pressure level. Unknown speakers share a
    local baseline labeled unreliable. No measurement is promoted to shouting.
    """
    out = []
    history: dict[str, list[tuple[float, float]]] = {}
    with wave.open(path, "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            raise ValueError("DSP expects mono 16-bit PCM")
        rate = wav.getframerate()
        step = max(1, int(rate*hop))
        index = 0
        while raw := wav.readframes(step):
            values = array.array("h", raw)
            if sys.byteorder != "little":
                values.byteswap()
            start = offset + index/rate
            end = start + len(values)/rate
            index += len(values)
            active = [s for s in speech if s["start"] < end and s["end"] > start]
            if not active:
                continue
            speaker = active[0].get("speaker") if len(active) == 1 else None
            key = speaker or "unresolved"
            rms = math.sqrt(sum(v*v for v in values)/len(values)) / 32768
            db = 20*math.log10(max(rms, 1e-8))
            prior = [(t, value) for t, value in history.get(key, []) if t >= start-5]
            history[key] = prior
            baseline = statistics.median([value for _, value in prior]) if prior else db
            delta = db-baseline
            if len(prior) >= 4 and delta >= threshold_db:
                out.append({"start": start, "end": end, "speaker": speaker,
                            "dbfs": round(db, 3), "delta_db": round(delta, 3),
                            "baseline_status": "speaker_labeled" if speaker else "unresolved",
                            "behavior": "raised_voice_candidate"})
            prior.append((start, db))
    return out
