from __future__ import annotations

import argparse
import json
import os
import sys
import sqlite3
from pathlib import Path

from .cache import ArtifactCache, atomic_json
from .demo import DemoBackend, DemoEncoder, build_demo
from .download import download, entries
from .evaluation import evaluate
from .pipeline import IndexConfig, Indexer
from .providers import OpenRouterBackend, OpenRouterClient, OpenRouterEncoder
from .search import SearchConfig, SearchEngine
from .store import Store
from .speech import WhisperXTranscriber, WebRTCSpeechDetector


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evidence-grounded video search. Live indexing, searching, and evaluation may call provider APIs.")
    p.add_argument("--data-dir", default="data/index", help="Database and artifact directory")
    p.add_argument("--demo", action="store_true", help="Synthetic test adapters; never valid for real videos")
    p.add_argument("--max-requests", type=int, default=200, help="HTTP attempts, including retries, per command")
    p.add_argument("--retries", type=int, default=1, help="Maximum retries per HTTP request (0–3)")
    p.add_argument("--env-file", default=".env", help="Optional local configuration file; existing environment takes precedence")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="Create synthetic contract fixtures, without API calls")
    d.add_argument("--labels", default=None)
    dl = sub.add_parser("download", help="Parse (never execute) a Code Four download script")
    dl.add_argument("script")
    dl.add_argument("--videos", nargs="+")
    dl.add_argument("--output-dir", default="data/videos")
    dl.add_argument("--list", action="store_true")
    idx = sub.add_parser("index", help="Index a local video; original media is preserved")
    idx.add_argument("video")
    idx.add_argument("--transcript", help="JSON list of aligned segments on original timeline")
    idx.add_argument("--window", type=float, default=20)
    idx.add_argument("--overlap", type=float, default=5)
    idx.add_argument("--max-seconds", type=float)
    idx.add_argument("--skip-ocr", action="store_true")
    idx.add_argument("--ocr-every", type=float, default=10)
    idx.add_argument("--height", type=int, default=720)
    idx.add_argument("--fps", type=int, default=8, help="Transport frame rate; provider may sample more sparsely")
    idx.add_argument("--no-rolling-context", action="store_true", help="Ablate previous-window state")
    idx.add_argument("--no-reconciliation", action="store_true", help="Ablate evidence-linked continuation pass")
    idx.add_argument("--asr", choices=["openrouter", "whisperx"], default="openrouter")
    idx.add_argument("--whisper-model", default="small.en")
    idx.add_argument("--device", default="cpu")
    idx.add_argument("--diarize", action="store_true", help="WhisperX speaker separation; requires HF_TOKEN")
    idx.add_argument("--vad-mode", type=int, choices=range(4), default=2)
    search = sub.add_parser("search", help="Retrieve and verify bounded candidate clips")
    search.add_argument("query")
    search.add_argument("--all", action="store_true", dest="enumerate_all")
    search.add_argument("--offset", type=int, default=0, help="Candidate offset, not result offset")
    search.add_argument("--snapshot", help="Snapshot from previous page; reject pagination across changed index")
    search.add_argument("--unverified", action="store_true", help="Return candidates, never call them verified")
    search.add_argument("--verify-budget", type=int, default=5)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--max-clip-seconds", type=float, default=30)
    search.add_argument("--policy", choices=["lexical", "dense", "structured", "combined"], default="combined")
    search.add_argument("--candidate-budget", type=int, default=30)
    search.add_argument("--no-assessment", action="store_true", help="Ablate the indexed-evidence LLM assessment pass")
    ev = sub.add_parser("evaluate", help="Run equal-budget retrieval ablations on reviewed labels")
    ev.add_argument("labels")
    ev.add_argument("--split", choices=["dev", "test"], default="test")
    ev.add_argument("--candidate-budget", type=int, default=10)
    ev.add_argument("--policies", nargs="+", default=None)
    ev.add_argument("--output", default="reports/evaluation.json")
    sub.add_parser("coverage", help="Inspect processing success, failures, and skipped intervals")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    store = None
    try:
        env = Path(args.env_file)
        if env.exists():
            allowed = {"OPENROUTER_API_KEY", "VIDEO_MOMENT_RETRIEVAL_VIDEO_MODEL",
                       "VIDEO_MOMENT_RETRIEVAL_TEXT_MODEL", "VIDEO_MOMENT_RETRIEVAL_AUDIO_MODEL",
                       "VIDEO_MOMENT_RETRIEVAL_EMBEDDING_MODEL", "VIDEO_SEARCH_VIDEO_MODEL",
                       "VIDEO_SEARCH_TEXT_MODEL", "VIDEO_SEARCH_AUDIO_MODEL",
                       "VIDEO_SEARCH_EMBEDDING_MODEL", "HF_TOKEN"}
            for line in env.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, sep, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if sep and key in allowed and value:
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                        value = value[1:-1]
                    os.environ.setdefault(key, value)
        if args.command == "download":
            result = {"files": [name for name, _ in entries(args.script)]} if args.list else {
                "downloaded": download(args.script, args.output_dir, args.videos or [])}
        else:
            root = Path(args.data_dir).resolve()
            store = Store(root / "index.sqlite")
            cache = ArtifactCache(root / "cache")
            if args.command == "demo":
                result = build_demo(store, Path(args.labels) if args.labels else root / "demo-labels.json")
            elif args.command == "coverage":
                result = store.coverage()
            else:
                synthetic_index = store.get_meta("encoder") == DemoEncoder.identity
                if synthetic_index != args.demo and store.get_meta("encoder"):
                    raise ValueError("Synthetic and live indexes must use separate directories and matching --demo mode")
                if args.demo:
                    if args.command == "index":
                        raise ValueError("Synthetic adapters cannot index real media")
                    backend, encoder, client = DemoBackend(), DemoEncoder(), None
                else:
                    client = OpenRouterClient(max_requests=args.max_requests, retries=args.retries)
                    backend = OpenRouterBackend(client, cache)
                    encoder = OpenRouterEncoder(client, cache)
                if args.command == "index":
                    transcriber = WhisperXTranscriber(args.whisper_model, args.device, args.diarize,
                        os.getenv("HF_TOKEN")) if args.asr == "whisperx" else backend
                    detector = WebRTCSpeechDetector(args.vad_mode)
                    indexer = Indexer(store, cache, root / "artifacts", backend, transcriber, backend, encoder,
                                      progress=lambda message: print(message, file=sys.stderr, flush=True),
                                      speech_detector=detector, reconciler=backend)
                    result = indexer.index(args.video, IndexConfig(args.window, args.overlap, args.height,
                        args.fps, args.ocr_every, args.max_seconds, not args.skip_ocr,
                        not args.no_rolling_context, not args.no_reconciliation), args.transcript)
                else:
                    engine = SearchEngine(store, cache, root / "verification", encoder, backend, backend, args.demo)
                    if args.command == "search":
                        result = engine.search(args.query, SearchConfig(candidate_budget=args.candidate_budget,
                            verify_budget=args.verify_budget, top_k=args.top_k, max_clip_seconds=args.max_clip_seconds,
                            policy=args.policy, verify=not args.unverified, enumerate_all=args.enumerate_all,
                            offset=args.offset, snapshot=args.snapshot, assess=not args.no_assessment))
                    else:
                        result = evaluate(engine, json.loads(Path(args.labels).read_text()),
                            SearchConfig(candidate_budget=args.candidate_budget), args.policies, args.split)
                        atomic_json(Path(args.output), result)
                result["api_usage"] = client.stats() if client else {"http_attempts": 0, "synthetic": True}
        print(json.dumps(result, indent=2, allow_nan=False))
        return 2 if result.get("errors") or result.get("operational_errors") else 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, sqlite3.Error) as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; completed artifacts are cached, pending stages are visible in coverage.", file=sys.stderr)
        return 130
    finally:
        if store:
            store.close()
