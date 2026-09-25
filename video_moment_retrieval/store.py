from __future__ import annotations

import json
import math
import re
import sqlite3
import uuid
from pathlib import Path
from urllib.parse import quote

from .types import Record, interval


def normalize_vector(vector: list[float]) -> list[float]:
    if not vector or not all(math.isfinite(x) for x in vector):
        raise ValueError("Embedding must be nonempty and finite")
    norm = math.sqrt(sum(x*x for x in vector))
    if norm == 0:
        raise ValueError("Zero embedding")
    return [x/norm for x in vector]


class Store:
    def __init__(self, path: str | Path, read_only: bool = False):
        self.path = str(Path(path).resolve())
        if not read_only:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(f"file:{quote(self.path)}?mode=ro", uri=True) if read_only else sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        if read_only:
            return
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS videos(
          id TEXT PRIMARY KEY, path TEXT NOT NULL, duration REAL NOT NULL,
          has_audio INTEGER NOT NULL, metadata TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS records(
          id TEXT PRIMARY KEY, video_id TEXT NOT NULL REFERENCES videos(id),
          start REAL NOT NULL, end REAL NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS record_time ON records(video_id,start,end);
        CREATE VIRTUAL TABLE IF NOT EXISTS lexical USING fts5(record_id UNINDEXED, text);
        CREATE TABLE IF NOT EXISTS facts(
          record_id TEXT NOT NULL REFERENCES records(id) ON DELETE CASCADE,
          field TEXT NOT NULL, value TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS fact_lookup ON facts(field,value);
        CREATE TABLE IF NOT EXISTS vectors(
          record_id TEXT PRIMARY KEY REFERENCES records(id) ON DELETE CASCADE,
          encoder TEXT NOT NULL, dimension INTEGER NOT NULL, vector TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS coverage(
          video_id TEXT NOT NULL REFERENCES videos(id), stage TEXT NOT NULL,
          start REAL NOT NULL, end REAL NOT NULL, status TEXT NOT NULL, detail TEXT NOT NULL,
          PRIMARY KEY(video_id,stage,start,end));
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS index_attempts(
          id TEXT PRIMARY KEY, video_id TEXT, status TEXT NOT NULL, report_path TEXT NOT NULL, staging_path TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS relationships(
          id TEXT PRIMARY KEY, video_id TEXT NOT NULL REFERENCES videos(id),
          kind TEXT NOT NULL, start REAL NOT NULL, end REAL NOT NULL,
          status TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS relationship_lookup ON relationships(video_id,kind,start,end);
        """)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO meta VALUES('index_revision',?)", (uuid.uuid4().hex,))

    def close(self) -> None:
        self.db.close()

    def set_meta(self, key: str, value: str) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, value))

    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def video(self, video_id: str) -> dict:
        row = self.db.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
        if not row:
            raise ValueError(f"Unknown video {video_id}")
        return {**dict(row), "metadata": json.loads(row["metadata"])}

    def add_video(self, video_id: str, path: str, duration: float, has_audio: bool,
                  metadata: dict | None = None) -> None:
        interval(0, duration)
        with self.db:
            self.db.execute("""INSERT INTO videos VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
              path=excluded.path, duration=excluded.duration, has_audio=excluded.has_audio,
              metadata=excluded.metadata""",
              (video_id, path, duration, has_audio, json.dumps(metadata or {})))
            self.db.execute("INSERT OR REPLACE INTO meta VALUES('index_revision',?)", (uuid.uuid4().hex,))

    def replace_records(self, video_id: str, records: list[Record], vectors: list[list[float]] | None,
                        encoder: str) -> None:
        video = self.video(video_id)
        if (vectors is not None and len(records) != len(vectors)) or len({r.id for r in records}) != len(records):
            raise ValueError("Records/vectors count mismatch or duplicate record IDs")
        normalized = [normalize_vector(v) for v in vectors] if vectors is not None else []
        dimensions = {len(v) for v in normalized}
        other = self.db.execute("""SELECT DISTINCT encoder,dimension FROM vectors v JOIN records r
          ON v.record_id=r.id WHERE r.video_id!=?""", (video_id,)).fetchall()
        if len(dimensions) > 1 or any(r["encoder"] != encoder or
                                     (dimensions and r["dimension"] not in dimensions) for r in other):
            raise ValueError("Encoder/dimension changed: rebuild the index in a new data directory")
        for r in records:
            # Records can acquire relationships after construction; validate again before publication.
            Record.from_dict(r.to_dict())
            if r.video_id != video_id:
                raise ValueError("Cross-video record")
            interval(r.start, r.end, video["duration"])
            for e in r.evidence:
                interval(e.start, e.end, video["duration"])
            for link in r.links:
                interval(link["start"], link["end"], video["duration"])
        with self.db:
            self.db.execute("DELETE FROM lexical WHERE record_id IN (SELECT id FROM records WHERE video_id=?)", (video_id,))
            self.db.execute("DELETE FROM records WHERE video_id=?", (video_id,))
            self.db.execute("DELETE FROM relationships WHERE video_id=?", (video_id,))
            for i, record in enumerate(records):
                self.db.execute("INSERT INTO records VALUES(?,?,?,?,?,?)", (record.id, video_id,
                    record.start, record.end, record.kind, json.dumps(record.to_dict())))
                text = record.text + " " + " ".join(record.attributes.values())
                self.db.execute("INSERT INTO lexical VALUES(?,?)", (record.id, text))
                facts = {"kind": record.kind, **record.attributes}
                for field in ("subject", "actor", "recipient", "speaker"):
                    value = getattr(record, field)
                    if value:
                        facts[field] = value
                self.db.executemany("INSERT INTO facts VALUES(?,?,?)",
                    [(record.id, key, str(value).casefold()) for key, value in facts.items()])
                if vectors is not None:
                    vector = normalized[i]
                    self.db.execute("INSERT INTO vectors VALUES(?,?,?,?)", (record.id, encoder,
                        len(vector), json.dumps(vector)))
                for j, link in enumerate(record.links):
                    self.db.execute("INSERT OR REPLACE INTO relationships VALUES(?,?,?,?,?,?,?)",
                        (link.get("id", f"{record.id}:link:{j}"), video_id, link.get("kind", "unspecified"),
                         link["start"], link["end"], link["status"], json.dumps(link)))
            self.db.execute("INSERT OR REPLACE INTO meta VALUES('encoder',?)", (encoder,))
            self.db.execute("INSERT OR REPLACE INTO meta VALUES('index_revision',?)", (uuid.uuid4().hex,))

    def record_attempt(self, id_: str, video_id: str | None, status: str, report: str, staging: str) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO index_attempts VALUES(?,?,?,?,?)", (id_, video_id, status, report, staging))

    def publish_video(self, staging: Store, video_id: str) -> None:
        """Copy a validated staging snapshot in one transaction; readers see old or new."""
        video = staging.db.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
        encoder = staging.get_meta("encoder")
        incoming = staging.db.execute("SELECT DISTINCT dimension FROM vectors").fetchall()
        dimensions = {r[0] for r in incoming}
        other = self.db.execute("SELECT DISTINCT encoder,dimension FROM vectors v JOIN records r ON r.id=v.record_id WHERE r.video_id!=?", (video_id,)).fetchall()
        if any(r['encoder'] != encoder or (dimensions and r['dimension'] not in dimensions) for r in other):
            raise ValueError("Encoder/dimension changed: rebuild the index in a new data directory")
        with self.db:
            self.db.execute("INSERT INTO videos VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET path=excluded.path,duration=excluded.duration,has_audio=excluded.has_audio,metadata=excluded.metadata", tuple(video))
            self.db.execute("DELETE FROM lexical WHERE record_id IN (SELECT id FROM records WHERE video_id=?)", (video_id,))
            for table in ('records', 'relationships', 'coverage'):
                self.db.execute(f"DELETE FROM {table} WHERE video_id=?", (video_id,))
            for table in ('records', 'lexical', 'facts', 'vectors', 'relationships', 'coverage'):
                rows = staging.db.execute(f"SELECT * FROM {table}").fetchall()
                if rows:
                    placeholders = ','.join('?' for _ in rows[0])
                    self.db.executemany(f"INSERT INTO {table} VALUES({placeholders})", [tuple(r) for r in rows])
            self.db.execute("INSERT OR REPLACE INTO meta VALUES('encoder',?)", (encoder,))
            self.db.execute("INSERT OR REPLACE INTO meta VALUES('index_revision',?)", (uuid.uuid4().hex,))

    def mark(self, video_id: str, stage: str, start: float, end: float, status: str, detail: str = "") -> None:
        interval(start, end, self.video(video_id)["duration"])
        if status not in {"complete", "failed", "skipped", "not_applicable", "pending"}:
            raise ValueError("Unknown coverage status")
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO coverage VALUES(?,?,?,?,?,?)",
                            (video_id, stage, start, end, status, detail))

    def clear_coverage(self, video_id: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM coverage WHERE video_id=?", (video_id,))

    def records(self, ids: list[str] | None = None) -> list[Record]:
        if ids is not None:
            return [Record.from_dict(json.loads(row[0])) for id_ in ids
                    if (row := self.db.execute("SELECT payload FROM records WHERE id=?", (id_,)).fetchone())]
        return [Record.from_dict(json.loads(row[0])) for row in
                self.db.execute("SELECT payload FROM records ORDER BY video_id,start,id")]

    def context(self, video_id: str, start: float, end: float, limit: int = 100) -> list[Record]:
        return [Record.from_dict(json.loads(r[0])) for r in self.db.execute(
            "SELECT payload FROM records WHERE video_id=? AND start<? AND end>? ORDER BY start,id LIMIT ?",
            (video_id, end, start, limit))]

    def relationships(self, video_id: str, start: float, end: float) -> list[dict]:
        return [json.loads(r[0]) for r in self.db.execute(
            "SELECT payload FROM relationships WHERE video_id=? AND start<? AND end>? ORDER BY start,id",
            (video_id, end, start))]

    def lexical(self, keywords: list[str], limit: int) -> list[tuple[str, float]]:
        tokens = list(dict.fromkeys(re.findall(r"\w+", " ".join(keywords), re.UNICODE)))[:64]
        if not tokens:
            return []
        query = " OR ".join('"' + t + '"' for t in tokens)
        return [(r[0], -r[1]) for r in self.db.execute(
            "SELECT record_id,bm25(lexical) s FROM lexical WHERE lexical MATCH ? ORDER BY s,record_id LIMIT ?",
            (query, limit))]

    def structured(self, constraints: dict[str, str], limit: int) -> list[tuple[str, float]]:
        # OR generates candidates. No uncertain attribute becomes an exclusion rule.
        counts: dict[str, float] = {}
        for key, value in constraints.items():
            for row in self.db.execute("SELECT DISTINCT record_id FROM facts WHERE field=? AND value=?",
                                       (key, value.casefold())):
                counts[row[0]] = counts.get(row[0], 0) + 1
        return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:limit]

    def semantic(self, vector: list[float], encoder: str, limit: int) -> list[tuple[str, float]]:
        expected = self.get_meta("encoder")
        if expected and expected != encoder:
            raise ValueError(f"Index uses {expected}; query encoder is {encoder}. Rebuild or select matching encoder.")
        q = normalize_vector(vector)
        scored = []
        for row in self.db.execute("SELECT record_id,dimension,vector FROM vectors WHERE encoder=?", (encoder,)):
            if row["dimension"] != len(q):
                raise ValueError("Query/index embedding dimension mismatch")
            value = sum(a*b for a, b in zip(q, json.loads(row["vector"])))
            scored.append((row["record_id"], value))
        return sorted(scored, key=lambda x: (-x[1], x[0]))[:limit]

    def coverage(self) -> dict:
        videos = []
        for v in self.db.execute("SELECT * FROM videos ORDER BY id"):
            rows = [dict(r) for r in self.db.execute("SELECT * FROM coverage WHERE video_id=? ORDER BY stage,start", (v["id"],))]
            stages = {}
            for stage in sorted({r["stage"] for r in rows}):
                spans = sorted((r["start"], r["end"]) for r in rows
                               if r["stage"] == stage and r["status"] == "complete")
                merged: list[list[float]] = []
                for a, b in spans:
                    if merged and a <= merged[-1][1]:
                        merged[-1][1] = max(b, merged[-1][1])
                    else:
                        merged.append([a, b])
                stages[stage] = {"processed_seconds": sum(b-a for a, b in merged),
                                 "duration_seconds": v["duration"],
                                 "statuses": sorted({r["status"] for r in rows if r["stage"] == stage})}
            videos.append({"video_id": v["id"], "path": v["path"], "stages": stages, "intervals": rows})
        # Older databases can still be inspected without a write-side migration.
        has_attempts = self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='index_attempts'").fetchone()
        attempts = [dict(r) for r in self.db.execute("SELECT * FROM index_attempts ORDER BY rowid DESC LIMIT 50")] if has_attempts else []
        return {"videos": videos, "attempts": attempts, "note": "Processing coverage is not semantic retrieval completeness."}
