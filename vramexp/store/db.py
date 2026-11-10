"""Storico su SQLite: campioni, processi visti, incidenti.

Il servizio scrive un campione al secondo per ogni processo che usa VRAM. Su una
macchina con 15 processi attivi sono ~1,3 milioni di righe al giorno: senza una
strategia di riduzione il file crescerebbe di gigabyte a settimana. La soluzione
e' il downsampling progressivo in `retention.py` — piena risoluzione per l'ultima
ora, poi 10 s, poi 60 s.

Scelte tecniche:
  * WAL, cosi' la GUI puo' leggere mentre il servizio scrive senza bloccarsi;
  * `synchronous = NORMAL`, perche' perdere l'ultimo secondo di telemetria in un
    crash e' irrilevante, mentre un fsync per campione non lo e';
  * i processi sono in tabella separata dai campioni: nome, cmdline e contenuto
    non cambiano e ripeterli un milione di volte sarebbe uno spreco.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..core.config import DB_PATH
from ..core.models import Incident, Snapshot

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS gpu_samples (
    ts              REAL    NOT NULL,
    gpu_index       INTEGER NOT NULL,
    total_bytes     INTEGER NOT NULL,
    used_bytes      INTEGER NOT NULL,
    free_bytes      INTEGER NOT NULL,
    committed_bytes INTEGER,
    shared_bytes    INTEGER,
    util_gpu        INTEGER,
    util_mem        INTEGER,
    temperature_c   INTEGER,
    power_w         REAL,
    clock_sm_mhz    INTEGER,
    balloon_bytes   INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_gpu_ts ON gpu_samples(ts);

CREATE TABLE IF NOT EXISTS proc_samples (
    ts              REAL    NOT NULL,
    pid             INTEGER NOT NULL,
    gpu_index       INTEGER NOT NULL,
    dedicated_bytes INTEGER NOT NULL,
    shared_bytes    INTEGER NOT NULL,
    committed_bytes INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_proc_ts  ON proc_samples(ts);
CREATE INDEX IF NOT EXISTS ix_proc_pid ON proc_samples(pid, ts);

CREATE TABLE IF NOT EXISTS processes (
    pid           INTEGER NOT NULL,
    create_time   REAL    NOT NULL,
    name          TEXT,
    exe           TEXT,
    cmdline       TEXT,
    username      TEXT,
    publisher     TEXT,
    content_json  TEXT,
    first_seen    REAL,
    last_seen     REAL,
    peak_bytes    INTEGER DEFAULT 0,
    PRIMARY KEY (pid, create_time)
);
CREATE INDEX IF NOT EXISTS ix_processes_last ON processes(last_seen);

CREATE TABLE IF NOT EXISTS incidents (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                   REAL NOT NULL,
    kind                 TEXT NOT NULL,
    pid                  INTEGER,
    process_name         TEXT,
    content_label        TEXT,
    gpu_index            INTEGER,
    peak_dedicated_bytes INTEGER,
    gpu_used_bytes       INTEGER,
    gpu_total_bytes      INTEGER,
    detail               TEXT,
    history_json         TEXT
);
CREATE INDEX IF NOT EXISTS ix_incidents_ts ON incidents(ts);

CREATE TABLE IF NOT EXISTS model_observations (
    model_key       TEXT NOT NULL,
    ts              REAL NOT NULL,
    observed_bytes  INTEGER NOT NULL,
    context_length  INTEGER,
    quantization    TEXT,
    backend         TEXT,
    PRIMARY KEY (model_key, ts)
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class HistoryStore:
    """Accesso allo storico. Thread-safe tramite un lock esplicito.

    Si usa un'unica connessione con `check_same_thread=False` e un lock invece di
    una connessione per thread: la frequenza di scrittura e' bassa (1 Hz) e il
    lock e' molto piu' semplice da ragionare rispetto a un pool.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path or DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # -- scrittura ---------------------------------------------------------
    def write_snapshot(self, snap: Snapshot) -> None:
        gpu_rows = [
            (snap.ts, g.gpu_index, g.total_bytes, g.used_bytes, g.free_bytes,
             g.committed_bytes, g.shared_used_bytes, g.util_gpu_pct, g.util_memory_pct,
             g.temperature_c, g.power_w, g.clock_sm_mhz, snap.balloon_bytes)
            for g in snap.gpus
        ]
        proc_rows = [
            (snap.ts, e.sample.pid, e.sample.gpu_index, e.sample.dedicated_bytes,
             e.sample.shared_bytes, e.sample.committed_bytes)
            for e in snap.processes
        ]
        ident_rows = [
            (e.info.pid, e.info.create_time or 0.0, e.info.name, e.info.exe,
             e.info.cmdline, e.info.username, e.info.publisher,
             json.dumps(e.info.content.to_dict(), ensure_ascii=False),
             snap.ts, snap.ts, e.sample.dedicated_bytes)
            for e in snap.processes
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT INTO gpu_samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", gpu_rows)
            self._conn.executemany(
                "INSERT INTO proc_samples VALUES (?,?,?,?,?,?)", proc_rows)
            # `peak_bytes` conserva il massimo storico: e' il dato che serve
            # nell'autopsia, quando il processo non c'e' piu' per essere misurato.
            self._conn.executemany(
                """INSERT INTO processes
                     (pid, create_time, name, exe, cmdline, username, publisher,
                      content_json, first_seen, last_seen, peak_bytes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(pid, create_time) DO UPDATE SET
                     last_seen    = excluded.last_seen,
                     content_json = excluded.content_json,
                     peak_bytes   = MAX(processes.peak_bytes, excluded.peak_bytes)""",
                ident_rows)
            self._conn.commit()

    def add_incident(self, inc: Incident) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO incidents
                     (ts, kind, pid, process_name, content_label, gpu_index,
                      peak_dedicated_bytes, gpu_used_bytes, gpu_total_bytes,
                      detail, history_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (inc.ts, inc.kind, inc.pid, inc.process_name, inc.content_label,
                 inc.gpu_index, inc.peak_dedicated_bytes, inc.gpu_used_bytes,
                 inc.gpu_total_bytes, inc.detail, inc.history_json))
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def record_model_observation(self, model_key: str, observed_bytes: int,
                                 context_length: int | None, quantization: str | None,
                                 backend: str | None) -> None:
        """Memoria realmente osservata per un modello.

        Alimenta il planner: dopo qualche caricamento smette di stimare il
        fabbisogno e comincia a saperlo per misura diretta.
        """
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO model_observations
                   VALUES (?,?,?,?,?,?)""",
                (model_key, time.time(), observed_bytes, context_length,
                 quantization, backend))
            self._conn.commit()

    # -- lettura -----------------------------------------------------------
    def gpu_history(self, since: float, until: float | None = None,
                    gpu_index: int = 0) -> list[dict[str, Any]]:
        until = until if until is not None else time.time()
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM gpu_samples
                   WHERE gpu_index = ? AND ts BETWEEN ? AND ? ORDER BY ts""",
                (gpu_index, since, until)).fetchall()
        return [dict(r) for r in rows]

    def proc_history(self, since: float, until: float | None = None,
                     pid: int | None = None, top_n: int = 12) -> list[dict[str, Any]]:
        """Serie temporali per processo.

        Senza `pid` restituisce solo i primi `top_n` consumatori del periodo: un
        grafico impilato con settanta serie sarebbe illeggibile, e il costo di
        trasferirle tutte alla GUI non sarebbe ripagato.
        """
        until = until if until is not None else time.time()
        with self._lock:
            if pid is not None:
                rows = self._conn.execute(
                    """SELECT ts, pid, dedicated_bytes, shared_bytes
                       FROM proc_samples WHERE pid = ? AND ts BETWEEN ? AND ?
                       ORDER BY ts""", (pid, since, until)).fetchall()
                return [dict(r) for r in rows]

            top = self._conn.execute(
                """SELECT pid, MAX(dedicated_bytes) AS peak FROM proc_samples
                   WHERE ts BETWEEN ? AND ? GROUP BY pid
                   ORDER BY peak DESC LIMIT ?""", (since, until, top_n)).fetchall()
            pids = [int(r["pid"]) for r in top]
            if not pids:
                return []
            placeholders = ",".join("?" * len(pids))
            rows = self._conn.execute(
                f"""SELECT ts, pid, dedicated_bytes, shared_bytes FROM proc_samples
                    WHERE ts BETWEEN ? AND ? AND pid IN ({placeholders})
                    ORDER BY ts""", (since, until, *pids)).fetchall()
        return [dict(r) for r in rows]

    def process_names(self, pids: Iterable[int]) -> dict[int, dict[str, Any]]:
        pids = list(pids)
        if not pids:
            return {}
        placeholders = ",".join("?" * len(pids))
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT pid, name, exe, content_json, peak_bytes, last_seen
                    FROM processes WHERE pid IN ({placeholders})
                    ORDER BY last_seen DESC""", pids).fetchall()
        out: dict[int, dict[str, Any]] = {}
        for r in rows:
            if r["pid"] in out:
                continue   # il piu' recente vince, in caso di PID riciclato
            d = dict(r)
            try:
                d["content"] = json.loads(d.pop("content_json") or "{}")
            except json.JSONDecodeError:
                d["content"] = {}
            out[int(r["pid"])] = d
        return out

    # Colonne della lista: si esclude `history_json`, che pesa qualche kilobyte
    # per incidente e trasformerebbe l'elenco in un trasferimento da megabyte.
    # Lo storico completo si ottiene con `incident(id)`, quando serve davvero.
    _INCIDENT_COLS = ("id, ts, kind, pid, process_name, content_label, gpu_index, "
                      "peak_dedicated_bytes, gpu_used_bytes, gpu_total_bytes, detail")

    def incidents(self, limit: int = 50, since: float | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if since is not None:
                rows = self._conn.execute(
                    f"SELECT {self._INCIDENT_COLS} FROM incidents "
                    "WHERE ts >= ? ORDER BY ts DESC LIMIT ?",
                    (since, limit)).fetchall()
            else:
                rows = self._conn.execute(
                    f"SELECT {self._INCIDENT_COLS} FROM incidents "
                    "ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def incident(self, incident_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
        return dict(row) if row else None

    def model_observations(self, model_key: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM model_observations WHERE model_key = ?
                   ORDER BY ts DESC LIMIT 20""", (model_key,)).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            gpu_n = self._conn.execute("SELECT COUNT(*) FROM gpu_samples").fetchone()[0]
            proc_n = self._conn.execute("SELECT COUNT(*) FROM proc_samples").fetchone()[0]
            inc_n = self._conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
            oldest = self._conn.execute("SELECT MIN(ts) FROM gpu_samples").fetchone()[0]
        size = self.path.stat().st_size if self.path.exists() else 0
        return {"gpu_samples": gpu_n, "proc_samples": proc_n, "incidents": inc_n,
                "oldest_ts": oldest, "db_bytes": size, "path": str(self.path)}

    def execute(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
