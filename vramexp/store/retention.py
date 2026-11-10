"""Downsampling progressivo dello storico.

Un campione al secondo per processo genera ~1,3 milioni di righe al giorno. La
risoluzione al secondo serve pero' solo per il passato recente: per capire cosa
sia successo tre settimane fa basta un punto al minuto.

La riduzione avviene in tre fasce (piena / 10 s / 60 s) tenendo, per ogni
finestra, il campione con il **valore massimo** invece del primo o della media.
La ragione e' che questo storico serve a indagare i picchi: una media livellerebbe
proprio il momento in cui la VRAM si e' riempita, cioe' l'unica cosa che si vuole
poter vedere a posteriori.
"""

from __future__ import annotations

import logging
import time

from ..core.config import Config
from .db import HistoryStore

log = logging.getLogger("vramexp.retention")


class RetentionManager:
    def __init__(self, store: HistoryStore, cfg: Config | None = None) -> None:
        self.store = store
        self.cfg = cfg or Config()
        self._last_run = 0.0
        self._last_vacuum = 0.0

    def maybe_run(self, interval_s: float = 300.0) -> bool:
        """Esegue la manutenzione se e' passato abbastanza tempo. True se ha agito."""
        now = time.time()
        if (now - self._last_run) < interval_s:
            return False
        self._last_run = now
        try:
            self.run()
        except Exception as e:
            log.warning("manutenzione storico fallita: %s", e)
            return False
        return True

    def run(self) -> dict[str, int]:
        now = time.time()
        full_edge = now - self.cfg.retain_full_hours * 3600
        medium_edge = now - self.cfg.retain_medium_hours * 3600
        coarse_edge = now - self.cfg.retain_coarse_days * 86400

        removed = {
            "gpu_medium": self._downsample("gpu_samples", medium_edge, full_edge, 10,
                                           ("gpu_index",), "used_bytes"),
            "gpu_coarse": self._downsample("gpu_samples", coarse_edge, medium_edge, 60,
                                           ("gpu_index",), "used_bytes"),
            "proc_medium": self._downsample("proc_samples", medium_edge, full_edge, 10,
                                            ("pid", "gpu_index"), "dedicated_bytes"),
            "proc_coarse": self._downsample("proc_samples", coarse_edge, medium_edge, 60,
                                            ("pid", "gpu_index"), "dedicated_bytes"),
            "gpu_expired": self._delete_older("gpu_samples", coarse_edge),
            "proc_expired": self._delete_older("proc_samples", coarse_edge),
            "processes_expired": self._prune_processes(coarse_edge),
        }
        self.store.commit()

        if (now - self._last_vacuum) > self.cfg.vacuum_interval_h * 3600:
            # Le cancellazioni liberano pagine ma non restituiscono spazio al
            # filesystem finche' non si compatta.
            self.store.execute("VACUUM")
            self._last_vacuum = now
            removed["vacuum"] = 1

        total = sum(v for k, v in removed.items() if k != "vacuum")
        if total:
            log.info("manutenzione storico: %d righe rimosse (%s)", total, removed)
        return removed

    def _downsample(self, table: str, since: float, until: float, bucket_s: int,
                    keys: tuple[str, ...], value_col: str) -> int:
        """Riduce a un campione per bucket, conservando quello di valore massimo.

        `rowid` identifica la riga sopravvissuta: SQLite lo espone su ogni tabella
        senza bisogno di una chiave primaria esplicita, il che tiene i campioni
        leggeri (nessun indice aggiuntivo da mantenere in scrittura).
        """
        if until <= since:
            return 0
        group = ", ".join(keys)
        sql = f"""
            DELETE FROM {table}
            WHERE ts >= :since AND ts < :until
              AND rowid NOT IN (
                  SELECT rowid FROM (
                      SELECT rowid,
                             ROW_NUMBER() OVER (
                                 PARTITION BY {group}, CAST(ts / :bucket AS INTEGER)
                                 ORDER BY {value_col} DESC, ts ASC
                             ) AS rn
                      FROM {table}
                      WHERE ts >= :since AND ts < :until
                  ) WHERE rn = 1
              )
        """
        before = self._count(table, since, until)
        self.store.execute(sql, {"since": since, "until": until, "bucket": bucket_s})
        after = self._count(table, since, until)
        return max(0, before - after)

    def _count(self, table: str, since: float, until: float) -> int:
        rows = self.store.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE ts >= ? AND ts < ?", (since, until))
        return int(rows[0]["n"]) if rows else 0

    def _delete_older(self, table: str, edge: float) -> int:
        before = self.store.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE ts < ?", (edge,))
        n = int(before[0]["n"]) if before else 0
        if n:
            self.store.execute(f"DELETE FROM {table} WHERE ts < ?", (edge,))
        return n

    def _prune_processes(self, edge: float) -> int:
        """Rimuove le identita' di processi non piu' visti e senza campioni.

        Si conservano quelli citati da un incidente: sono la documentazione di
        cosa stesse girando quando qualcosa e' andato storto, ed e' esattamente
        il dato che si vuole ritrovare mesi dopo.
        """
        rows = self.store.execute(
            """SELECT COUNT(*) AS n FROM processes
               WHERE last_seen < ? AND pid NOT IN (SELECT pid FROM incidents)""", (edge,))
        n = int(rows[0]["n"]) if rows else 0
        if n:
            self.store.execute(
                """DELETE FROM processes
                   WHERE last_seen < ? AND pid NOT IN (SELECT pid FROM incidents)""",
                (edge,))
        return n
