"""Timeline storica e incidenti: l'autopsia di cosa e' successo.

Il grafico impilato per processo risponde alla domanda che nessun monitor
istantaneo puo' affrontare: "chi ha riempito la VRAM alle tre di notte". Accanto
c'e' l'elenco degli incidenti, e selezionandone uno il grafico si posiziona sul
momento in cui e' avvenuto, con lo stato dei processi in quel preciso istante.
"""

from __future__ import annotations

import time

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QSplitter, QTextEdit, QVBoxLayout, QWidget,
)

from vramexp.core.units import human_bytes

from . import theme

RANGES = [("15 minuti", 15), ("1 ora", 60), ("6 ore", 360),
          ("24 ore", 1440), ("7 giorni", 10080)]

INCIDENT_LABELS = {
    "oom_suspect": ("Sospetto esaurimento memoria", theme.CRIT),
    "process_gone": ("Processo terminato", theme.TEXT_DIM),
    "spillover": ("Spostamento su RAM", theme.SPILL),
    "pressure_warn": ("Memoria quasi esaurita", theme.WARN),
    "pressure_critical": ("Memoria critica", theme.CRIT),
}


class TimelineView(QWidget):
    def __init__(self, client, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.client = client
        self._curves: list = []
        self._build()
        self.refresh()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("Storico e incidenti")
        title.setObjectName("Title")
        header.addWidget(title)
        header.addStretch(1)

        self.range_combo = QComboBox()
        for label, _ in RANGES:
            self.range_combo.addItem(label)
        self.range_combo.setCurrentIndex(1)
        self.range_combo.currentIndexChanged.connect(self.refresh)
        header.addWidget(QLabel("Intervallo:"))
        header.addWidget(self.range_combo)

        refresh_button = QPushButton("Aggiorna")
        refresh_button.clicked.connect(self.refresh)
        header.addWidget(refresh_button)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        chart_box = QFrame()
        chart_box.setObjectName("Card")
        chart_layout = QVBoxLayout(chart_box)
        chart_layout.setContentsMargins(10, 10, 10, 10)

        pg.setConfigOptions(antialias=True, background=theme.BG_CARD,
                            foreground=theme.TEXT_DIM)
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.14)
        self.plot.setLabel("left", "VRAM")
        self.plot.getAxis("left").setScale(1 / (1024 ** 3))
        self.plot.setLabel("left", "VRAM (GB)")
        self.plot.setAxisItems({"bottom": pg.DateAxisItem()})
        # Legenda ancorata in alto a destra con fondo opaco: senza di esso, con
        # molte serie finiva a coprire proprio la zona del grafico dove si
        # leggono i picchi.
        self.legend = self.plot.addLegend(offset=(-8, 8), labelTextSize="8pt",
                                          labelTextColor=theme.TEXT_DIM,
                                          brush=pg.mkBrush(22, 27, 34, 225),
                                          pen=pg.mkPen(theme.BORDER))
        self.legend.setColumnCount(2)
        chart_layout.addWidget(self.plot, 1)

        self.chart_note = QLabel()
        self.chart_note.setObjectName("Faint")
        self.chart_note.setWordWrap(True)
        chart_layout.addWidget(self.chart_note)
        splitter.addWidget(chart_box)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(8)

        incidents_title = QLabel("Incidenti registrati")
        incidents_title.setObjectName("SectionTitle")
        side_layout.addWidget(incidents_title)

        self.incident_list = QListWidget()
        self.incident_list.currentItemChanged.connect(self._on_incident_selected)
        side_layout.addWidget(self.incident_list, 1)

        self.incident_detail = QTextEdit()
        self.incident_detail.setReadOnly(True)
        self.incident_detail.setMinimumHeight(190)
        side_layout.addWidget(self.incident_detail)

        splitter.addWidget(side)
        splitter.setSizes([700, 380])
        layout.addWidget(splitter, 1)

    # -- dati ---------------------------------------------------------------
    def refresh(self) -> None:
        minutes = RANGES[self.range_combo.currentIndex()][1]
        self._load_chart(minutes)
        self._load_incidents()

    def _load_chart(self, minutes: int) -> None:
        try:
            # Sei serie sono il limite oltre il quale il grafico diventa
            # illeggibile e la legenda occupa piu' spazio dei dati.
            data = self.client.get("/history/processes", minutes=minutes, top=6)
            gpu = self.client.get("/history/gpu", minutes=minutes)
        except Exception as e:
            self.chart_note.setText(f"storico non disponibile: {e}")
            return

        self.plot.clear()
        if self.legend is not None:
            self.legend.clear()

        samples = data.get("samples") or []
        names = data.get("processes") or {}
        if not samples:
            self.chart_note.setText(
                "Nessun dato per questo intervallo. Lo storico si popola solo mentre "
                "il servizio è in esecuzione.")
            return

        by_pid: dict[int, list[tuple[float, int]]] = {}
        for row in samples:
            by_pid.setdefault(int(row["pid"]), []).append(
                (float(row["ts"]), int(row["dedicated_bytes"])))

        order = sorted(by_pid, key=lambda p: -max(v for _, v in by_pid[p]))
        for index, pid in enumerate(order):
            series = sorted(by_pid[pid])
            xs = [t for t, _ in series]
            ys = [v for _, v in series]
            meta = names.get(str(pid)) or names.get(pid) or {}
            content = meta.get("content") or {}
            label = content.get("label") or meta.get("name") or f"pid {pid}"
            color = theme.series_color(index)
            self.plot.plot(xs, ys, pen=pg.mkPen(color, width=2),
                           name=f"{label[:34]} ({pid})")

        gpu_samples = gpu.get("samples") or []
        if gpu_samples:
            xs = [float(r["ts"]) for r in gpu_samples]
            ys = [int(r["used_bytes"]) for r in gpu_samples]
            self.plot.plot(xs, ys, pen=pg.mkPen(theme.TEXT, width=2,
                                                style=Qt.PenStyle.DashLine),
                           name="totale residente")
            total = gpu_samples[-1].get("total_bytes") or 0
            if total:
                line = pg.InfiniteLine(pos=total, angle=0,
                                       pen=pg.mkPen(theme.CRIT, width=1,
                                                    style=Qt.PenStyle.DotLine))
                self.plot.addItem(line)

        self.chart_note.setText(
            "Le curve mostrano la memoria *committed* per processo: la loro somma può "
            "superare la capacità della GPU. La linea tratteggiata è la memoria "
            "realmente residente, quella rossa la capacità fisica.")

    def _load_incidents(self) -> None:
        self.incident_list.clear()
        try:
            rows = self.client.get("/incidents", limit=80).get("incidents", [])
        except Exception as e:
            self.incident_detail.setPlainText(f"incidenti non disponibili: {e}")
            return
        if not rows:
            self.incident_detail.setPlainText(
                "Nessun incidente registrato.\n\n"
                "Vengono annotati qui: processi scomparsi mentre la VRAM era quasi "
                "piena, passaggi alla RAM di sistema e superamenti delle soglie.")
            return

        for row in rows:
            label, color = INCIDENT_LABELS.get(row["kind"], (row["kind"], theme.TEXT))
            when = time.strftime("%d/%m %H:%M:%S", time.localtime(row["ts"]))
            who = row.get("process_name") or "sistema"
            item = QListWidgetItem(f"{when}  {label}\n    {who}")
            item.setData(Qt.ItemDataRole.UserRole, row)
            item.setForeground(_brush(color))
            self.incident_list.addItem(item)

    def _on_incident_selected(self, current: QListWidgetItem, _previous) -> None:
        if current is None:
            return
        row = current.data(Qt.ItemDataRole.UserRole)
        label, _ = INCIDENT_LABELS.get(row["kind"], (row["kind"], theme.TEXT))
        when = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(row["ts"]))

        lines = [
            f"{label}",
            f"{when}",
            "",
            f"Processo:  {row.get('process_name') or '—'} (pid {row.get('pid')})",
            f"Contenuto: {row.get('content_label') or '—'}",
            f"Picco:     {human_bytes(row.get('peak_dedicated_bytes') or 0)}",
            f"GPU:       {human_bytes(row.get('gpu_used_bytes') or 0)} / "
            f"{human_bytes(row.get('gpu_total_bytes') or 0)}",
            "",
            row.get("detail") or "",
        ]

        # Lo storico dell'incidente non arriva con l'elenco (sarebbero megabyte):
        # si carica solo per quello effettivamente selezionato.
        history = row.get("history_json")
        if not history and row.get("id"):
            try:
                full = self.client.get(f"/incidents/{int(row['id'])}")
                history = (full or {}).get("history_json")
            except Exception:
                history = None
        if history:
            lines.extend(["", "─" * 46,
                          "Stato nei secondi precedenti l'evento:", ""])
            lines.extend(_format_history(history))
        self.incident_detail.setPlainText("\n".join(lines))


def _format_history(history_json: str) -> list[str]:
    """Ricostruisce l'andamento dei principali consumatori attorno all'evento."""
    import json

    try:
        data = json.loads(history_json)
    except (json.JSONDecodeError, TypeError):
        return ["(storico non decodificabile)"]

    frames = data.get("frames") or []
    if not frames:
        return ["(nessun campione conservato)"]

    focus = data.get("focus_pid")
    peaks: dict[int, dict] = {}
    for frame in frames:
        for proc in frame.get("procs", []):
            entry = peaks.setdefault(proc["pid"], {"name": proc.get("label")
                                                   or proc.get("name"), "peak": 0})
            entry["peak"] = max(entry["peak"], proc.get("ded", 0))

    lines = []
    last = frames[-1]
    for gpu in last.get("gpus", []):
        lines.append(f"GPU {gpu['i']}: {human_bytes(gpu['used'])} / "
                     f"{human_bytes(gpu['total'])} residenti")
    lines.append("")
    lines.append(f"{len(frames)} campioni conservati "
                 f"({frames[-1]['ts'] - frames[0]['ts']:.0f} secondi)")
    lines.append("")
    lines.append("Principali consumatori nel periodo:")
    for pid, entry in sorted(peaks.items(), key=lambda kv: -kv[1]["peak"])[:10]:
        marker = " ←" if pid == focus else ""
        lines.append(f"  {human_bytes(entry['peak']):>10}  "
                     f"{(entry['name'] or '?')[:44]} (pid {pid}){marker}")
    return lines


def _brush(color: str):
    from PySide6.QtGui import QBrush, QColor
    return QBrush(QColor(color))
