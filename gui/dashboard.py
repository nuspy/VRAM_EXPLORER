"""Vista principale: sintesi curata sopra, dettaglio tecnico denso sotto.

L'organizzazione riflette due modi diversi di usare lo strumento. Nel primo si
guarda per un istante per capire se c'e' un problema: servono pochi elementi
grandi e una treemap che mostri subito chi occupa la GPU. Nel secondo si sta
indagando: servono tutte le colonne, ordinabili.

Il pannello tecnico e' quindi collassabile e ricorda il proprio stato: chi lo
tiene aperto lo ritrova aperto.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout, QHeaderView, QLabel, QPushButton, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from vramexp.core.units import human_bytes, pct

from . import theme
from .widgets import StatTile, Treemap, UsageRing

CONFIDENCE_LABELS = {
    "confirmed": ("confermato", theme.OK),
    "parsed": ("dedotto dai file", theme.ACCENT),
    "inferred": ("ipotesi", theme.WARN),
    "unknown": ("ignoto", theme.TEXT_FAINT),
}

# "Utente" non compare: su una postazione i processi appartengono quasi sempre
# allo stesso utente, e la colonna sottraeva spazio a "Contenuto", che e' la
# ragione per cui questo strumento esiste. Il dato resta nel pannello dettagli.
COLUMNS = [
    ("PID", 58), ("Processo", 140), ("Contenuto", 300), ("Fiducia", 86),
    ("Dedicata", 88), ("Su RAM", 84), ("Committed", 88),
    ("Motore", 104), ("Editore", 118),
]


class _SortableItem(QTableWidgetItem):
    """Cella con testo leggibile e ordinamento sul valore numerico.

    Serve perche' la colonna mostra "9.9 GB" ma deve ordinarsi come 10_630_000_000.
    Non si puo' ottenere lo stesso risultato con `EditRole`: Qt userebbe quel dato
    anche per disegnare la cella, facendo comparire i byte grezzi al posto del
    testo formattato.
    """

    def __init__(self, text: str, sort_value: float) -> None:
        super().__init__(text)
        self._sort_value = sort_value

    def __lt__(self, other: object) -> bool:
        if isinstance(other, _SortableItem):
            return self._sort_value < other._sort_value
        return super().__lt__(other)  # type: ignore[misc]


class Dashboard(QWidget):
    process_selected = Signal(int)
    process_activated = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._snapshot: dict | None = None
        self._show_system = True
        self._selected_gpu_index: int = 0
        self._build()

    # -- costruzione -------------------------------------------------------
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_summary())
        splitter.addWidget(self._build_details())
        splitter.setSizes([420, 340])
        outer.addWidget(splitter, 1)

    def _build_summary(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        top = QHBoxLayout()
        top.setSpacing(12)

        ring_card = QFrame()
        ring_card.setObjectName("Card")
        ring_layout = QVBoxLayout(ring_card)
        ring_layout.setContentsMargins(12, 12, 12, 10)

        gpu_header = QHBoxLayout()
        gpu_header.setSpacing(6)
        self.gpu_combo = QComboBox()
        self.gpu_combo.currentIndexChanged.connect(self._on_gpu_selected)
        self.gpu_combo.setMinimumWidth(140)
        gpu_header.addWidget(self.gpu_combo)
        self.gpu_type_label = QLabel("—")
        self.gpu_type_label.setObjectName("Dim")
        gpu_header.addWidget(self.gpu_type_label)
        ring_layout.addLayout(gpu_header)

        self.gpu_label = QLabel("—")
        self.gpu_label.setObjectName("SectionTitle")
        self.gpu_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ring = UsageRing()
        self.ring_legend = QLabel()
        self.ring_legend.setObjectName("Faint")
        self.ring_legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ring_legend.setWordWrap(True)
        ring_layout.addWidget(self.gpu_label)
        ring_layout.addWidget(self.ring, 1)
        ring_layout.addWidget(self.ring_legend)
        ring_card.setMinimumWidth(250)
        ring_card.setMaximumWidth(310)
        top.addWidget(ring_card)

        tiles_column = QVBoxLayout()
        tiles_column.setSpacing(10)
        tiles_row = QHBoxLayout()
        tiles_row.setSpacing(10)
        self.tile_free = StatTile("Disponibile")
        self.tile_committed = StatTile("Committed da Windows")
        self.tile_spill = StatTile("Su RAM di sistema")
        self.tile_reserve = StatTile("Riserva attiva")
        for tile in (self.tile_free, self.tile_committed, self.tile_spill, self.tile_reserve):
            tiles_row.addWidget(tile)
        tiles_column.addLayout(tiles_row)

        map_card = QFrame()
        map_card.setObjectName("Card")
        map_layout = QVBoxLayout(map_card)
        map_layout.setContentsMargins(12, 10, 12, 12)
        header = QHBoxLayout()
        title = QLabel("Chi occupa la VRAM")
        title.setObjectName("SectionTitle")
        header.addWidget(title)
        header.addStretch(1)
        hint = QLabel("click per selezionare · doppio click per i dettagli")
        hint.setObjectName("Faint")
        header.addWidget(hint)
        map_layout.addLayout(header)
        self.treemap = Treemap()
        self.treemap.item_clicked.connect(self.process_selected)
        self.treemap.item_double_clicked.connect(self.process_activated)
        map_layout.addWidget(self.treemap, 1)
        tiles_column.addWidget(map_card, 1)

        top.addLayout(tiles_column, 1)
        layout.addLayout(top, 1)
        return container

    def _build_details(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QHBoxLayout()
        self.toggle_button = QPushButton("▾  Dettaglio tecnico")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.clicked.connect(self._toggle_details)
        self.toggle_button.setToolTip("Mostra o nasconde la tabella completa")
        header.addWidget(self.toggle_button)

        self.system_check = QCheckBox("Includi i processi di Windows")
        self.system_check.setChecked(True)
        self.system_check.stateChanged.connect(self._on_system_toggle)
        header.addWidget(self.system_check)
        header.addStretch(1)

        self.count_label = QLabel()
        self.count_label.setObjectName("Dim")
        header.addWidget(self.count_label)
        layout.addLayout(header)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([c[0] for c in COLUMNS])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(30)
        header = self.table.horizontalHeader()
        for i, (_, width) in enumerate(COLUMNS):
            self.table.setColumnWidth(i, width)
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
        # "Contenuto" e' la colonna che porta l'informazione utile: prende lo
        # spazio residuo invece di essere schiacciata dalle altre.
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(52)
        self.table.setHorizontalScrollMode(
            QTableWidget.ScrollMode.ScrollPerPixel)
        self.table.itemSelectionChanged.connect(self._on_selection)
        self.table.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.table, 1)
        return container

    # -- aggiornamento -----------------------------------------------------
    def update_snapshot(self, snapshot: dict) -> None:
        self._snapshot = snapshot
        gpus = snapshot.get("gpus") or []
        gpu_infos = snapshot.get("gpu_info") or [{}]

        if not gpus:
            return

        # Popola il ComboBox con le GPU disponibili
        self.gpu_combo.blockSignals(True)
        current_index = self.gpu_combo.currentIndex()
        self.gpu_combo.clear()
        for gpu_info in gpu_infos:
            gpu_idx = gpu_info.get("index", 0)
            name = gpu_info.get("name", f"GPU {gpu_idx}")
            # Memoria < 512MB = integrata, altrimenti dedicata
            total_bytes = gpu_info.get("total_bytes", 0)
            is_integrated = total_bytes < 512 * 1024 * 1024 or gpu_idx == -1
            gpu_type = "(iGPU)" if is_integrated else "(Dedicata)"
            self.gpu_combo.addItem(f"{name} {gpu_type}", gpu_idx)

        # Ripristina la selezione precedente se ancora valida
        if 0 <= current_index < self.gpu_combo.count():
            self.gpu_combo.setCurrentIndex(current_index)
        else:
            self._selected_gpu_index = 0
        self.gpu_combo.blockSignals(False)

        # Mostra la GPU selezionata
        gpu = None
        info = None

        # Cerca la GPU selezionata per indice
        for g, i in zip(gpus, gpu_infos):
            if g.get("gpu_index", 0) == self._selected_gpu_index:
                gpu = g
                info = i
                break

        # Se non trovata, usa la prima disponibile
        if gpu is None or info is None:
            gpu = gpus[0] if gpus else {}
            info = gpu_infos[0] if gpu_infos else {}
            self._selected_gpu_index = gpu.get("gpu_index", 0)

        total = gpu.get("total_bytes", 0)
        used = gpu.get("used_bytes", 0)
        committed = gpu.get("committed_bytes") or 0
        shared = gpu.get("shared_used_bytes") or 0
        reserved = snapshot.get("balloon_bytes", 0)

        # Indicatore di GPU integrata vs dedicata
        is_integrated = total < 512 * 1024 * 1024
        self.gpu_type_label.setText("(GPU Integrata)" if is_integrated else "(GPU Dedicata)")

        self.gpu_label.setText(info.get("name", "GPU"))
        self.ring.set_values(used, total, committed, shared, reserved,
                             info.get("name", ""))
        legend = "cerchio interno: residente"
        if committed:
            legend += " · anello esterno: committed"
        if reserved:
            legend += " · anello interno viola: riserva"
        self.ring_legend.setText(legend)

        free = gpu.get("free_bytes", 0)
        free_frac = pct(free, total)
        self.tile_free.set_value(
            human_bytes(free), f"{free_frac:.0f}% della capacità",
            theme.OK if free_frac > 15 else theme.CRIT)

        over = committed > total
        self.tile_committed.set_value(
            human_bytes(committed),
            "supera la capacità fisica: Windows sposta pagine" if over
            else "memoria impegnata dai processi",
            theme.WARN if over else theme.TEXT)

        if shared:
            self.tile_spill.set_value(
                human_bytes(shared), "prestazioni ridotte per i processi coinvolti",
                theme.SPILL)
        else:
            self.tile_spill.set_value("nessuno", "tutto risiede in VRAM", theme.OK)

        if reserved:
            self.tile_reserve.set_value(human_bytes(reserved),
                                        "trattenuta da VRAM Explorer", theme.SPILL)
        else:
            self.tile_reserve.set_value("nessuna", "nessuna VRAM trattenuta",
                                        theme.TEXT_DIM)

        processes = snapshot.get("processes") or []
        # Filtra i processi per la GPU selezionata
        filtered_processes = [p for p in processes if p.get("gpu_index", 0) == self._selected_gpu_index]
        self._update_treemap(filtered_processes)
        self._update_table(filtered_processes)

    def _update_treemap(self, processes: list[dict]) -> None:
        items = []
        for p in processes[:16]:
            content = p.get("content") or {}
            items.append({
                "pid": p.get("pid"),
                "label": content.get("label") or p.get("name") or "?",
                "bytes": p.get("dedicated_bytes", 0),
                "shared": p.get("shared_bytes", 0),
                "kind": content.get("kind"),
            })
        self.treemap.set_items(items)

    def _update_table(self, processes: list[dict]) -> None:
        from vramexp.collect.proc_source import PROTECTED_NAMES

        visible = processes
        if not self._show_system:
            visible = [p for p in processes
                       if (p.get("name") or "").lower() not in PROTECTED_NAMES]

        # Si conserva la selezione durante l'aggiornamento: la tabella si
        # ricostruisce ogni secondo e perdere la riga scelta la renderebbe
        # inutilizzabile.
        selected_pid = self.selected_pid()
        sorting = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(visible))

        for row, p in enumerate(visible):
            content = p.get("content") or {}
            conf_key = content.get("confidence", "unknown")
            conf_label, conf_color = CONFIDENCE_LABELS.get(
                conf_key, (conf_key, theme.TEXT_FAINT))
            shared = p.get("shared_bytes", 0)
            engines = p.get("engine_pct") or {}
            engine_text = " ".join(
                f"{k[:5]} {v:.0f}%"
                for k, v in sorted(engines.items(), key=lambda kv: -kv[1])[:2])

            label = content.get("label") or "—"
            if shared > 64 * 1024 * 1024:
                label = "▲ " + label

            values = [
                (str(p.get("pid", "")), p.get("pid", 0)),
                (p.get("name", ""), None),
                (label, None),
                (conf_label, None),
                (human_bytes(p.get("dedicated_bytes", 0)), p.get("dedicated_bytes", 0)),
                (human_bytes(shared) if shared else "—", shared),
                (human_bytes(p.get("committed_bytes", 0)), p.get("committed_bytes", 0)),
                (engine_text or "—", None),
                (p.get("publisher") or "—", None),
            ]
            for col, (text, sort_value) in enumerate(values):
                item = (_SortableItem(text, sort_value) if sort_value is not None
                        else QTableWidgetItem(text))
                if col in (4, 5, 6):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                if col == 3:
                    item.setForeground(theme_color(conf_color))
                if col == 5 and shared > 64 * 1024 * 1024:
                    item.setForeground(theme_color(theme.SPILL))
                if col == 2 and shared > 64 * 1024 * 1024:
                    item.setToolTip("Questo processo sta usando la RAM di sistema "
                                    "al posto della VRAM: prestazioni molto ridotte.")
                self.table.setItem(row, col, item)

        self.table.setSortingEnabled(sorting)
        self.count_label.setText(f"{len(visible)} processi")
        if selected_pid is not None:
            self._restore_selection(selected_pid)

    def _restore_selection(self, pid: int) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text() == str(pid):
                self.table.blockSignals(True)
                self.table.selectRow(row)
                self.table.blockSignals(False)
                return

    # -- interazione -------------------------------------------------------
    def selected_pid(self) -> int | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 0)
        return int(item.text()) if item and item.text().isdigit() else None

    def _on_selection(self) -> None:
        pid = self.selected_pid()
        if pid is not None:
            self.process_selected.emit(pid)

    def _on_double_click(self, item: QTableWidgetItem) -> None:
        pid_item = self.table.item(item.row(), 0)
        if pid_item and pid_item.text().isdigit():
            self.process_activated.emit(int(pid_item.text()))

    def _toggle_details(self, checked: bool) -> None:
        self.table.setVisible(checked)
        self.system_check.setVisible(checked)
        self.toggle_button.setText(("▾  " if checked else "▸  ") + "Dettaglio tecnico")

    def _on_gpu_selected(self, position: int) -> None:
        # Ottieni il gpu_index memorizzato come "data" nel ComboBox
        gpu_idx = self.gpu_combo.currentData()
        if gpu_idx is not None:
            self._selected_gpu_index = gpu_idx
            if self._snapshot:
                processes = self._snapshot.get("processes") or []
                # Filtra i processi per la GPU selezionata
                filtered_processes = [p for p in processes if p.get("gpu_index", 0) == self._selected_gpu_index]
                gpu_infos = self._snapshot.get("gpu_info") or [{}]
                gpus = self._snapshot.get("gpus") or []

                # Aggiorna le info della GPU
                if 0 <= self._selected_gpu_index < len(gpus):
                    gpu = gpus[self._selected_gpu_index]
                    info = gpu_infos[self._selected_gpu_index] if self._selected_gpu_index < len(gpu_infos) else {}
                    total = gpu.get("total_bytes", 0)
                    is_integrated = total < 512 * 1024 * 1024
                    self.gpu_type_label.setText("(GPU Integrata)" if is_integrated else "(GPU Dedicata)")

                self._update_treemap(filtered_processes)
                self._update_table(filtered_processes)

    def _on_system_toggle(self, state: int) -> None:
        self._show_system = bool(state)
        if self._snapshot:
            processes = self._snapshot.get("processes") or []
            # Filtra i processi per la GPU selezionata
            filtered_processes = [p for p in processes if p.get("gpu_index", 0) == self._selected_gpu_index]
            self._update_table(filtered_processes)

    def select_pid(self, pid: int) -> None:
        self._restore_selection(pid)


def theme_color(hex_color: str):
    from PySide6.QtGui import QBrush, QColor
    return QBrush(QColor(hex_color))
