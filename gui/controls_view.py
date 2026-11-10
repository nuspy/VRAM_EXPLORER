"""Controlli globali: riserva di VRAM e policy di fallback sulla RAM.

Sono le due manopole che agiscono sull'intero sistema anziche' su un processo, e
hanno entrambe implicazioni che vanno spiegate **prima** dell'uso, non scoperte
dopo: la riserva sottrae memoria in modo solo parzialmente garantito, e la policy
di fallback cambia il comportamento del driver per tutte le applicazioni (o per
quelle scelte) e richiede l'elevazione.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from vramexp.core.units import human_bytes, parse_size

from . import theme


class ControlsView(QWidget):
    def __init__(self, client, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.client = client
        self._total_bytes = 32 * 1024 ** 3
        self._build()
        self.refresh()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(14)

        title = QLabel("Controlli della VRAM")
        title.setObjectName("Title")
        layout.addWidget(title)

        layout.addWidget(self._build_reserve())
        layout.addWidget(self._build_policy())
        layout.addStretch(1)

    # -- riserva -----------------------------------------------------------
    def _build_reserve(self) -> QWidget:
        box = QGroupBox("Riserva di VRAM")
        layout = QVBoxLayout(box)
        layout.setSpacing(10)

        explain = QLabel(
            "Trattiene una quantità di VRAM occupandola, così altri processi non "
            "possono prendersela mentre non guardi.")
        explain.setObjectName("Dim")
        explain.setWordWrap(True)
        layout.addWidget(explain)

        row = QHBoxLayout()
        self.reserve_slider = QSlider(Qt.Orientation.Horizontal)
        self.reserve_slider.setMinimum(0)
        self.reserve_slider.setMaximum(32)
        self.reserve_slider.setTickInterval(4)
        self.reserve_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.reserve_slider.valueChanged.connect(self._on_slider)
        row.addWidget(self.reserve_slider, 1)

        self.reserve_value = QLabel("0 GB")
        self.reserve_value.setMinimumWidth(74)
        self.reserve_value.setAlignment(Qt.AlignmentFlag.AlignRight
                                        | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.reserve_value)

        apply_button = QPushButton("Applica")
        apply_button.setObjectName("Primary")
        apply_button.clicked.connect(self._apply_reserve)
        row.addWidget(apply_button)

        release_button = QPushButton("Rilascia")
        release_button.clicked.connect(lambda: self._set_reserve(0))
        row.addWidget(release_button)
        layout.addLayout(row)

        self.reserve_status = QLabel()
        self.reserve_status.setObjectName("Faint")
        self.reserve_status.setWordWrap(True)
        layout.addWidget(self.reserve_status)

        caveat = QLabel(
            "<b>Limite da conoscere.</b> La riserva è efficace contro i programmi che "
            "usano CUDA, che troveranno meno memoria libera e si adatteranno. Non è "
            "però un blocco hardware: il compositore di Windows e le applicazioni "
            "grafiche possono comunque ottenere memoria facendo sfrattare pagine. "
            "Il processo che detiene la riserva consuma inoltre alcune centinaia di "
            "megabyte per il proprio contesto CUDA.")
        caveat.setWordWrap(True)
        caveat.setTextFormat(Qt.TextFormat.RichText)
        caveat.setStyleSheet(
            f"color:{theme.TEXT_DIM}; background:{theme.BG_ELEVATED}; "
            f"border-left:3px solid {theme.WARN}; padding:9px; border-radius:5px;")
        layout.addWidget(caveat)
        return box

    def _on_slider(self, value: int) -> None:
        self.reserve_value.setText(f"{value} GB")

    def _apply_reserve(self) -> None:
        self._set_reserve(self.reserve_slider.value() * 1024 ** 3)

    def _set_reserve(self, size_bytes: int) -> None:
        try:
            state = self.client.post("/reserve", {"size": size_bytes})
        except Exception as e:
            QMessageBox.warning(self, "Riserva", f"Operazione fallita: {e}")
            return
        self._show_reserve_state(state)

    def _show_reserve_state(self, state: dict) -> None:
        allocated = state.get("allocated_bytes", 0)
        overhead = state.get("context_overhead_bytes", 0)
        parts = [f"Riserva attiva: {human_bytes(allocated)}"]
        if overhead:
            parts.append(f"più {human_bytes(overhead)} di contesto CUDA "
                         f"(totale sottratto: {human_bytes(allocated + overhead)})")
        if state.get("error"):
            parts.append(f"⚠ {state['error']}")
        self.reserve_status.setText(" · ".join(parts))
        self.reserve_slider.blockSignals(True)
        self.reserve_slider.setValue(int(allocated // 1024 ** 3))
        self.reserve_slider.blockSignals(False)
        self.reserve_value.setText(f"{allocated // 1024 ** 3} GB")

    # -- policy ------------------------------------------------------------
    def _build_policy(self) -> QWidget:
        box = QGroupBox("Comportamento quando la VRAM finisce")
        layout = QVBoxLayout(box)
        layout.setSpacing(10)

        explain = QLabel(
            "Quando la VRAM si esaurisce, il driver NVIDIA può continuare allocando "
            "nella RAM di sistema. È silenzioso ma può rendere il programma da 10 a "
            "100 volte più lento. L'alternativa è farlo fallire subito con un errore "
            "di memoria insufficiente.")
        explain.setObjectName("Dim")
        explain.setWordWrap(True)
        layout.addWidget(explain)

        grid = QGridLayout()
        grid.addWidget(QLabel("Ambito:"), 0, 0)
        self.scope_combo = QComboBox()
        self.scope_combo.addItem("Tutto il sistema", None)
        for exe in ("python.exe", "pythonw.exe", "llama-server.exe", "ollama.exe"):
            self.scope_combo.addItem(f"Solo {exe}", exe)
        self.scope_combo.setEditable(False)
        self.scope_combo.currentIndexChanged.connect(self._load_policy)
        grid.addWidget(self.scope_combo, 0, 1)

        grid.addWidget(QLabel("Altro eseguibile:"), 1, 0)
        self.custom_exe = QLineEdit()
        self.custom_exe.setPlaceholderText("nome o percorso di un .exe (facoltativo)")
        grid.addWidget(self.custom_exe, 1, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        self.policy_status = QLabel()
        self.policy_status.setWordWrap(True)
        self.policy_status.setTextFormat(Qt.TextFormat.RichText)
        self.policy_status.setStyleSheet(
            f"background:{theme.BG_ELEVATED}; padding:11px; border-radius:6px;")
        layout.addWidget(self.policy_status)

        buttons = QHBoxLayout()
        allow_button = QPushButton("Consenti l'uso della RAM")
        allow_button.clicked.connect(lambda: self._set_policy(0))
        buttons.addWidget(allow_button)

        fail_button = QPushButton("Fai fallire l'allocazione")
        fail_button.setObjectName("Primary")
        fail_button.clicked.connect(lambda: self._set_policy(1))
        buttons.addWidget(fail_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        note = QLabel(
            "La modifica scrive nelle impostazioni del driver e richiede "
            "l'elevazione: comparirà un avviso di Controllo account utente.")
        note.setObjectName("Faint")
        note.setWordWrap(True)
        layout.addWidget(note)
        return box

    def _current_exe(self) -> str | None:
        custom = self.custom_exe.text().strip()
        if custom:
            return custom
        return self.scope_combo.currentData()

    def _load_policy(self) -> None:
        exe = self._current_exe()
        try:
            state = self.client.get("/policy", exe=exe or "")
        except Exception as e:
            self.policy_status.setText(f"Stato non leggibile: {e}")
            return
        color = theme.WARN if state.get("effective_value") else theme.TEXT
        default_note = ("<br><i>Mai modificata: vale il comportamento predefinito "
                        "del driver.</i>" if state.get("is_default") else "")
        self.policy_status.setText(
            f"<b style='color:{color}'>{state.get('label')}</b><br>"
            f"<span style='color:{theme.TEXT_DIM}'>{state.get('explanation')}</span>"
            f"{default_note}")

    def _set_policy(self, value: int) -> None:
        exe = self._current_exe()
        scope = f"per {exe}" if exe else "per tutto il sistema"
        if value == 1:
            confirm = QMessageBox.question(
                self, "Conferma",
                f"Impostare il fallimento immediato {scope}?\n\n"
                "Le applicazioni che superano la VRAM disponibile riceveranno un "
                "errore invece di rallentare. È di solito preferibile per training "
                "e inferenza, meno per i giochi.",
                QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok,
                QMessageBox.StandardButton.Ok)
            if confirm != QMessageBox.StandardButton.Ok:
                return
        try:
            result = self.client.post("/policy", {"value": value, "exe": exe})
        except Exception as e:
            QMessageBox.warning(self, "Policy", f"Operazione fallita: {e}")
            return
        if result.get("ok"):
            QMessageBox.information(self, "Policy", str(result.get("message")))
        else:
            QMessageBox.warning(self, "Policy non modificata",
                                str(result.get("message")))
        self._load_policy()

    # -- aggiornamento -----------------------------------------------------
    def refresh(self) -> None:
        try:
            status = self.client.get("/status")
            gpus = status.get("gpus") or []
            if gpus:
                self._total_bytes = gpus[0].get("total_bytes", self._total_bytes)
                self.reserve_slider.setMaximum(max(1, int(self._total_bytes // 1024 ** 3)))
            self._show_reserve_state(self.client.get("/reserve"))
        except Exception as e:
            self.reserve_status.setText(f"stato non disponibile: {e}")
        self._load_policy()
