"""Pannello di dettaglio del processo selezionato, con le azioni disponibili.

E' il punto in cui l'utente decide di intervenire, quindi la priorita' non e' la
densita' ma la chiarezza: ogni azione dice cosa fa **prima** di essere premuta, e
quelle distruttive chiedono conferma nominando il processo e il contenuto. Un
pulsante "Termina" senza contesto e' un incidente in attesa di accadere.

Quando il processo espone l'agent di introspezione, il pannello mostra anche le
singole allocazioni con la riga di codice che le ha create.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea,
    QTextEdit, QVBoxLayout, QWidget,
)

from vramexp.core.units import human_bytes, pct

from . import theme


class DetailPanel(QWidget):
    action_requested = Signal(int, str, bool)     # pid, kind, confirmed
    attach_requested = Signal(int)
    record_requested = Signal(int)

    def __init__(self, client, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.client = client
        self.pid: int | None = None
        self._data: dict = {}
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        self.title = QLabel("Nessun processo selezionato")
        self.title.setObjectName("Title")
        self.title.setWordWrap(True)
        layout.addWidget(self.title)

        self.subtitle = QLabel("Selezionare un processo dalla mappa o dalla tabella.")
        self.subtitle.setObjectName("Subtitle")
        self.subtitle.setWordWrap(True)
        layout.addWidget(self.subtitle)

        self.badges = QLabel()
        self.badges.setObjectName("Dim")
        self.badges.setWordWrap(True)
        self.badges.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.badges)

        separator = QFrame()
        separator.setObjectName("Separator")
        separator.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(separator)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(0, 0, 6, 0)
        self.content_layout.setSpacing(10)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setTextFormat(Qt.TextFormat.RichText)
        self.info_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.content_layout.addWidget(self.info_label)

        self.actions_title = QLabel("Azioni")
        self.actions_title.setObjectName("SectionTitle")
        self.actions_title.setVisible(False)
        self.content_layout.addWidget(self.actions_title)

        self.actions_box = QWidget()
        self.actions_layout = QVBoxLayout(self.actions_box)
        self.actions_layout.setContentsMargins(0, 0, 0, 0)
        self.actions_layout.setSpacing(8)
        self.content_layout.addWidget(self.actions_box)

        self.agent_title = QLabel("Allocazioni (agent attivo)")
        self.agent_title.setObjectName("SectionTitle")
        self.agent_title.setVisible(False)
        self.content_layout.addWidget(self.agent_title)

        self.agent_view = QTextEdit()
        self.agent_view.setReadOnly(True)
        self.agent_view.setMinimumHeight(180)
        self.agent_view.setVisible(False)
        self.content_layout.addWidget(self.agent_view)

        self.content_layout.addStretch(1)

    # -- popolamento -------------------------------------------------------
    def show_process(self, pid: int) -> None:
        self.pid = pid
        try:
            data = self.client.get(f"/process/{pid}")
        except Exception as e:
            self.title.setText(f"pid {pid}")
            self.subtitle.setText(f"impossibile leggere i dettagli: {e}")
            return
        if not data:
            self.title.setText(f"pid {pid}")
            self.subtitle.setText("il processo non sta più usando la GPU")
            return

        self._data = data
        content = data.get("content") or {}

        self.title.setText(content.get("label") or data.get("name") or f"pid {pid}")
        self.subtitle.setText(content.get("detail") or "")

        badges = [_badge(f"pid {pid}", theme.TEXT_DIM)]
        conf = content.get("confidence", "unknown")
        conf_color = {"confirmed": theme.OK, "parsed": theme.ACCENT,
                      "inferred": theme.WARN}.get(conf, theme.TEXT_FAINT)
        badges.append(_badge(_confidence_text(conf), conf_color))
        if content.get("backend"):
            badges.append(_badge(content["backend"], theme.ACCENT))
        if data.get("has_agent") or data.get("agent"):
            badges.append(_badge("agent attivo", theme.SPILL))
        if data.get("shared_bytes", 0) > 64 * 1024 * 1024:
            badges.append(_badge("▲ usa la RAM di sistema", theme.SPILL))
        self.badges.setText(" ".join(badges))

        self.info_label.setText(self._info_html(data, content))
        self._build_actions(data)
        self._build_agent_view(pid, data)

    def _info_html(self, data: dict, content: dict) -> str:
        rows: list[tuple[str, str]] = [
            ("Memoria dedicata", human_bytes(data.get("dedicated_bytes", 0))),
        ]
        shared = data.get("shared_bytes", 0)
        if shared:
            rows.append(("Su RAM di sistema",
                         f"<span style='color:{theme.SPILL}'>{human_bytes(shared)}</span>"))
        rows.append(("Committed", human_bytes(data.get("committed_bytes", 0))))

        engines = data.get("engine_pct") or {}
        if engines:
            rows.append(("Motori attivi", ", ".join(
                f"{k} {v:.0f}%" for k, v in sorted(engines.items(), key=lambda kv: -kv[1])
                if v > 0.5) or "inattivo"))

        if content.get("model_name"):
            rows.append(("Modello", content["model_name"]))
        for key, label in (("quantization", "Quantizzazione"),
                           ("parameters", "Parametri"),
                           ("context_length", "Contesto")):
            if content.get(key):
                value = content[key]
                rows.append((label, f"{value:,}".replace(",", " ")
                             if isinstance(value, int) else str(value)))
        if content.get("file_size_bytes"):
            rows.append(("Dimensione file", human_bytes(content["file_size_bytes"])))

        extra = content.get("extra") or {}
        for key, label in (("architecture", "Architettura"), ("n_layers", "Layer"),
                           ("n_kv_heads", "Teste KV"), ("head_dim", "Dimensione testa"),
                           ("gpu_layers", "Layer in GPU")):
            if extra.get(key):
                rows.append((label, str(extra[key])))
        if extra.get("partial_offload"):
            rows.append(("Attenzione",
                         f"<span style='color:{theme.WARN}'>parte dei layer gira sulla "
                         "CPU: la generazione è molto più lenta</span>"))
        if extra.get("advice"):
            rows.append(("Nota", str(extra["advice"])))

        if data.get("exe"):
            rows.append(("Eseguibile", _escape(data["exe"])))
        if data.get("username"):
            rows.append(("Utente", data["username"]))
        if content.get("file_path"):
            rows.append(("File", _escape(content["file_path"])))

        html = ["<table cellspacing='0' cellpadding='4' width='100%'>"]
        for label, value in rows:
            html.append(
                f"<tr><td style='color:{theme.TEXT_DIM}; vertical-align:top; "
                f"white-space:nowrap;'>{label}</td>"
                f"<td style='color:{theme.TEXT};'>{value}</td></tr>")
        html.append("</table>")
        return "".join(html)

    def _build_actions(self, data: dict) -> None:
        while self.actions_layout.count():
            item = self.actions_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        actions = data.get("actions") or []
        self.actions_title.setVisible(bool(actions))
        for action in actions:
            self.actions_layout.addWidget(self._action_widget(action))

        if not data.get("has_agent") and not data.get("agent"):
            name = (data.get("name") or "").lower()
            if "python" in name:
                self.actions_layout.addWidget(self._attach_widget())

    def _action_widget(self, action: dict) -> QWidget:
        box = QFrame()
        box.setObjectName("Card")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(5)

        row = QHBoxLayout()
        button = QPushButton(action["label"])
        button.setMinimumHeight(34)
        if action.get("destructive"):
            button.setObjectName("Danger")
        elif action["kind"] == "backend_api":
            button.setObjectName("Primary")
        button.setEnabled(bool(action.get("available")))
        button.clicked.connect(
            lambda _=False, a=action: self._run_action(a))
        row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)

        description = QLabel(action.get("description", ""))
        description.setObjectName("Faint")
        description.setWordWrap(True)
        layout.addWidget(description)

        if not action.get("available") and action.get("unavailable_reason"):
            reason = QLabel("Non disponibile: " + action["unavailable_reason"])
            reason.setWordWrap(True)
            reason.setStyleSheet(f"color: {theme.WARN}; font-size: 9pt;")
            layout.addWidget(reason)
        return box

    def _attach_widget(self) -> QWidget:
        box = QFrame()
        box.setObjectName("Card")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(5)

        button = QPushButton("Inietta l'agent di introspezione")
        button.setMinimumHeight(34)
        button.clicked.connect(lambda: self.attach_requested.emit(self.pid))
        layout.addWidget(button)

        note = QLabel(
            "Permette di vedere le singole allocazioni con la riga di codice che le "
            "ha create. <b>Può destabilizzare il processo</b>: la via sicura è "
            "riavviarlo con la variabile d'ambiente <code>VRAMEXP_AGENT=1</code>.")
        note.setObjectName("Faint")
        note.setWordWrap(True)
        note.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(note)
        return box

    def _build_agent_view(self, pid: int, data: dict) -> None:
        agent = data.get("agent")
        if not agent:
            self.agent_title.setVisible(False)
            self.agent_view.setVisible(False)
            return

        self.agent_title.setVisible(True)
        self.agent_view.setVisible(True)

        allocated = agent.get("allocated_bytes", 0)
        reserved = agent.get("reserved_bytes", 0)
        frag = agent.get("fragmentation_bytes", 0)
        lines = [
            f"allocato        {human_bytes(allocated):>12}",
            f"riservato       {human_bytes(reserved):>12}",
            f"frammentazione  {human_bytes(frag):>12}   "
            f"({pct(frag, reserved):.0f}% del riservato)",
            f"picco allocato  {human_bytes(agent.get('max_allocated_bytes', 0)):>12}",
            f"retry alloc     {agent.get('num_alloc_retries', 0):>12}",
            f"OOM             {agent.get('num_ooms', 0):>12}",
        ]
        if frag and reserved and frag / reserved > 0.25:
            lines.append("")
            lines.append("La frammentazione supera un quarto della memoria riservata.")
            lines.append("Può aiutare: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

        try:
            snapshot = self.client.get(f"/process/{pid}/allocations", limit=18)
        except Exception:
            snapshot = None
        if snapshot and snapshot.get("ok"):
            lines.append("")
            lines.append(f"{snapshot['block_count']} blocchi attivi, "
                         f"{human_bytes(snapshot['total_bytes'])} in totale")
            lines.append("")
            for block in snapshot.get("blocks", [])[:14]:
                frames = block.get("frames") or []
                origin = frames[0] if frames else "origine ignota"
                lines.append(f"{human_bytes(block['size']):>11}  ← {origin[-76:]}")
            if not any(b.get("frames") for b in snapshot.get("blocks", [])):
                lines.append("")
                lines.append("Nessuno stack trace: attivare la registrazione per")
                lines.append("associare le nuove allocazioni alla riga di codice.")

        self.agent_view.setPlainText("\n".join(lines))

    # -- esecuzione --------------------------------------------------------
    def _run_action(self, action: dict) -> None:
        if self.pid is None:
            return
        confirmed = True
        if action.get("requires_confirm") or action.get("destructive"):
            label = self.title.text()
            box = QMessageBox(self)
            box.setWindowTitle("Conferma azione")
            box.setIcon(QMessageBox.Icon.Warning)
            box.setText(f"{action['label']} — «{label}» (pid {self.pid})?")
            box.setInformativeText(
                action.get("description", "")
                + "\n\nQuesta azione non può essere annullata.")
            box.setStandardButtons(QMessageBox.StandardButton.Cancel
                                   | QMessageBox.StandardButton.Ok)
            box.setDefaultButton(QMessageBox.StandardButton.Cancel)
            if box.exec() != QMessageBox.StandardButton.Ok:
                return
        self.action_requested.emit(self.pid, action["kind"], confirmed)


def _badge(text: str, color: str) -> str:
    return (f"<span style='background:{theme.BG_ELEVATED}; color:{color}; "
            f"padding:3px 8px; border-radius:9px; font-size:9pt;'>{text}</span>")


def _confidence_text(key: str) -> str:
    return {
        "confirmed": "confermato dal backend",
        "parsed": "letto dai file",
        "inferred": "ipotesi dal nome",
        "unknown": "contenuto ignoto",
    }.get(key, key)


def _escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
