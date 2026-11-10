"""Pianificatore e profili operativi.

Due strumenti che guardano avanti invece che al presente: "questo modello ci
sta?" e "riconfigura la macchina per questo tipo di lavoro".

Per i profili l'anteprima e' obbligatoria: il pulsante di applicazione resta
disabilitato finche' non si e' visto l'elenco esatto delle azioni. Un click che
scarica modelli e cambia le impostazioni del driver senza mostrarlo prima sarebbe
comodo esattamente quanto pericoloso.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QSpinBox, QSplitter, QTextEdit,
    QVBoxLayout, QWidget,
)

from vramexp.core.units import human_bytes

from . import theme


class PlannerView(QWidget):
    def __init__(self, client, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.client = client
        self._previewed: str | None = None
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_planner())
        splitter.addWidget(self._build_profiles())
        splitter.setSizes([640, 440])
        layout.addWidget(splitter, 1)

    # -- pianificatore -----------------------------------------------------
    def _build_planner(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)

        title = QLabel("Ci sta in VRAM?")
        title.setObjectName("Title")
        layout.addWidget(title)

        subtitle = QLabel(
            "Calcola pesi, cache KV e overhead per un modello della tua libreria, "
            "e dice cosa liberare se non entra.")
        subtitle.setObjectName("Subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        row = QHBoxLayout()
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("nome del modello, anche parziale…")
        self.model_input.returnPressed.connect(self._run_plan)
        row.addWidget(self.model_input, 1)

        row.addWidget(QLabel("Contesto:"))
        self.context_spin = QSpinBox()
        self.context_spin.setRange(0, 4_000_000)
        self.context_spin.setSingleStep(4096)
        self.context_spin.setValue(32768)
        self.context_spin.setSpecialValueText("del modello")
        row.addWidget(self.context_spin)

        row.addWidget(QLabel("Cache KV:"))
        self.kv_combo = QComboBox()
        self.kv_combo.addItems(["f16", "q8_0", "q5_1", "q4_0"])
        row.addWidget(self.kv_combo)

        run_button = QPushButton("Calcola")
        run_button.setObjectName("Primary")
        run_button.clicked.connect(self._run_plan)
        row.addWidget(run_button)
        layout.addLayout(row)

        browse = QPushButton("Sfoglia la libreria di modelli")
        browse.clicked.connect(self._browse_library)
        layout.addWidget(browse, 0, Qt.AlignmentFlag.AlignLeft)

        self.plan_output = QTextEdit()
        self.plan_output.setReadOnly(True)
        self.plan_output.setPlaceholderText(
            "Il risultato apparirà qui: fabbisogno dettagliato, verdetto e — se non "
            "entra — quali processi scaricare e in che ordine.")
        layout.addWidget(self.plan_output, 1)
        return box

    def _browse_library(self) -> None:
        try:
            data = self.client.get("/library", limit=60)
        except Exception as e:
            QMessageBox.warning(self, "Libreria", f"Scansione fallita: {e}")
            return
        models = data.get("models", [])
        if not models:
            QMessageBox.information(
                self, "Libreria",
                "Nessun modello trovato nelle cartelle configurate.\n"
                "Verificare model_dirs nella configurazione.")
            return

        dialog = QMessageBox(self)
        dialog.setWindowTitle("Libreria di modelli")
        lines = [f"{data['count']} modelli · {human_bytes(data['total_bytes'])}\n"]
        for m in models[:30]:
            extra = " · ".join(x for x in (m.get("quantization"),
                                           f"{m['n_layers']}L" if m.get("n_layers") else None)
                               if x)
            lines.append(f"{human_bytes(m['size_bytes']):>9}  {m['name'][:56]}  {extra}")
        dialog.setText("Primi modelli per dimensione:")
        dialog.setDetailedText("\n".join(lines))
        dialog.exec()

    def _run_plan(self) -> None:
        query = self.model_input.text().strip()
        if not query:
            return
        context = self.context_spin.value() or None
        try:
            result = self.client.get("/plan", model=query, context=context,
                                     kv_type=self.kv_combo.currentText())
        except Exception as e:
            self.plan_output.setPlainText(f"Calcolo fallito: {e}")
            return
        self.plan_output.setHtml(_format_plan(result))

    # -- profili -----------------------------------------------------------
    def _build_profiles(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(10)

        title = QLabel("Profili operativi")
        title.setObjectName("Title")
        layout.addWidget(title)

        subtitle = QLabel("Riconfigurano la macchina per un tipo di lavoro. "
                          "L'anteprima è obbligatoria prima di applicare.")
        subtitle.setObjectName("Subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        self.profile_list = QListWidget()
        self.profile_list.setMinimumHeight(210)
        self.profile_list.setMaximumHeight(260)
        self.profile_list.setWordWrap(True)
        self.profile_list.currentItemChanged.connect(self._on_profile_selected)
        layout.addWidget(self.profile_list)

        buttons = QHBoxLayout()
        preview_button = QPushButton("Anteprima")
        preview_button.clicked.connect(self._preview_profile)
        buttons.addWidget(preview_button)

        self.apply_button = QPushButton("Applica")
        self.apply_button.setObjectName("Primary")
        self.apply_button.setEnabled(False)
        self.apply_button.setToolTip("Disponibile dopo aver visto l'anteprima")
        self.apply_button.clicked.connect(self._apply_profile)
        buttons.addWidget(self.apply_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.profile_output = QTextEdit()
        self.profile_output.setReadOnly(True)
        layout.addWidget(self.profile_output, 1)

        self.refresh_profiles()
        return box

    def refresh_profiles(self) -> None:
        self.profile_list.clear()
        try:
            profiles = self.client.get("/profiles").get("profiles", [])
        except Exception as e:
            self.profile_output.setPlainText(f"profili non disponibili: {e}")
            return
        for p in profiles:
            item = QListWidgetItem(f"{p['title']}\n    {p['description'][:76]}")
            item.setData(Qt.ItemDataRole.UserRole, p)
            self.profile_list.addItem(item)

    def _on_profile_selected(self, current, _previous) -> None:
        self.apply_button.setEnabled(False)
        self._previewed = None
        if current is None:
            return
        p = current.data(Qt.ItemDataRole.UserRole)
        lines = [f"<b>{p['title']}</b>", p["description"], "", "<b>Passi:</b>"]
        for step in p.get("steps", []):
            lines.append(f"• {step.get('description') or step.get('kind')}")
        self.profile_output.setHtml("<br>".join(lines))

    def _selected_profile(self) -> dict | None:
        item = self.profile_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _preview_profile(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            return
        try:
            result = self.client.post("/profiles/preview", {"name": profile["name"]})
        except Exception as e:
            self.profile_output.setPlainText(f"anteprima fallita: {e}")
            return
        if not result.get("ok"):
            self.profile_output.setPlainText(str(result.get("error")))
            return

        lines = [f"<b>{result['profile']['title']}</b>", "",
                 f"Passi applicabili: {result['applicable_steps']} su "
                 f"{result['total_steps']}",
                 f"Memoria liberabile stimata: "
                 f"<b>{result['estimated_total_free_human'] or '—'}</b>", "", "───", ""]
        for item in result["items"]:
            mark = "●" if item["applicable"] else "○"
            color = theme.TEXT if item["applicable"] else theme.TEXT_FAINT
            free = (f"  <b>~{item['estimated_free_human']}</b>"
                    if item.get("estimated_free_human") else "")
            lines.append(f"<span style='color:{color}'>{mark} {item['title']}{free}</span>")
            lines.append(f"<span style='color:{theme.TEXT_DIM}'>&nbsp;&nbsp;&nbsp;"
                         f"{item['detail']}</span>")
            if item.get("warning"):
                lines.append(f"<span style='color:{theme.WARN}'>&nbsp;&nbsp;&nbsp;⚠ "
                             f"{item['warning']}</span>")
            lines.append("")
        self.profile_output.setHtml("<br>".join(lines))
        self._previewed = profile["name"]
        self.apply_button.setEnabled(True)

    def _apply_profile(self) -> None:
        profile = self._selected_profile()
        if profile is None or self._previewed != profile["name"]:
            QMessageBox.information(self, "Profili",
                                    "Eseguire prima l'anteprima di questo profilo.")
            return
        confirm = QMessageBox.question(
            self, "Applicare il profilo?",
            f"Applicare «{profile['title']}»?\n\n"
            "Verranno eseguite le azioni mostrate nell'anteprima.",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Cancel)
        if confirm != QMessageBox.StandardButton.Ok:
            return

        try:
            result = self.client.post("/profiles/apply",
                                      {"name": profile["name"], "confirmed": True})
        except Exception as e:
            QMessageBox.warning(self, "Profili", f"Applicazione fallita: {e}")
            return

        lines = [f"<b>{result.get('summary', '')}</b>", ""]
        for step in result.get("steps", []):
            mark = "✓" if step.get("ok") else "✗"
            color = theme.OK if step.get("ok") else theme.CRIT
            lines.append(f"<span style='color:{color}'>{mark} {step.get('kind')}</span>"
                         f" — {step.get('message')}")
        self.profile_output.setHtml("<br>".join(lines))
        self.apply_button.setEnabled(False)
        self._previewed = None


def _format_plan(result: dict) -> str:
    if not result.get("ok"):
        return (f"<span style='color:{theme.WARN}'>{result.get('error')}</span>"
                f"<br><br>{result.get('hint', '')}")

    model = result.get("model", {})
    fits = result.get("fits")
    color = theme.OK if fits else theme.CRIT
    lines = [
        f"<h3 style='margin-bottom:2px'>{model.get('name', '')}</h3>",
        f"<span style='color:{theme.TEXT_DIM}'>{human_bytes(model.get('size_bytes', 0))}"
        f" · {model.get('quantization') or '?'}"
        f" · {model.get('n_layers') or '?'} layer</span>",
        "<br>",
        f"<p style='color:{color}; font-size:11pt'><b>{result.get('verdict')}</b></p>",
    ]

    est = result.get("estimate") or {}
    if est:
        lines.append("<table cellpadding='3'>")
        for label, key in (("Pesi", "weights_bytes"), ("Cache KV", "kv_cache_bytes"),
                           ("Overhead", "overhead_bytes")):
            lines.append(
                f"<tr><td style='color:{theme.TEXT_DIM}'>{label}</td>"
                f"<td align='right'><b>{human_bytes(est.get(key, 0))}</b></td></tr>")
        lines.append(
            f"<tr><td style='color:{theme.TEXT}'><b>Totale</b></td>"
            f"<td align='right'><b>{human_bytes(est.get('total_bytes', 0))}</b></td></tr>")
        lines.append("</table>")
        if not est.get("is_estimate"):
            lines.append(f"<p style='color:{theme.OK}'>{est.get('confidence_note')}</p>")
        for w in est.get("warnings", []):
            lines.append(f"<p style='color:{theme.WARN}'>⚠ {w}</p>")

    steps = result.get("plan_steps") or []
    if steps:
        lines.append("<p><b>Per farlo entrare:</b></p><ul>")
        lines.extend(f"<li>{s}</li>" for s in steps)
        lines.append("</ul>")

    alternatives = result.get("alternatives") or []
    if alternatives:
        lines.append("<p><b>Oppure, senza liberare nulla:</b></p><ul>")
        lines.extend(f"<li>{a}</li>" for a in alternatives)
        lines.append("</ul>")

    if result.get("max_context_now"):
        lines.append(
            f"<p style='color:{theme.TEXT_DIM}'>Con la VRAM libera adesso il contesto "
            f"massimo sarebbe {result['max_context_now']:,} token.</p>".replace(",", " "))
    return "".join(lines)
