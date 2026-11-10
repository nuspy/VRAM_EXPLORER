"""Finestra principale e avvio dell'applicazione.

Organizzata in schede: la dashboard e' cio' che si guarda di solito, le altre
servono quando si sta facendo qualcosa di specifico. Il pannello di dettaglio sta
a destra e resta sempre visibile, perche' selezionare un processo e poi dover
cambiare vista per agire sarebbe un passaggio di troppo.

L'icona nell'area di notifica consente di chiudere la finestra lasciando attivo
il monitoraggio, e mostra gli avvisi quando succede qualcosa che merita
attenzione — sempre come notifica, mai come azione automatica.
"""

from __future__ import annotations

import logging
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow, QMenu, QMessageBox,
    QSplitter, QStatusBar, QSystemTrayIcon, QTabWidget, QVBoxLayout, QWidget,
)

from vramexp.core.config import Config
from vramexp.core.units import human_bytes, pct

from . import theme
from .client import ServiceClient
from .controls_view import ControlsView
from .dashboard import Dashboard
from .detail_panel import DetailPanel
from .planner_view import PlannerView
from .timeline import TimelineView

log = logging.getLogger("vramexp.gui")


class MainWindow(QMainWindow):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.client = ServiceClient(cfg)
        self._last_snapshot: dict | None = None
        self._selected_pid: int | None = None

        self.setWindowTitle("VRAM Explorer")
        self.resize(1500, 950)
        self.setMinimumSize(1040, 680)
        self.setWindowIcon(_make_icon(theme.ACCENT))

        self._build()
        self._build_tray()

        self.client.snapshot_ready.connect(self._on_snapshot)
        self.client.incident_received.connect(self._on_incident)
        self.client.connection_changed.connect(self._on_connection)
        self.client.start()

        # Il pannello di dettaglio interroga il servizio: aggiornarlo a ogni
        # campione (1 Hz) moltiplicherebbe le richieste senza che l'utente
        # percepisca la differenza.
        self._detail_timer = QTimer(self)
        self._detail_timer.setInterval(2500)
        self._detail_timer.timeout.connect(self._refresh_detail)
        self._detail_timer.start()

    # -- costruzione -------------------------------------------------------
    def _build(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        self.tabs = QTabWidget()
        self.dashboard = Dashboard()
        self.dashboard.process_selected.connect(self._on_process_selected)
        self.dashboard.process_activated.connect(self._on_process_activated)
        self.tabs.addTab(self.dashboard, "Panoramica")

        self.timeline = TimelineView(self.client)
        self.tabs.addTab(self.timeline, "Storico e incidenti")

        self.controls = ControlsView(self.client)
        self.tabs.addTab(self.controls, "Controlli")

        self.planner = PlannerView(self.client)
        self.tabs.addTab(self.planner, "Pianificatore")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        splitter.addWidget(self.tabs)

        self.detail = DetailPanel(self.client)
        self.detail.action_requested.connect(self._on_action)
        self.detail.attach_requested.connect(self._on_attach)
        self.detail.setMinimumWidth(360)
        self.detail.setMaximumWidth(560)
        splitter.addWidget(self.detail)
        splitter.setSizes([1080, 420])

        layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

        status = QStatusBar()
        self.connection_label = QLabel("avvio…")
        status.addWidget(self.connection_label)
        self.summary_label = QLabel()
        status.addPermanentWidget(self.summary_label)
        self.setStatusBar(status)

        self._build_menu()

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("&File")

        refresh = QAction("&Aggiorna tutto", self)
        refresh.setShortcut(QKeySequence.StandardKey.Refresh)
        refresh.triggered.connect(self._refresh_all)
        menu.addAction(refresh)

        menu.addSeparator()
        hide = QAction("&Riduci nell'area di notifica", self)
        hide.setShortcut("Ctrl+H")
        hide.triggered.connect(self.hide)
        menu.addAction(hide)

        quit_action = QAction("&Esci", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self._quit)
        menu.addAction(quit_action)

        help_menu = self.menuBar().addMenu("&?")
        about = QAction("&Informazioni", self)
        about.triggered.connect(self._show_about)
        help_menu.addAction(about)

    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(_make_icon(theme.ACCENT), self)
        self.tray.setToolTip("VRAM Explorer")

        menu = QMenu()
        show = QAction("Mostra la finestra", self)
        show.triggered.connect(self._restore)
        menu.addAction(show)
        menu.addSeparator()
        quit_action = QAction("Esci", self)
        quit_action.triggered.connect(self._quit)
        menu.addAction(quit_action)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda reason: self._restore()
            if reason == QSystemTrayIcon.ActivationReason.DoubleClick else None)
        self.tray.show()

    # -- eventi ------------------------------------------------------------
    def _on_snapshot(self, snapshot: dict) -> None:
        self._last_snapshot = snapshot
        if self.tabs.currentWidget() is self.dashboard:
            self.dashboard.update_snapshot(snapshot)

        gpus = snapshot.get("gpus") or []
        if not gpus:
            return
        gpu = gpus[0]
        used_pct = pct(gpu.get("used_bytes", 0), gpu.get("total_bytes", 0))
        shared = gpu.get("shared_used_bytes") or 0

        summary = (f"{human_bytes(gpu.get('used_bytes', 0))} / "
                   f"{human_bytes(gpu.get('total_bytes', 0))}  ({used_pct:.0f}%)")
        if shared:
            summary += f"   ▲ {human_bytes(shared)} su RAM"
        self.summary_label.setText(summary)

        color = theme.color_for_usage(used_pct / 100)
        self.tray.setIcon(_make_icon(color, used_pct / 100))
        self.tray.setToolTip(f"VRAM Explorer — {summary}")

    def _on_incident(self, incident: dict) -> None:
        titles = {
            "oom_suspect": "Possibile esaurimento della VRAM",
            "spillover": "Un processo sta usando la RAM di sistema",
            "pressure_critical": "VRAM quasi esaurita",
            "pressure_warn": "VRAM in esaurimento",
        }
        title = titles.get(incident.get("kind", ""))
        if not title:
            return   # gli eventi ordinari restano nello storico senza notifica
        name = incident.get("process_name") or ""
        detail = (incident.get("detail") or "")[:160]
        self.tray.showMessage(f"{title}{f' — {name}' if name else ''}", detail,
                              QSystemTrayIcon.MessageIcon.Warning, 8000)

    def _on_connection(self, connected: bool, message: str) -> None:
        color = theme.OK if connected else theme.WARN
        self.connection_label.setText(f"● {message}")
        self.connection_label.setStyleSheet(f"color: {color};")

    def _on_process_selected(self, pid: int) -> None:
        self._selected_pid = pid
        self.detail.show_process(pid)

    def _on_process_activated(self, pid: int) -> None:
        self._on_process_selected(pid)
        self.dashboard.select_pid(pid)

    def _on_tab_changed(self, index: int) -> None:
        widget = self.tabs.widget(index)
        if widget is self.dashboard and self._last_snapshot:
            self.dashboard.update_snapshot(self._last_snapshot)
        elif widget is self.timeline:
            self.timeline.refresh()
        elif widget is self.controls:
            self.controls.refresh()

    def _refresh_detail(self) -> None:
        if self._selected_pid is not None and self.detail.isVisible():
            self.detail.show_process(self._selected_pid)

    def _refresh_all(self) -> None:
        if self._last_snapshot:
            self.dashboard.update_snapshot(self._last_snapshot)
        self.timeline.refresh()
        self.controls.refresh()
        self.planner.refresh_profiles()
        self._refresh_detail()

    def _on_action(self, pid: int, kind: str, confirmed: bool) -> None:
        try:
            result = self.client.post("/action", {"pid": pid, "kind": kind,
                                                  "confirmed": confirmed})
        except Exception as e:
            QMessageBox.warning(self, "Azione fallita", str(e))
            return

        if result.get("ok"):
            freed = result.get("freed_bytes", 0)
            message = str(result.get("message", ""))
            if freed:
                message += f"\n\nMemoria liberata: {human_bytes(freed)}"
            else:
                message += ("\n\nNessuna variazione immediata della memoria libera: "
                            "alcuni backend impiegano qualche secondo a rilasciarla.")
            QMessageBox.information(self, "Fatto", message)
        else:
            QMessageBox.warning(self, "Azione non eseguita",
                                str(result.get("message", "")))
        self._refresh_detail()

    def _on_attach(self, pid: int) -> None:
        try:
            eligibility = self.client.get(f"/probe/eligibility/{pid}")
        except Exception as e:
            QMessageBox.warning(self, "Iniezione", str(e))
            return

        if not eligibility.get("eligible"):
            QMessageBox.information(self, "Non possibile", eligibility.get("reason", ""))
            return

        if not eligibility.get("hot_attach_enabled"):
            QMessageBox.information(
                self, "Iniezione disattivata",
                "L'iniezione a caldo è disattivata nella configurazione.\n\n"
                "Può destabilizzare il processo osservato, quindi va abilitata "
                "consapevolmente (enable_hot_attach).\n\n"
                "L'alternativa sicura è riavviare il processo con la variabile "
                "d'ambiente VRAMEXP_AGENT=1.")
            return

        confirm = QMessageBox.warning(
            self, "Iniettare l'agent?",
            f"Iniettare l'agent nel processo {pid}?\n\n"
            "Verrà eseguito codice all'interno del processo. L'operazione riesce "
            "quasi sempre, ma in casi sfortunati può bloccare o far terminare "
            "l'applicazione osservata.\n\n"
            "Se il processo sta svolgendo un lavoro lungo e prezioso, conviene "
            "rinunciare e riavviarlo invece con VRAMEXP_AGENT=1.",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Cancel)
        if confirm != QMessageBox.StandardButton.Ok:
            return

        try:
            result = self.client.post("/probe/attach", {"pid": pid, "confirmed": True})
        except Exception as e:
            QMessageBox.warning(self, "Iniezione fallita", str(e))
            return
        if result.get("ok"):
            QMessageBox.information(self, "Fatto", result.get("message", ""))
        else:
            QMessageBox.warning(self, "Non riuscita", result.get("message", ""))
        self._refresh_detail()

    def _show_about(self) -> None:
        QMessageBox.about(
            self, "VRAM Explorer",
            "<h3>VRAM Explorer</h3>"
            "<p>Esplorazione e gestione della memoria video su Windows.</p>"
            "<p style='color:#9198a1'>La memoria per processo viene letta dai "
            "contatori di prestazione di Windows, perché su driver WDDM NVML non "
            "la espone. I valori per processo sono <i>committed</i> e la loro "
            "somma può superare la capacità della GPU: è normale.</p>"
            "<p style='color:#9198a1'>Il servizio osserva e segnala; non libera "
            "né termina nulla di propria iniziativa.</p>")

    # -- finestra ----------------------------------------------------------
    def _restore(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event) -> None:  # noqa: N802
        # La chiusura riduce nell'area di notifica invece di terminare: il
        # monitoraggio ha senso solo se continuo.
        if self.tray.isVisible():
            self.hide()
            self.tray.showMessage(
                "VRAM Explorer", "Il monitoraggio continua in background.",
                QSystemTrayIcon.MessageIcon.Information, 3000)
            event.ignore()
            return
        self._shutdown()
        event.accept()

    def _quit(self) -> None:
        self._shutdown()
        QApplication.quit()

    def _shutdown(self) -> None:
        self._detail_timer.stop()
        self.client.stop()
        self.tray.hide()


def _make_icon(color: str, fill: float = 1.0) -> QIcon:
    """Icona generata a runtime: un anello che riflette l'occupazione.

    Evita di dipendere da file esterni e rende l'icona nell'area di notifica
    informativa invece che decorativa.
    """
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    from PySide6.QtGui import QPen
    painter.setPen(QPen(QColor(theme.BG_ELEVATED), 9))
    painter.drawEllipse(8, 8, 48, 48)
    painter.setPen(QPen(QColor(color), 9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawArc(8, 8, 48, 48, 90 * 16, int(-max(0.02, min(1.0, fill)) * 360 * 16))
    painter.end()
    return QIcon(pixmap)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s %(message)s")

    app = QApplication(sys.argv)
    app.setApplicationName("VRAM Explorer")
    app.setOrganizationName("VramExplorer")
    app.setStyleSheet(theme.STYLESHEET)
    app.setQuitOnLastWindowClosed(False)   # la chiusura riduce nel tray

    cfg = Config.load()
    window = MainWindow(cfg)
    if not cfg.start_minimized:
        window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
