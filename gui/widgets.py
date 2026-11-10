"""Widget personalizzati: anello di occupazione, treemap, barre e badge.

Sono disegnati a mano perche' devono rappresentare una cosa che i widget standard
non sanno mostrare: la **doppia contabilita'** della VRAM. La memoria residente e
quella committed da Windows sono grandezze diverse e sommarle sarebbe scorretto,
quindi l'anello mostra la prima come riempimento e la seconda come arco esterno,
tenendole visivamente distinte.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QFontMetrics, QLinearGradient, QPainter, QPainterPath, QPen,
)
from PySide6.QtWidgets import QFrame, QSizePolicy, QWidget

from vramexp.core.units import human_bytes, pct

from . import theme


class UsageRing(QWidget):
    """Anello che mostra insieme memoria residente, committed e riserva.

    Tre livelli concentrici invece di una sola barra, perche' l'utente deve poter
    vedere a colpo d'occhio che il committed supera il residente — la situazione
    piu' comune e piu' fraintesa su Windows.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(190, 190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.used = 0
        self.total = 0
        self.committed = 0
        self.shared = 0
        self.reserved = 0
        self.gpu_name = ""

    def set_values(self, used: int, total: int, committed: int = 0,
                   shared: int = 0, reserved: int = 0, gpu_name: str = "") -> None:
        self.used, self.total = used, total
        self.committed, self.shared, self.reserved = committed, shared, reserved
        self.gpu_name = gpu_name
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (nome imposto da Qt)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        side = min(self.width(), self.height()) - 16
        rect = QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)
        thickness = max(12, side * 0.085)

        fraction = (self.used / self.total) if self.total else 0.0
        color = QColor(theme.color_for_usage(fraction))

        # Traccia di fondo
        pen = QPen(QColor(theme.BG_ELEVATED), thickness, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, 0, 360 * 16)

        # Arco della memoria residente: il dato principale
        span = int(-fraction * 360 * 16)
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, color.lighter(125))
        gradient.setColorAt(1.0, color)
        painter.setPen(QPen(QBrush(gradient), thickness, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, 90 * 16, span)

        # Arco esterno sottile: committed da Windows, che puo' superare il 100%.
        # Disegnarlo separato evita di far credere che la GPU sia oltre capienza.
        if self.committed and self.total:
            outer = rect.adjusted(-thickness * 0.72, -thickness * 0.72,
                                  thickness * 0.72, thickness * 0.72)
            cfrac = min(1.0, self.committed / self.total)
            painter.setPen(QPen(QColor(theme.TEXT_FAINT), max(2.0, thickness * 0.16),
                                Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawArc(outer, 90 * 16, int(-cfrac * 360 * 16))

        # Arco interno: quota riservata dal balloon
        if self.reserved and self.total:
            inner = rect.adjusted(thickness * 0.78, thickness * 0.78,
                                  -thickness * 0.78, -thickness * 0.78)
            rfrac = min(1.0, self.reserved / self.total)
            painter.setPen(QPen(QColor(theme.SPILL), max(2.0, thickness * 0.16),
                                Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawArc(inner, 90 * 16, int(-rfrac * 360 * 16))

        # Testo centrale
        painter.setPen(QColor(theme.TEXT))
        font = QFont(theme.FONT_UI, int(side * 0.115), QFont.Weight.Light)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{fraction * 100:.0f}%")

        painter.setPen(QColor(theme.TEXT_DIM))
        painter.setFont(QFont(theme.FONT_UI, max(7, int(side * 0.048))))
        sub = QRectF(rect.x(), rect.center().y() + side * 0.10, rect.width(), side * 0.14)
        painter.drawText(sub, Qt.AlignmentFlag.AlignCenter,
                         f"{human_bytes(self.used)} / {human_bytes(self.total)}")

        if self.shared:
            painter.setPen(QColor(theme.SPILL))
            painter.setFont(QFont(theme.FONT_UI, max(7, int(side * 0.042)),
                                  QFont.Weight.DemiBold))
            warn = QRectF(rect.x(), rect.center().y() + side * 0.23,
                          rect.width(), side * 0.13)
            painter.drawText(warn, Qt.AlignmentFlag.AlignCenter,
                             f"▲ {human_bytes(self.shared)} su RAM")
        painter.end()


class Treemap(QWidget):
    """Mappa proporzionale dei consumatori di VRAM.

    L'area di ogni riquadro e' proporzionale alla memoria occupata: e' il modo
    piu' immediato per far vedere "chi si e' preso la GPU" senza leggere numeri.
    Usa l'algoritmo squarified, che produce rettangoli vicini al quadrato e quindi
    piu' facili da confrontare a occhio rispetto a strisce lunghe e sottili.
    """

    item_clicked = Signal(int)          # pid
    item_double_clicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setMouseTracking(True)
        self._items: list[dict] = []
        self._rects: list[tuple[QRectF, dict]] = []
        self._hover_pid: int | None = None

    def set_items(self, items: list[dict]) -> None:
        """`items`: dict con pid, label, bytes, shared, kind."""
        self._items = [i for i in items if i.get("bytes", 0) > 0]
        self._layout()
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._layout()
        super().resizeEvent(event)

    def _layout(self) -> None:
        self._rects = []
        if not self._items:
            return
        total = sum(i["bytes"] for i in self._items)
        if total <= 0:
            return
        area = QRectF(0, 0, self.width(), self.height())
        self._squarify(list(self._items), area, total)

    def _squarify(self, items: list[dict], rect: QRectF, total: int) -> None:
        """Suddivisione ricorsiva mantenendo i rettangoli vicini al quadrato."""
        if not items or rect.width() <= 1 or rect.height() <= 1:
            return
        if len(items) == 1:
            self._rects.append((rect, items[0]))
            return

        # Divide la lista in due gruppi di peso simile e taglia lungo il lato
        # piu' lungo: mantiene le proporzioni ragionevoli senza il costo pieno
        # dell'algoritmo originale.
        half, acc = 0, 0
        target = total / 2
        for i, item in enumerate(items):
            if acc + item["bytes"] > target and i > 0:
                half = i
                break
            acc += item["bytes"]
            half = i + 1
        half = max(1, min(half, len(items) - 1))

        first, second = items[:half], items[half:]
        w1 = sum(i["bytes"] for i in first)
        w2 = total - w1
        if w1 <= 0 or w2 <= 0:
            self._rects.append((rect, items[0]))
            return

        if rect.width() >= rect.height():
            split = rect.width() * (w1 / total)
            r1 = QRectF(rect.x(), rect.y(), split, rect.height())
            r2 = QRectF(rect.x() + split, rect.y(), rect.width() - split, rect.height())
        else:
            split = rect.height() * (w1 / total)
            r1 = QRectF(rect.x(), rect.y(), rect.width(), split)
            r2 = QRectF(rect.x(), rect.y() + split, rect.width(), rect.height() - split)

        self._squarify(first, r1, w1)
        self._squarify(second, r2, w2)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self._rects:
            painter.setPen(QColor(theme.TEXT_FAINT))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "nessun processo sta usando la VRAM")
            painter.end()
            return

        for index, (rect, item) in enumerate(self._rects):
            inner = rect.adjusted(2, 2, -2, -2)
            if inner.width() < 3 or inner.height() < 3:
                continue

            base = QColor(theme.series_color(index))
            if item.get("pid") == self._hover_pid:
                base = base.lighter(135)

            path = QPainterPath()
            path.addRoundedRect(inner, 7, 7)
            gradient = QLinearGradient(inner.topLeft(), inner.bottomRight())
            gradient.setColorAt(0.0, base)
            gradient.setColorAt(1.0, base.darker(145))
            painter.fillPath(path, QBrush(gradient))

            # Il bordo colorato segnala lo spillover anche a chi non distingue i
            # colori del riempimento: e' accompagnato dal simbolo nel testo.
            if item.get("shared", 0) > 64 * 1024 * 1024:
                painter.setPen(QPen(QColor(theme.SPILL), 2.5))
            else:
                painter.setPen(QPen(QColor(theme.BG), 1))
            painter.drawPath(path)

            self._draw_label(painter, inner, item)
        painter.end()

    def _draw_label(self, painter: QPainter, rect: QRectF, item: dict) -> None:
        if rect.width() < 56 or rect.height() < 30:
            return
        painter.setPen(QColor("#ffffff"))
        font = QFont(theme.FONT_UI, 9, QFont.Weight.DemiBold)
        painter.setFont(font)
        metrics = QFontMetrics(font)

        label = str(item.get("label") or item.get("name") or "?")
        if item.get("shared", 0) > 64 * 1024 * 1024:
            label = "▲ " + label
        elided = metrics.elidedText(label, Qt.TextElideMode.ElideRight,
                                    int(rect.width() - 14))
        text_rect = rect.adjusted(7, 6, -7, -7)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                         elided)

        if rect.height() >= 46:
            painter.setFont(QFont(theme.FONT_UI, 8))
            painter.setPen(QColor(255, 255, 255, 205))
            painter.drawText(text_rect,
                             Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
                             human_bytes(item.get("bytes", 0)))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pid = self._pid_at(event.position())
        if pid != self._hover_pid:
            self._hover_pid = pid
            item = next((i for _, i in self._rects if i.get("pid") == pid), None)
            if item:
                shared = item.get("shared", 0)
                tip = (f"<b>{item.get('label')}</b><br>pid {item.get('pid')}<br>"
                       f"{human_bytes(item.get('bytes', 0))} dedicati")
                if shared:
                    tip += (f"<br><span style='color:{theme.SPILL}'>"
                            f"{human_bytes(shared)} su RAM di sistema</span>")
                self.setToolTip(tip)
            else:
                self.setToolTip("")
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover_pid = None
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pid = self._pid_at(event.position())
        if pid is not None:
            self.item_clicked.emit(pid)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        pid = self._pid_at(event.position())
        if pid is not None:
            self.item_double_clicked.emit(pid)
        super().mouseDoubleClickEvent(event)

    def _pid_at(self, pos: QPointF) -> int | None:
        for rect, item in self._rects:
            if rect.contains(pos):
                return item.get("pid")
        return None


class StatTile(QFrame):
    """Riquadro con un valore grande e un'etichetta."""

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._label = label
        self._value = "—"
        self._detail = ""
        self._accent = theme.TEXT
        self.setMinimumHeight(84)

    def set_value(self, value: str, detail: str = "", accent: str | None = None) -> None:
        self._value, self._detail = value, detail
        self._accent = accent or theme.TEXT
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(14, 10, -14, -10)

        painter.setPen(QColor(theme.TEXT_DIM))
        painter.setFont(QFont(theme.FONT_UI, 8))
        painter.drawText(rect, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                         self._label.upper())

        painter.setPen(QColor(self._accent))
        painter.setFont(QFont(theme.FONT_UI, 16, QFont.Weight.DemiBold))
        painter.drawText(rect.adjusted(0, 16, 0, 0),
                         Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                         self._value)

        if self._detail:
            painter.setPen(QColor(theme.TEXT_FAINT))
            painter.setFont(QFont(theme.FONT_UI, 8))
            painter.drawText(rect, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
                             self._detail)
        painter.end()


class MiniBar(QWidget):
    """Barra orizzontale compatta per le celle della tabella."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.fraction = 0.0
        self.color = theme.ACCENT
        self.setFixedHeight(6)

    def set_fraction(self, fraction: float, color: str | None = None) -> None:
        self.fraction = max(0.0, min(1.0, fraction))
        if color:
            self.color = color
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.BG_ELEVATED))
        width = int(self.width() * self.fraction)
        if width > 0:
            painter.fillRect(0, 0, width, self.height(), QColor(self.color))
        painter.end()
