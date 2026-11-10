"""Tema visivo dell'applicazione.

Palette scura con accenti freddi per lo stato normale e caldi per gli allarmi.
I colori di stato non sono mai l'unico veicolo di informazione: ogni condizione
critica porta anche un'icona o un testo, perche' una barra rossa non dice nulla a
chi non distingue il rosso dal verde.

I contrasti testo/sfondo rispettano il rapporto 4,5:1 richiesto per il testo
normale.
"""

from __future__ import annotations

# --- palette ---------------------------------------------------------------
BG = "#0d1117"
BG_ELEVATED = "#161b22"
BG_CARD = "#1c2128"
BG_HOVER = "#22272e"
BORDER = "#30363d"
BORDER_STRONG = "#444c56"

TEXT = "#e6edf3"
TEXT_DIM = "#9198a1"
TEXT_FAINT = "#6e7681"

ACCENT = "#4493f8"
ACCENT_DIM = "#1f6feb"

OK = "#3fb950"
WARN = "#d29922"
CRIT = "#f85149"
SPILL = "#bc8cff"          # spillover: categoria a se', non un semplice "peggio"

# Serie per i grafici e la treemap: scelte per restare distinguibili anche in
# scala di grigi e per chi non percepisce le differenze rosso/verde.
SERIES = [
    "#4493f8", "#bc8cff", "#3fb950", "#d29922", "#f778ba",
    "#56d4dd", "#ff9e64", "#a5d6ff", "#7ee787", "#ffa198",
    "#d2a8ff", "#79c0ff",
]

FONT_UI = "Segoe UI"
FONT_MONO = "Cascadia Mono, Consolas, monospace"


def color_for_usage(fraction: float) -> str:
    """Colore in base alla pressione sulla memoria."""
    if fraction >= 0.97:
        return CRIT
    if fraction >= 0.90:
        return WARN
    return ACCENT


def series_color(index: int) -> str:
    return SERIES[index % len(SERIES)]


STYLESHEET = f"""
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: "{FONT_UI}";
    font-size: 10pt;
}}

QMainWindow, QDialog {{ background-color: {BG}; }}

QLabel#Title {{ font-size: 17pt; font-weight: 600; color: {TEXT}; }}
QLabel#Subtitle {{ font-size: 10pt; color: {TEXT_DIM}; }}
QLabel#SectionTitle {{
    font-size: 11pt; font-weight: 600; color: {TEXT}; padding: 2px 0 6px 0;
}}
QLabel#Dim {{ color: {TEXT_DIM}; }}
QLabel#Faint {{ color: {TEXT_FAINT}; font-size: 9pt; }}
QLabel#BigNumber {{ font-size: 26pt; font-weight: 300; color: {TEXT}; }}
QLabel#Mono {{ font-family: {FONT_MONO}; }}

QFrame#Card {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QFrame#Card:hover {{ border-color: {BORDER_STRONG}; }}

QFrame#Separator {{ background-color: {BORDER}; max-height: 1px; border: none; }}

QPushButton {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 8px 15px;
    color: {TEXT};
    min-height: 20px;
}}
QPushButton:hover {{ background-color: {BG_HOVER}; border-color: {BORDER_STRONG}; }}
QPushButton:pressed {{ background-color: {BG}; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; border-color: {BORDER}; }}
QPushButton:focus {{ border: 2px solid {ACCENT}; outline: none; }}

QPushButton#Primary {{
    background-color: {ACCENT_DIM};
    border-color: {ACCENT};
    font-weight: 600;
}}
QPushButton#Primary:hover {{ background-color: {ACCENT}; }}
QPushButton#Danger {{ border-color: {CRIT}; color: {CRIT}; }}
QPushButton#Danger:hover {{ background-color: {CRIT}; color: {BG}; }}

QTabWidget::pane {{ border: none; background: {BG}; }}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_DIM};
    padding: 10px 18px;
    border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:focus {{ border: 1px solid {ACCENT}; }}

QTableView, QTreeView {{
    background-color: {BG_ELEVATED};
    alternate-background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    gridline-color: {BORDER};
    selection-background-color: {ACCENT_DIM};
    selection-color: #ffffff;
}}
QHeaderView::section {{
    background-color: {BG_CARD};
    color: {TEXT_DIM};
    padding: 7px;
    border: none;
    border-bottom: 1px solid {BORDER};
    font-weight: 600;
}}
QHeaderView::section:hover {{ color: {TEXT}; }}

QScrollBar:vertical {{ background: transparent; width: 11px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: {BORDER_STRONG}; border-radius: 5px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {TEXT_FAINT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; }}
QScrollBar::handle:horizontal {{
    background: {BORDER_STRONG}; border-radius: 5px; min-width: 30px;
}}

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 7px 10px;
    min-height: 20px;
}}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER_STRONG};
    selection-background-color: {ACCENT_DIM};
}}

QSlider::groove:horizontal {{
    background: {BG_ELEVATED}; height: 6px; border-radius: 3px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: {TEXT}; width: 16px; height: 16px;
    margin: -6px 0; border-radius: 8px;
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 17px; height: 17px;
    border: 1px solid {BORDER_STRONG}; border-radius: 4px;
    background: {BG_ELEVATED};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QCheckBox:focus, QRadioButton:focus {{ border: 1px solid {ACCENT}; }}

QToolTip {{
    background-color: {BG_CARD};
    color: {TEXT};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    padding: 7px;
}}

QGroupBox {{
    border: 1px solid {BORDER}; border-radius: 8px;
    margin-top: 14px; padding-top: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {TEXT_DIM};
}}

QTextEdit, QPlainTextEdit {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    border-radius: 8px;
    font-family: {FONT_MONO};
}}

QProgressBar {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    border-radius: 6px;
    text-align: center;
    height: 20px;
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 5px; }}

QSplitter::handle {{ background-color: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 2px; }}
QSplitter::handle:vertical {{ height: 2px; }}
QMenu {{
    background-color: {BG_CARD}; border: 1px solid {BORDER_STRONG};
    border-radius: 8px; padding: 5px;
}}
QMenu::item {{ padding: 8px 26px 8px 14px; border-radius: 5px; }}
QMenu::item:selected {{ background-color: {ACCENT_DIM}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 5px 8px; }}
"""
