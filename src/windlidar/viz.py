"""Shared plotting style (validated categorical palette, thin marks, recessive grid)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
         green="#008300", violet="#6250d6", red="#e34948")
CYCLE = list(C.values())


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=CYCLE), "lines.linewidth": 2.0, "lines.markersize": 5,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "legend.frameon": False, "legend.fontsize": 9, "figure.dpi": 110, "savefig.dpi": 200,
        "savefig.bbox": "tight", "axes.labelsize": 10,
    })


apply()


def title(fig, text, sub=None):
    fig.suptitle(text, x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK, y=1.115)
    if sub:
        fig.text(0.01, 1.04, sub, ha="left", va="top", fontsize=9.5, color=INK2)


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)
