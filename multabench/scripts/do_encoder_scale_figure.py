"""Generate encoder scale figures for §6.1 of the MulTaBench paper.

encoder_scale.pdf        — main figure: combined image+text, single panel
encoder_scale_cls.pdf    — appendix: two panels (DINO + E5), classification only
encoder_scale_reg.pdf    — appendix: two panels (DINO + E5), regression only
"""
import os
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIGS = os.path.join(_HERE, "../../../paper-multabench/figures")

import sys
sys.path.insert(0, os.path.join(_HERE, "../.."))

from multabench.leaderboard.main_paper.encoder_scale import make_figure, make_appendix_figure


def _save(fig, name):
    out = os.path.join(_FIGS, f"{name}.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format="pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    fig, _ = make_figure("all")
    _save(fig, "encoder_scale")

    fig, _ = make_appendix_figure("cls")
    _save(fig, "encoder_scale_cls")

    fig, _ = make_appendix_figure("reg")
    _save(fig, "encoder_scale_reg")
