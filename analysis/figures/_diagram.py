"""Tiny box-and-arrow layer over matplotlib for the BAB 3 method diagrams.

Coordinates are inches on the printed page, origin top-left, y downwards, so a
9 pt label in the code is 9 pt in the thesis. Box heights are computed from the
number of text lines they hold, which keeps every box exactly as tall as its
content and lets the scripts lay out rows without hand-tuned magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import _common as C

PT = 1 / 72  # inch per point


def line_h(size: float) -> float:
    return size * 1.32 * PT


@dataclass
class Line:
    text: str
    size: float = 8.0
    weight: str = "normal"
    color: str = C.INK2
    style: str = "normal"


class Canvas:
    def __init__(self, width: float, height: float):
        self.w, self.h = width, height
        self.fig = plt.figure(figsize=(width, height))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.axis("off")

    # ----------------------------------------------------------------- shapes
    def rect(self, x, y, w, h, fill="white", edge=C.RULE, lw=0.8, radius=0.06,
             z=1):
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z))

    def text(self, x, y, s, size=8.0, weight="normal", color=C.INK2,
             ha="left", va="top", style="normal", z=5):
        self.ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
                     ha=ha, va=va, fontstyle=style, zorder=z, linespacing=1.15)

    def lines(self, x, y, lines: Sequence[Line], ha="left") -> float:
        """Stack lines downwards from y; returns the y below the last line."""
        for ln in lines:
            self.text(x, y, ln.text, size=ln.size, weight=ln.weight,
                      color=ln.color, ha=ha, style=ln.style)
            y += line_h(ln.size)
        return y

    def arrow(self, pts: Sequence[Tuple[float, float]], color=C.INK2, lw=0.9,
              head: float = 8.0):
        """Poly-line arrow through pts; the head sits on the last point."""
        for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
            self.ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=2,
                         solid_capstyle="projecting")
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        self.ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=head,
            color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=2))


def block_height(lines: Sequence[Line], pad_y: float) -> float:
    return 2 * pad_y + sum(line_h(l.size) for l in lines)


def card(cv: Canvas, x: float, y: float, w: float, lines: Sequence[Line],
         fill="white", edge=C.RULE, pad_x=0.08, pad_y=0.07, ha="left",
         h: Optional[float] = None, lw=0.8) -> float:
    """A box sized to its lines (or to h). Returns its height."""
    h = h if h is not None else block_height(lines, pad_y)
    cv.rect(x, y, w, h, fill=fill, edge=edge, lw=lw)
    tx = x + pad_x if ha == "left" else x + w / 2
    # vertically centre the text block inside a box taller than its content
    content = sum(line_h(l.size) for l in lines)
    cv.lines(tx, y + (h - content) / 2, lines, ha=ha)
    return h


def text_width(s: str, size: float, weight: str = "normal") -> float:
    """Rendered width of s in inches, measured with the font actually in use."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    import matplotlib as mpl
    fp = FontProperties(family=mpl.rcParams["font.sans-serif"][0], weight=weight)
    if not s.strip():
        return 0.0
    return TextPath((0, 0), s, size=size, prop=fp).get_extents().width * PT


def wrap(text: str, size: float, max_w: float, weight: str = "normal",
         first: str = "", cont: str = "") -> List[str]:
    """Greedy word wrap to a measured width (inches)."""
    words, out, cur = text.split(), [], first
    for w in words:
        trial = cur + w if (not cur.strip() or cur.endswith(" ")) else cur + " " + w
        if text_width(trial, size, weight) <= max_w or cur in (first, cont):
            cur = trial
        else:
            out.append(cur)
            cur = cont + w
    out.append(cur)
    return out


def title(text: str, max_w: float, size=8.6, color=C.INK) -> List[Line]:
    return [Line(t, size=size, weight="bold", color=color)
            for t in wrap(text, size, max_w, weight="bold")]


def bullets(items: Sequence[str], size=7.8, color=C.INK2,
            max_w: Optional[float] = None) -> List[Line]:
    """Bullet lines; with ``max_w`` (inches), long items continue on an
    indented line instead of running past the box edge."""
    out: List[Line] = []
    for t in items:
        parts = wrap(t, size, max_w, first="• ", cont="   ") if max_w \
            else [f"• {t}"]
        out += [Line(p, size=size, color=color) for p in parts]
    return out
