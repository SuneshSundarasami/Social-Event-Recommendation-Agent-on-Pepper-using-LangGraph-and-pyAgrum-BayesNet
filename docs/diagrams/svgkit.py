"""Tiny SVG toolkit for the README figures.

Draws in the visual language of the project slides: solid rounded boxes with a
bold white title and italic role line, colour-coded by function (red = sensing,
green = decision, blue = acting, grey = knowledge), grey terminal ovals and
colour-coded feedback arrows. Text is measured with the real Arial metrics when
Pillow and the font are available, so boxes and labels fit their contents.

Figures have a transparent background. Every figure is written twice: NAME.svg
for light page backgrounds and NAME-dark.svg, where each colour that sits
directly on the page (text, lines, pale fills) is swapped through DARK. The
saturated node colours in PALETTE read well on both and are never swapped, so
the light-mode colours listed in DARK must not appear in PALETTE.
"""

from __future__ import annotations

import base64
import math
import os
from dataclasses import dataclass
from html import escape

FONT = "Arial, Helvetica, sans-serif"
MONO = "Consolas, Menlo, 'DejaVu Sans Mono', monospace"

INK = "#1F2328"
MUTED = "#5A6270"
LINE = "#5F6670"
HALO = "#FFFFFE"      # page background behind labels
SURFACE = "#FFFFFD"   # fill of cards, chips and bubbles
FAINT = "#C9CED4"     # thin guide lines

# name: (gradient top, gradient bottom, stroke, text colour)
PALETTE = {
    "red": ("#EE5243", "#D93A2B", "#B8301F", "#FFFFFF"),
    "darkred": ("#C93030", "#A91C1C", "#8E1515", "#FFFFFF"),
    "green": ("#4BB847", "#369F32", "#2A8427", "#FFFFFF"),
    "darkgreen": ("#25995D", "#1A7B47", "#146339", "#FFFFFF"),
    "lightgreen": ("#7CCB5A", "#62B444", "#4C9634", "#FFFFFF"),
    "blue": ("#167CCC", "#0262AB", "#024F8A", "#FFFFFF"),
    "steel": ("#5A88BA", "#44709F", "#375C85", "#FFFFFF"),
    "salmon": ("#EC6868", "#D84E4E", "#B83E3E", "#FFFFFF"),
    "grey": ("#9AA0A6", "#7D838A", "#676D73", "#FFFFFF"),
    "navy": ("#1A4468", "#0A2A43", "#061C2E", "#FFFFFF"),
    "amber": ("#F7C04A", "#EDA923", "#C98A0C", "#3A2A00"),
    "purple": ("#8A63C9", "#6E48AE", "#573790", "#FFFFFF"),
}

# name: (fill, stroke, text) for light containers, chips and outlines
LIGHT = {
    "green": ("#F2FAF1", "#8CCB88", "#1E7A2B"),
    "blue": ("#EEF5FB", "#8DB8DE", "#0A63AD"),
    "grey": ("#F5F6F8", "#CDD2D8", "#4A525C"),
    "amber": ("#FFF8E6", "#EBC567", "#8A5A00"),
    "red": ("#FDF0EE", "#F0A79E", "#B32B21"),
    "orange": ("#FDF3E7", "#E3A869", "#6B4218"),
    "navy": ("#EEF2F6", "#9FB2C6", "#0B2B46"),
    "white": (SURFACE, "#D0D5DB", INK),
    "darkgreen": ("#EEF7F1", "#7FBF98", "#15643A"),
    "lightgreen": ("#F3FAEF", "#A6D88E", "#3F7F2A"),
    "darkred": ("#FBEDED", "#E09A9A", "#8F1616"),
    "steel": ("#EFF4FA", "#9DB7D5", "#385D86"),
    "salmon": ("#FDF0F0", "#F0A8A8", "#B93F3F"),
    "purple": ("#F4F0FB", "#B9A5DD", "#583891"),
}

# light-mode colour -> dark-mode colour, for everything drawn on the page itself
DARK = {
    INK: "#E6EDF3", MUTED: "#9DA7B3", LINE: "#9DA7B3", HALO: "#0D1117", SURFACE: "#161B22",
    FAINT: "#3D444D", "#C3C8CE": "#30363D", "#EEF0F2": "#21262D", "#A0A6AD": "#6E7681",
    "#F7F8FA": "#161B22", "#E3E6EA": "#30363D", "#AEB4BB": "#484F58", "#B9BFC6": "#3D444D",
    "#D0D5DB": "#3D444D", "#CDD2D8": "#3D444D", "#F4F5F7": "#21262D", "#6B4E00": "#E3B341",
    # arrow colours
    "#7A8088": "#8B949E", "#C62828": "#F85149", "#1E8449": "#3FB950", "#0A64B0": "#58A6FF",
    # pale fills
    "#F2FAF1": "#0F2417", "#EEF5FB": "#0C1D33", "#F5F6F8": "#161B22", "#FFF8E6": "#2A2108",
    "#FDF0EE": "#2D1215", "#FDF3E7": "#2B1D0E", "#EEF2F6": "#0F1B29", "#EEF7F1": "#0E231A",
    "#F3FAEF": "#142613", "#FBEDED": "#2A1010", "#EFF4FA": "#111C2B", "#FDF0F0": "#2B1414",
    "#F4F0FB": "#1E1630",
    # text on pale fills
    "#1E7A2B": "#56D364", "#0A63AD": "#79C0FF", "#4A525C": "#9DA7B3", "#8A5A00": "#E3B341",
    "#B32B21": "#FF7B72", "#6B4218": "#F0B07A", "#0B2B46": "#A5C8E8", "#15643A": "#3FB950",
    "#3F7F2A": "#7EE787", "#8F1616": "#FF7B72", "#385D86": "#A5C8E8", "#B93F3F": "#FF9492",
    "#583891": "#D2A8FF",
}
assert not set(DARK) & {c for row in PALETTE.values() for c in row}, "DARK must not touch PALETTE"


# --------------------------------------------------------------------------- text metrics

_FONT_FILES = {
    ("normal", False): "arial.ttf",
    ("bold", False): "arialbd.ttf",
    ("normal", True): "ariali.ttf",
    ("bold", True): "arialbi.ttf",
}
_font_cache: dict = {}


def _pil_font(size: float, weight: str, italic: bool, mono: bool):
    key = (round(size * 4), weight, italic, mono)
    if key in _font_cache:
        return _font_cache[key]
    font = None
    try:
        from PIL import ImageFont

        name = ("consolab.ttf" if weight == "bold" else "consola.ttf") if mono \
            else _FONT_FILES[(weight, italic)]
        for folder in (os.environ.get("WINDIR", "C:/Windows") + "/Fonts",
                       "/usr/share/fonts/truetype/msttcorefonts", "/Library/Fonts"):
            path = os.path.join(folder, name)
            if os.path.exists(path):
                font = ImageFont.truetype(path, size=max(1, round(size * 4)))
                break
    except Exception:
        font = None
    _font_cache[key] = font
    return font


def text_width(s: str, size: float = 13, weight: str = "normal",
               italic: bool = False, mono: bool = False) -> float:
    font = _pil_font(size, weight, italic, mono)
    if font is not None:
        return font.getlength(s) / 4
    per = 0.55 if mono else (0.58 if weight == "bold" else 0.52)
    return len(s) * size * per


# --------------------------------------------------------------------------- geometry


@dataclass
class Box:
    """A placed shape; exposes anchor points on each side."""

    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def l(self, dy: float = 0):
        return (self.x, self.cy + dy)

    def r(self, dy: float = 0):
        return (self.x + self.w, self.cy + dy)

    def t(self, dx: float = 0):
        return (self.cx + dx, self.y)

    def b(self, dx: float = 0):
        return (self.cx + dx, self.y + self.h)


def _rounded_path(pts, r: float) -> str:
    if len(pts) < 2:
        return ""
    d = ["M%.1f,%.1f" % pts[0]]
    for i in range(1, len(pts) - 1):
        (px, py), (cx, cy), (nx, ny) = pts[i - 1], pts[i], pts[i + 1]
        d1 = math.hypot(cx - px, cy - py)
        d2 = math.hypot(nx - cx, ny - cy)
        rr = min(r, d1 / 2, d2 / 2)
        if rr < 0.5 or d1 == 0 or d2 == 0:
            d.append("L%.1f,%.1f" % (cx, cy))
            continue
        ax, ay = cx - (cx - px) / d1 * rr, cy - (cy - py) / d1 * rr
        bx, by = cx + (nx - cx) / d2 * rr, cy + (ny - cy) / d2 * rr
        d.append("L%.1f,%.1f Q%.1f,%.1f %.1f,%.1f" % (ax, ay, cx, cy, bx, by))
    d.append("L%.1f,%.1f" % pts[-1])
    return " ".join(d)


# --------------------------------------------------------------------------- canvas


class Svg:
    # Layouts were drawn with an 84 px title band above ``top``; the band is no
    # longer rendered, so the canvas is shortened by that much instead.
    HEADER = 84

    def __init__(self, width: float, height: float, title: str | None = None,
                 caption: str | None = None, pad: float = 20):
        self.w, self.h, self.pad = width, height - self.HEADER, pad
        self.title = ". ".join(t for t in (title, caption) if t)
        self.items: list[str] = []
        self.defs: dict[str, str] = {}
        self.top = pad
        self._shadow()

    # -- defs ------------------------------------------------------------------

    def _shadow(self) -> None:
        self.defs["shadow"] = (
            '<filter id="shadow" x="-10%" y="-10%" width="120%" height="140%">'
            '<feDropShadow dx="0" dy="2" stdDeviation="2.2" flood-color="#000000" '
            'flood-opacity="0.18"/></filter>')

    def _grad(self, color: str) -> str:
        gid = "g-" + color
        if gid not in self.defs:
            top, bottom, _, _ = PALETTE[color]
            self.defs[gid] = ('<linearGradient id="%s" x1="0" y1="0" x2="0" y2="1">'
                              '<stop offset="0" stop-color="%s"/><stop offset="1" stop-color="%s"/>'
                              '</linearGradient>' % (gid, top, bottom))
        return "url(#%s)" % gid

    def _marker(self, color: str) -> str:
        mid = "m-" + color.lstrip("#")
        if mid not in self.defs:
            self.defs[mid] = ('<marker id="%s" viewBox="0 0 12 12" refX="10.5" refY="6" '
                              'markerWidth="12" markerHeight="12" markerUnits="userSpaceOnUse" '
                              'orient="auto-start-reverse"><path d="M1,1.5 L11,6 L1,10.5 L3.5,6 z" '
                              'fill="%s"/></marker>' % (mid, color))
        return "url(#%s)" % mid

    # -- primitives ------------------------------------------------------------

    def raw(self, s: str) -> None:
        self.items.append(s)

    def text(self, x, y, s, size=13, weight="normal", fill=INK, anchor="middle",
             italic=False, mono=False, opacity=1.0, halo=None) -> None:
        attrs = ['x="%.1f"' % x, 'y="%.1f"' % y, 'font-size="%.1f"' % size,
                 'fill="%s"' % fill, 'text-anchor="%s"' % anchor]
        if mono:
            attrs.append('font-family="%s"' % MONO)
        if weight != "normal":
            attrs.append('font-weight="%s"' % weight)
        if italic:
            attrs.append('font-style="italic"')
        if opacity < 1:
            attrs.append('fill-opacity="%.2f"' % opacity)
        if halo:
            attrs.append('stroke="%s" stroke-width="4" stroke-linejoin="round" '
                         'paint-order="stroke"' % halo)
        self.items.append("<text %s>%s</text>" % (" ".join(attrs), escape(s)))

    def rect(self, x, y, w, h, fill, stroke=None, rx=12, sw=1.2, shadow=False,
             dash=None, opacity=1.0) -> Box:
        attrs = ['x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%.1f"' % (x, y, w, h, rx),
                 'fill="%s"' % fill]
        if stroke:
            attrs.append('stroke="%s" stroke-width="%.1f"' % (stroke, sw))
        if dash:
            attrs.append('stroke-dasharray="%s"' % dash)
        if shadow:
            attrs.append('filter="url(#shadow)"')
        if opacity < 1:
            attrs.append('opacity="%.2f"' % opacity)
        self.items.append("<rect %s/>" % " ".join(attrs))
        return Box(x, y, w, h)

    # -- composite shapes ------------------------------------------------------

    def _text_block(self, cx, cy, title, role, lines, color, title_size, line_size,
                    mono_lines=False, title_fill=None, line_fill=None) -> None:
        tcol = title_fill or PALETTE.get(color, (None, None, None, INK))[3]
        lcol = line_fill or tcol
        rows = []
        if title:
            for t in title.split("\n"):
                rows.append(("title", t, title_size))
        if role:
            rows.append(("role", role, line_size - 0.5))
        for ln in lines:
            rows.append(("line", ln, line_size))
        gaps = {"title": 1.22, "role": 1.38, "line": 1.36}
        heights = [s * gaps[k] for k, _, s in rows]
        total = sum(heights)
        y = cy - total / 2
        for (kind, t, size), hgt in zip(rows, heights):
            baseline = y + hgt / 2 + size * 0.36
            if kind == "title":
                self.text(cx, baseline, t, size=size, weight="bold", fill=tcol)
            elif kind == "role":
                self.text(cx, baseline, t, size=size, italic=True, fill=lcol, opacity=0.9)
            else:
                self.text(cx, baseline, t, size=size, fill=lcol, mono=mono_lines,
                          opacity=0.95 if lcol == "#FFFFFF" else 1)
            y += hgt

    def node(self, x, y, w, h, title, role=None, lines=(), color="green", rx=12,
             title_size=15, line_size=12.5, mono_lines=False) -> Box:
        _, _, stroke, _ = PALETTE[color]
        self.rect(x, y, w, h, self._grad(color), stroke, rx=rx, sw=1.2, shadow=True)
        self._text_block(x + w / 2, y + h / 2, title, role, list(lines), color,
                         title_size, line_size, mono_lines)
        return Box(x, y, w, h)

    def outline(self, x, y, w, h, title, lines=(), color="blue", rx=10, title_size=13.5,
                line_size=12, mono_lines=False, dash=None, fill=None) -> Box:
        lfill, stroke, tcol = LIGHT[color]
        self.rect(x, y, w, h, fill or SURFACE, stroke, rx=rx, sw=1.6, shadow=True, dash=dash)
        self._text_block(x + w / 2, y + h / 2, title, None, list(lines), color, title_size,
                         line_size, mono_lines, title_fill=tcol, line_fill=INK)
        return Box(x, y, w, h)

    def oval(self, cx, cy, w, h, title, sub=None, color="grey", outline=False) -> Box:
        x, y = cx - w / 2, cy - h / 2
        if outline:
            self.items.append('<ellipse cx="%.1f" cy="%.1f" rx="%.1f" ry="%.1f" fill="%s" '
                              'stroke="%s" stroke-width="2" filter="url(#shadow)"/>'
                              % (cx, cy, w / 2, h / 2, SURFACE, INK))
            self._text_block(cx, cy, title, None, [sub] if sub else [], "white", 15, 12,
                             title_fill=INK, line_fill=MUTED)
        else:
            _, _, stroke, tcol = PALETTE[color]
            self.items.append('<ellipse cx="%.1f" cy="%.1f" rx="%.1f" ry="%.1f" fill="%s" '
                              'stroke="%s" stroke-width="1.2" filter="url(#shadow)"/>'
                              % (cx, cy, w / 2, h / 2, self._grad(color), stroke))
            self._text_block(cx, cy, title, None, [sub] if sub else [], color, 13.5, 11.5)
        return Box(x, y, w, h)

    def pill(self, cx, cy, text, color="navy", h=34, size=13, padx=18, mono=False,
             light=False, weight="bold") -> Box:
        w = text_width(text, size, weight, mono=mono) + 2 * padx
        x, y = cx - w / 2, cy - h / 2
        if light:
            fill, stroke, tcol = LIGHT[color]
            self.rect(x, y, w, h, fill, stroke, rx=h / 2, sw=1.4, shadow=True)
        else:
            _, _, stroke, tcol = PALETTE[color]
            self.rect(x, y, w, h, self._grad(color), stroke, rx=h / 2, shadow=True)
        self.text(cx, cy + size * 0.36, text, size=size, weight=weight, fill=tcol, mono=mono)
        return Box(x, y, w, h)

    def diamond(self, cx, cy, w, h, text, color="amber", size=12.5) -> Box:
        _, _, stroke, tcol = PALETTE[color]
        pts = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)]
        self.items.append('<polygon points="%s" fill="%s" stroke="%s" stroke-width="1.2" '
                          'stroke-linejoin="round" filter="url(#shadow)"/>'
                          % (" ".join("%.1f,%.1f" % p for p in pts), self._grad(color), stroke))
        self._text_block(cx, cy, None, None, text.split("\n"), color, 13, size,
                         line_fill=tcol)
        return Box(cx - w / 2, cy - h / 2, w, h)

    def hexagon(self, cx, cy, w, h, title, lines=(), color="navy") -> Box:
        _, _, stroke, _ = PALETTE[color]
        k = h * 0.36
        pts = [(cx - w / 2 + k, cy - h / 2), (cx + w / 2 - k, cy - h / 2), (cx + w / 2, cy),
               (cx + w / 2 - k, cy + h / 2), (cx - w / 2 + k, cy + h / 2), (cx - w / 2, cy)]
        self.items.append('<polygon points="%s" fill="%s" stroke="%s" stroke-width="1.2" '
                          'stroke-linejoin="round" filter="url(#shadow)"/>'
                          % (" ".join("%.1f,%.1f" % p for p in pts), self._grad(color), stroke))
        self._text_block(cx, cy, title, None, list(lines), color, 15, 12.5)
        return Box(cx - w / 2, cy - h / 2, w, h)

    def bubble(self, x, y, w, h, lines, color="white", tail="bottom-left", italic=False,
               size=12.5, title=None) -> Box:
        """A speech bubble with a small tail; used for spoken lines."""
        fill, stroke, tcol = LIGHT[color]
        tx = {"bottom-left": x + 26, "bottom-right": x + w - 26}.get(tail, x + w / 2)
        r = 12
        d = ("M%.1f,%.1f H%.1f Q%.1f,%.1f %.1f,%.1f V%.1f Q%.1f,%.1f %.1f,%.1f H%.1f L%.1f,%.1f "
             "L%.1f,%.1f H%.1f Q%.1f,%.1f %.1f,%.1f V%.1f Q%.1f,%.1f %.1f,%.1f Z") % (
            x + r, y, x + w - r, x + w, y, x + w, y + r, y + h - r, x + w, y + h, x + w - r, y + h,
            tx + 8, tx, y + h + 11, tx - 4, y + h, x + r, x, y + h, x, y + h - r, y + r, x, y,
            x + r, y)
        self.items.append('<path d="%s" fill="%s" stroke="%s" stroke-width="1.6" '
                          'stroke-linejoin="round" filter="url(#shadow)"/>' % (d, fill, stroke))
        rows = list(lines)
        total = len(rows) * size * 1.36 + (16 if title else 0)
        yy = y + h / 2 - total / 2
        if title:
            self.text(x + w / 2, yy + 11, title, size=11, weight="bold", fill=tcol)
            yy += 16
        for ln in rows:
            self.text(x + w / 2, yy + size * 1.36 / 2 + size * 0.36, ln, size=size, fill=INK,
                      italic=italic)
            yy += size * 1.36
        return Box(x, y, w, h)

    def container(self, x, y, w, h, label=None, sub=None, color="green", rx=22,
                  dash=None, label_size=15) -> Box:
        fill, stroke, tcol = LIGHT[color]
        self.rect(x, y, w, h, fill, stroke, rx=rx, sw=1.8, dash=dash)
        if label:
            self.text(x + 22, y + 28, label, size=label_size, weight="bold", fill=tcol,
                      anchor="start")
        if sub:
            self.text(x + 22, y + 47, sub, size=12, fill=MUTED, anchor="start")
        return Box(x, y, w, h)

    def badge(self, x, y, text, color="navy", size=10.5) -> Box:
        """Small pill tag pinned to a node's edge, ringed in the page colour."""
        w = text_width(text, size, "bold") + 14
        h = size + 9
        self.rect(x - w / 2, y - h / 2, w, h, self._grad(color), HALO, rx=h / 2, sw=2,
                  shadow=True)
        self.text(x, y + size * 0.36, text, size=size, weight="bold", fill=PALETTE[color][3])
        return Box(x - w / 2, y - h / 2, w, h)

    def number(self, cx, cy, n, color="navy", r=11) -> None:
        self.items.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" stroke="%s" '
                          'stroke-width="2"/>' % (cx, cy, r, PALETTE[color][1], HALO))
        self.text(cx, cy + 4, str(n), size=11.5, weight="bold", fill="#FFFFFF")

    def edge(self, pts, color=LINE, width=2.0, dash=False, label=None, lpos=None,
             lanchor="middle", lcolor=None, head=True, tail=False, r=12, lsize=12,
             litalic=False, lmono=False) -> None:
        attrs = ['d="%s"' % _rounded_path(pts, r), 'fill="none"', 'stroke="%s"' % color,
                 'stroke-width="%.1f"' % width, 'stroke-linecap="round"', 'stroke-linejoin="round"']
        if dash:
            attrs.append('stroke-dasharray="%s"' % ("7 5" if dash is True else dash))
        if head:
            attrs.append('marker-end="%s"' % self._marker(color))
        if tail:
            attrs.append('marker-start="%s"' % self._marker(color))
        self.items.append("<path %s/>" % " ".join(attrs))
        if label:
            if lpos is None:
                (x1, y1), (x2, y2) = pts[0], pts[1]
                lpos = ((x1 + x2) / 2, (y1 + y2) / 2 - 7)
            self.text(lpos[0], lpos[1], label, size=lsize, fill=lcolor or color, anchor=lanchor,
                      halo=HALO, italic=litalic, mono=lmono)

    def line(self, x1, y1, x2, y2, color=FAINT, width=1.2, dash=None) -> None:
        self.items.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
                          'stroke-width="%.1f"%s/>' % (x1, y1, x2, y2, color, width,
                                                      ' stroke-dasharray="%s"' % dash if dash else ""))

    def legend(self, x, y, items, size=12) -> None:
        """items: [(colour name, label)] drawn as small swatches in a row."""
        for color, label in items:
            if color in PALETTE:
                self.rect(x, y - 10, 14, 14, self._grad(color), PALETTE[color][2], rx=4)
            else:
                self.rect(x, y - 10, 14, 14, LIGHT[color][0], LIGHT[color][1], rx=4, sw=1.4)
            self.text(x + 20, y + 1.5, label, size=size, fill=MUTED, anchor="start")
            x += 20 + text_width(label, size) + 22

    def check(self, cx, cy, size=14, color="#FFFFFF", width=3.0) -> None:
        k = size / 2
        self.items.append('<path d="M%.1f,%.1f L%.1f,%.1f L%.1f,%.1f" fill="none" stroke="%s" '
                          'stroke-width="%.1f" stroke-linecap="round" stroke-linejoin="round"/>'
                          % (cx - k, cy, cx - k * 0.25, cy + k * 0.7, cx + k, cy - k * 0.75,
                             color, width))

    def cross(self, cx, cy, size=12, color="#FFFFFF", width=3.0) -> None:
        k = size / 2
        self.items.append('<path d="M%.1f,%.1f L%.1f,%.1f M%.1f,%.1f L%.1f,%.1f" fill="none" '
                          'stroke="%s" stroke-width="%.1f" stroke-linecap="round"/>'
                          % (cx - k, cy - k, cx + k, cy + k, cx + k, cy - k, cx - k, cy + k,
                             color, width))

    def tile(self, x, y, ok: bool, size=34) -> Box:
        """A detection tile: green with a tick, or light grey with a cross."""
        if ok:
            self.rect(x, y, size, size, self._grad("green"), PALETTE["green"][2], rx=8, shadow=True)
            self.check(x + size / 2, y + size / 2, size * 0.45)
        else:
            self.rect(x, y, size, size, "#EEF0F2", FAINT, rx=8, sw=1.4)
            self.cross(x + size / 2, y + size / 2, size * 0.32, color="#A0A6AD", width=2.6)
        return Box(x, y, size, size)

    def fill(self, color: str) -> str:
        """Gradient fill reference for a palette colour."""
        return self._grad(color)

    def frame(self, x, y, w, h, tag, note=None, color="grey") -> Box:
        """A sequence-diagram frame (loop / alt) with a tag in the corner."""
        fill, stroke, tcol = LIGHT[color]
        self.rect(x, y, w, h, "none", stroke, rx=10, sw=1.4, dash="5 4")
        tw = text_width(tag, 11.5, "bold") + 16
        self.items.append('<path d="M%.1f,%.1f H%.1f V%.1f L%.1f,%.1f H%.1f V%.1f Q%.1f,%.1f %.1f,%.1f Z" '
                          'fill="%s" stroke="%s" stroke-width="1.4"/>'
                          % (x + 10, y, x + tw, y + 14, x + tw - 8, y + 22, x, y + 10,
                             x, y, x + 10, y, fill, stroke))
        self.text(x + tw / 2 - 2, y + 15.5, tag, size=11.5, weight="bold", fill=tcol)
        if note:
            self.text(x + tw + 10, y + 15.5, note, size=11.5, fill=MUTED, italic=True,
                      anchor="start")
        return Box(x, y, w, h)

    def image(self, x, y, w, h, path, rx=6) -> None:
        with open(path, "rb") as fh:
            data = base64.b64encode(fh.read()).decode("ascii")
        cid = "clip%d" % len(self.defs)
        self.defs[cid] = ('<clipPath id="%s"><rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                          'rx="%.1f"/></clipPath>' % (cid, x, y, w, h, rx))
        self.items.append('<image x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                          'preserveAspectRatio="xMidYMid slice" clip-path="url(#%s)" '
                          'href="data:image/jpeg;base64,%s"/>' % (x, y, w, h, cid, data))

    # -- output ----------------------------------------------------------------

    def save(self, path: str) -> None:
        """Writes ``path`` (light) and ``path`` with a ``-dark`` suffix."""
        body = "\n".join(self.items)
        defs = "".join(self.defs.values())
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" '
               'font-family="%s">\n<title>%s</title>\n<defs>%s</defs>\n%s\n</svg>\n') % (
            self.w, self.h, self.w, self.h, FONT, escape(self.title), defs, body)
        dark = svg
        for light, swap in DARK.items():
            dark = dark.replace('"%s"' % light, '"%s"' % swap)
        for out, text in ((path, svg), (path[:-4] + "-dark.svg", dark)):
            with open(out, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
