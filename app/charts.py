"""Graphiques de l'application : du HTML/SVG écrit à la main, animé en CSS, sans dépendance.

Chaque fonction renvoie une chaîne à passer à `st.html`. Les couleurs viennent des variables CSS
du thème (--green, --amber, --red...), donc tout suit le mode clair / sombre.
"""
from __future__ import annotations

import html
import math
import re
from typing import Optional, Sequence

import numpy as np

CHARTS_CSS = """
/* compteurs : le chiffre monte de 0 à sa valeur (la valeur finale est la valeur de base, sans animation) */
@property --n { syntax: "<integer>"; initial-value: 0; inherits: false; }
@keyframes count { from { --n: 0; } }
.cnt { --n: var(--to); animation: count 1.2s cubic-bezier(.2,.7,.2,1) both; counter-reset: cnt var(--n); }
.cnt::after { content: counter(cnt) attr(data-suf); }

@keyframes grow { from { transform: scaleY(0); } }
@keyframes wid { from { width: 0; } }
@keyframes arc { from { stroke-dashoffset: var(--len); } }
@keyframes seg { from { stroke-dasharray: 0 100; } }
@keyframes fade { from { opacity: 0; transform: scale(.7); } }

.panel { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: .9rem;
  animation: rise .55s cubic-bezier(.2,.7,.2,1) both; }
.panel h4 { margin: 0 0 .7rem; font-size: .78rem; letter-spacing: .12em; text-transform: uppercase; color: var(--green); }
.panel .note { font-size: .8rem; color: var(--muted); margin: .5rem 0 0; }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.2rem; align-items: center; }

/* carte de confiance */
.heat { display: grid; grid-template-columns: repeat(9, minmax(0, 1fr)); gap: 3px; max-width: 380px; }
.heat div { aspect-ratio: 1; display: grid; place-items: center; border-radius: 6px; color: var(--ink);
  font-family: var(--serif); font-size: clamp(.8rem, 2vw, 1.05rem); animation: fade .45s cubic-bezier(.3,1.3,.5,1) both;
  animation-delay: calc(var(--i) * 12ms); }
.heat .t3 { background: color-mix(in srgb, var(--green) 40%, var(--card)); }
.heat .t2 { background: color-mix(in srgb, var(--green) 26%, var(--card)); }
.heat .t1 { background: color-mix(in srgb, var(--green) 13%, var(--card)); }
.heat .lo { background: var(--amber); }
.heat .bad { background: var(--red); }
.heat-legend { display: flex; align-items: center; gap: .5rem; margin-top: .6rem; font-size: .78rem; color: var(--muted); }
.heat-legend i { width: 7rem; height: .5rem; border-radius: 999px; background: linear-gradient(90deg, var(--red), var(--amber), var(--green)); }

/* histogramme */
.hist { width: 100%; max-width: 420px; display: block; }
.hist rect { transform-box: fill-box; transform-origin: bottom; animation: grow .7s cubic-bezier(.2,.8,.2,1) both; animation-delay: calc(var(--i) * 55ms); }
.hist .good { fill: var(--green); } .hist .mid { fill: var(--amber); } .hist .low { fill: var(--red); }
.hist text { fill: var(--muted); font-size: 9px; text-anchor: middle; font-family: sans-serif; }
.hist text.v { fill: var(--ink); font-weight: 600; }

/* jauge et donut */
.gauge, .donut { display: block; margin: 0 auto; max-width: 190px; width: 100%; overflow: visible; }
.gauge .bg, .donut .bg { fill: none; stroke: var(--line); stroke-width: 9; }
.gauge .fg { fill: none; stroke: var(--green); stroke-width: 9; stroke-linecap: round; stroke-dasharray: var(--len);
  animation: arc 1.3s cubic-bezier(.2,.8,.2,1) both; }
.gauge text, .donut text { text-anchor: middle; fill: var(--ink); font-family: var(--serif); }
.gauge .big { font-size: 26px; } .gauge .small { font-size: 8px; fill: var(--muted); font-family: sans-serif; letter-spacing: .08em; text-transform: uppercase; }
.donut .seg { fill: none; stroke-width: 6; animation: seg 1.1s cubic-bezier(.2,.8,.2,1) both; }
.donut .big { font-size: 9px; } .donut .small { font-size: 2.6px; fill: var(--muted); font-family: sans-serif; }
.legend-list { list-style: none; padding: 0; margin: .6rem 0 0; font-size: .85rem; color: var(--ink); }
.legend-list li { display: flex; align-items: center; gap: .5rem; margin: .2rem 0; }
.legend-list i { width: .7rem; height: .7rem; border-radius: 3px; flex: none; }
.legend-list span { margin-left: auto; color: var(--muted); font-variant-numeric: tabular-nums; }

/* barre des temps */
.stack { display: flex; height: 1.6rem; border-radius: 999px; overflow: hidden; background: var(--line); gap: 2px; }
.stack i { width: var(--w); min-width: 4px; animation: wid 1s cubic-bezier(.2,.8,.2,1) both; animation-delay: calc(var(--i) * 120ms); }
.c0 { background: var(--blue); } .c1 { background: var(--green); } .c2 { background: var(--amber); } .c3 { background: var(--rose); }
.c4 { background: var(--muted); }

/* cartes des modèles */
.model { display: flex; flex-wrap: wrap; align-items: baseline; gap: .6rem; margin-bottom: .8rem; }
.model b { font-family: var(--serif); font-size: 1.25rem; }
.model span { font-size: .75rem; padding: .12rem .6rem; border-radius: 999px; background: var(--green-soft); color: var(--green); }
.mrow { display: grid; grid-template-columns: 11rem 1fr 4.2rem; align-items: center; gap: .7rem; margin: .45rem 0; font-size: .88rem; }
.mrow .bar { height: .55rem; border-radius: 999px; background: var(--line); overflow: hidden; }
.mrow .bar i { display: block; height: 100%; width: var(--w); background: var(--green); border-radius: 999px;
  animation: wid 1.1s cubic-bezier(.2,.8,.2,1) both; animation-delay: calc(var(--i) * 90ms); }
.mrow em { font-style: normal; text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
@media (max-width: 640px) { .mrow { grid-template-columns: 7rem 1fr 3.6rem; } }

/* onglets natifs */
[data-baseweb="tab"] { color: var(--muted); }
[data-baseweb="tab"][aria-selected="true"] { color: var(--green); }
[data-baseweb="tab-highlight"] { background-color: var(--green); }
"""

HALF_CIRCLE = math.pi * 50  # longueur de l'arc de la jauge (rayon 50)


def counter(text: str) -> str:
    """« 1564 ms » -> un compteur animé qui monte jusqu'à 1564, suivi de « ms ». Texte sans chiffre : inchangé."""
    m = re.match(r"^(\d+)(.*)$", text)
    if not m:
        return html.escape(text)
    return (f'<span class="cnt" style="--to:{m[1]}" data-suf="{html.escape(m[2])}" '
            f'aria-label="{html.escape(text)}"></span>')


def panel(title: str, body: str, note: str = "") -> str:
    head = f"<h4>{title}</h4>" if title else ""
    foot = f'<p class="note">{note}</p>' if note else ""
    return f'<div class="panel">{head}{body}{foot}</div>'


def _tier(conf: float) -> str:
    return "t3" if conf >= 0.98 else "t2" if conf >= 0.9 else "t1" if conf >= 0.8 else "lo" if conf >= 0.5 else "bad"


def heatmap_html(confidence: np.ndarray, grid: Sequence[Sequence[int]]) -> str:
    """9×9 : chaque case colorée selon la confiance du classifieur (les cases vides comptent aussi)."""
    cells = []
    for r in range(9):
        for c in range(9):
            conf = float(confidence[r, c])
            cells.append(f'<div class="{_tier(conf)}" style="--i:{r * 9 + c}" '
                         f'title="ligne {r + 1}, colonne {c + 1} : {conf:.1%}">{grid[r][c] or "·"}</div>')
    return (f'<div class="heat">{"".join(cells)}</div>'
            '<div class="heat-legend"><span>peu sûr</span><i></i><span>très sûr</span></div>')


HIST_EDGES = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.995, 1.0001]
HIST_LABELS = ["<50", "60", "70", "80", "85", "90", "95", "98", "99,5", "100"]


def histogram_svg(confidence: np.ndarray) -> str:
    """Répartition des 81 confiances. Hauteur en racine carrée : sinon les rares cases douteuses disparaissent."""
    counts, _ = np.histogram(np.asarray(confidence).ravel(), bins=HIST_EDGES)
    top = max(int(counts.max()), 1)
    bars = []
    for k, n in enumerate(counts):
        h = 70 * math.sqrt(n / top)
        x = 8 + k * 31
        tone = "low" if k < 4 else "mid" if k < 6 else "good"
        bars.append(f'<rect class="{tone}" style="--i:{k}" x="{x}" y="{82 - h:.1f}" width="24" height="{max(h, 1.5):.1f}" rx="3"/>'
                    f'<text class="v" x="{x + 12}" y="{78 - h:.1f}">{n if n else ""}</text>'
                    f'<text x="{x + 12}" y="95">{HIST_LABELS[k]}</text>')
    return (f'<svg class="hist" viewBox="0 0 330 104" role="img" aria-label="Répartition des confiances">'
            f'{"".join(bars)}</svg>')


def gauge_svg(value: float, label: str) -> str:
    """Demi-cercle rempli à `value` (0 à 1), avec la valeur au centre."""
    value = min(max(value, 0.0), 1.0)
    offset = HALF_CIRCLE * (1 - value)
    path = "M 10 60 A 50 50 0 0 1 110 60"
    return (f'<svg class="gauge" viewBox="0 0 120 72" role="img" aria-label="{html.escape(label)} : {value:.0%}">'
            f'<path class="bg" d="{path}"/>'
            f'<path class="fg" d="{path}" style="--len:{HALF_CIRCLE:.2f};stroke-dashoffset:{offset:.2f}"/>'
            f'<text class="big" x="60" y="52">{counter(f"{value * 100:.0f}%")}</text>'
            f'<text class="small" x="60" y="68">{html.escape(label)}</text></svg>')


def donut_html(parts: list[tuple[str, float, str]], center: str, unit: str) -> str:
    """Anneau en parts : (légende, valeur, classe de couleur c0..c4), avec la légende dessous."""
    total = sum(v for _, v, _ in parts) or 1
    palette = {"c0": "var(--blue)", "c1": "var(--green)", "c2": "var(--amber)", "c3": "var(--rose)", "c4": "var(--muted)"}
    arcs, items, start = [], [], 0.0
    for label, v, cls in parts:
        share = 100 * v / total
        arcs.append(f'<circle class="seg" cx="21" cy="21" r="15.9155" style="stroke:{palette[cls]};'
                    f'stroke-dasharray:{share:.2f} {100 - share:.2f};stroke-dashoffset:{-start:.2f}"/>')
        items.append(f'<li><i class="{cls}"></i>{html.escape(label)}<span>{v:g}</span></li>')
        start += share
    svg = (f'<svg class="donut" viewBox="0 0 42 42" role="img"><circle class="bg" cx="21" cy="21" r="15.9155" style="stroke-width:6"/>'
           f'<g transform="rotate(-90 21 21)">{"".join(arcs)}</g>'
           f'<text class="big" x="21" y="22.5">{html.escape(center)}</text>'
           f'<text class="small" x="21" y="27">{html.escape(unit)}</text></svg>')
    return svg + f'<ul class="legend-list">{"".join(items)}</ul>'


def stack_html(steps: list[tuple[str, float]]) -> str:
    """Barre empilée des durées (secondes) : (nom de l'étape, durée)."""
    steps = [(n, t) for n, t in steps if t and t > 0]
    total = sum(t for _, t in steps) or 1
    segs = "".join(f'<i class="c{k % 5}" style="--w:{100 * t / total:.2f}%;--i:{k}" title="{n} : {t * 1000:.0f} ms"></i>'
                   for k, (n, t) in enumerate(steps))
    legend = "".join(f'<li><i class="c{k % 5}"></i>{html.escape(n)}<span>{t * 1000:.0f} ms · {t / total:.0%}</span></li>'
                     for k, (n, t) in enumerate(steps))
    return f'<div class="stack">{segs}</div><ul class="legend-list">{legend}</ul>'


def model_card_html(title: str, tag: str, rows: list[tuple[str, Optional[float]]], note: str = "") -> str:
    """Un modèle et ses mesures : (nom de la mesure, valeur entre 0 et 1 ou None si absente)."""
    lines = []
    for k, (label, v) in enumerate(rows):
        if v is None:
            continue
        lines.append(f'<div class="mrow"><span>{html.escape(label)}</span>'
                     f'<div class="bar"><i style="--w:{min(max(v, 0), 1) * 100:.1f}%;--i:{k}"></i></div><em>{v:.1%}</em></div>')
    body = "".join(lines) or '<p class="note">Aucune mesure trouvée dans MLflow pour ce modèle.</p>'
    head = f'<div class="model"><b>{html.escape(title)}</b><span>{html.escape(tag)}</span></div>'
    return panel("", head + body, note)
