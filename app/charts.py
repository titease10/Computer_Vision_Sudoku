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

CHARTS_CSS = r"""
/* compteurs : le chiffre monte de 0 à sa valeur (la valeur finale est la valeur de base, sans animation) */
@property --n { syntax: "\3C integer\3E"; initial-value: 0; inherits: false; }
@keyframes count { from { --n: 0; } }
.cnt { --n: var(--to); animation: count 1.2s cubic-bezier(.2,.7,.2,1) both; counter-reset: cnt var(--n); }
.cnt::after { content: counter(cnt) attr(data-suf); }

@keyframes grow { from { transform: scaleY(0); } }
@keyframes wid { from { width: 0; } }
/* anneaux : --p (jauge, en %) et --k (donut, de 0 à 1) sont animés, d'où leur déclaration typée */
@property --p { syntax: "\3C number\3E"; initial-value: 0; inherits: false; }
@property --k { syntax: "\3C number\3E"; initial-value: 1; inherits: false; }
@keyframes fillp { from { --p: 0; } }
@keyframes fillk { from { --k: 0; } }
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
.hist { display: flex; align-items: flex-end; gap: 6px; height: 130px; width: 100%; max-width: 420px; }
.hist > div { flex: 1; height: 100%; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; min-width: 0; }
.hist .v { font-size: .72rem; font-weight: 600; color: var(--ink); height: 1rem; }
.hist .bar { width: 100%; height: calc(var(--h) * 1%); min-height: 2px; border-radius: 4px 4px 0 0; transform-origin: bottom;
  animation: grow .8s cubic-bezier(.2,.8,.2,1) both; animation-delay: calc(var(--i) * 55ms); }
.hist .good { background: var(--green); } .hist .mid { background: var(--amber); } .hist .low { background: var(--red); }
.hist .l { font-size: .62rem; color: var(--muted); margin-top: .25rem; height: .9rem; }

/* jauge et donut : un disque en conic-gradient, percé par un disque de la couleur de la carte */
.ring { position: relative; width: min(100%, 170px); aspect-ratio: 1; border-radius: 50%; margin: 0 auto; display: grid; place-items: center;
  text-align: center; }
.ring::before { content: ""; position: absolute; inset: 11%; border-radius: 50%; background: var(--card); }
.ring > div { position: relative; font-family: var(--serif); font-size: 1.9rem; line-height: 1.05; color: var(--ink); }
.ring > div small { display: block; font-family: sans-serif; font-size: .62rem; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin-top: .2rem; }
.gauge { background: conic-gradient(var(--green) calc(var(--p) * 1%), var(--line) 0); animation: fillp 1.4s cubic-bezier(.2,.8,.2,1) both; }
.donut { animation: fillk 1.3s cubic-bezier(.2,.8,.2,1) both; }
.ring-label { display: block; text-align: center; margin: .6rem 0 .8rem; font-size: .72rem; letter-spacing: .08em;
  text-transform: uppercase; color: var(--muted); }
.legend-list { list-style: none; padding: 0; margin: .6rem 0 0; font-size: .85rem; color: var(--ink); }
.legend-list li { display: flex; align-items: center; gap: .5rem; margin: .2rem 0; }
.legend-list i { width: .7rem; height: .7rem; border-radius: 3px; flex: none; }
.legend-list span { margin-left: auto; color: var(--muted); font-variant-numeric: tabular-nums; }

/* barre des temps */
.stack { display: flex; height: 1.6rem; border-radius: 999px; background: var(--line); gap: 2px; }
.stack i:first-child { border-radius: 999px 4px 4px 999px; } .stack i:last-child { border-radius: 4px 999px 999px 4px; }
.stack i:only-child { border-radius: 999px; }
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

/* avertissement : grille corrigée */
@keyframes wobble { 0%, 100% { transform: rotate(0); } 20% { transform: rotate(-12deg); } 40% { transform: rotate(10deg); }
  60% { transform: rotate(-6deg); } 80% { transform: rotate(4deg); } }
.warn { display: flex; gap: 1.1rem; align-items: flex-start; background: var(--amber); color: var(--ink);
  border: 1px solid #E0A800; border-left: 10px solid #E0A800; border-radius: 14px; padding: 1.2rem 1.4rem; margin: 0 0 1.3rem;
  animation: rise .55s cubic-bezier(.2,.7,.2,1) both; }
.warn .icon { font-size: 2.4rem; line-height: 1; animation: wobble 1.2s ease .5s 2 both; transform-origin: 50% 80%; }
.warn h3 { margin: 0 0 .35rem; font-family: var(--serif); font-size: 1.5rem; color: var(--ink); }
.warn p { margin: 0; font-size: .95rem; }
.warn ul { margin: .6rem 0 0; padding-left: 1.2rem; font-size: .92rem; }
.warn li b { font-variant-numeric: tabular-nums; }

/* infobulles maison : data-tip remplace l'attribut title, dont l'infobulle est celle du navigateur */
[data-tip] { position: relative; }
[data-tip]::before, [data-tip]::after { position: absolute; left: 50%; pointer-events: none; opacity: 0; z-index: 30;
  transition: opacity .14s ease, transform .14s ease; }
[data-tip]::after { content: attr(data-tip); bottom: calc(100% + 9px); transform: translate(-50%, 5px); white-space: nowrap;
  background: var(--ink); color: var(--paper); font: 500 .74rem/1.3 sans-serif; letter-spacing: 0; text-transform: none;
  padding: .4rem .65rem; border-radius: 8px; box-shadow: 0 6px 18px var(--shadow); }
[data-tip]::before { content: ""; bottom: calc(100% + 3px); transform: translate(-50%, 5px);
  border: 6px solid transparent; border-top-color: var(--ink); }
[data-tip]:hover::before, [data-tip]:hover::after { opacity: 1; transform: translate(-50%, 0); }

/* la grille : neuf blocs, chaque case est une petite tuile arrondie */
.board { position: relative; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 7px; padding: 8px;
  box-sizing: border-box; width: 100%; max-width: 440px; border-radius: 18px; border: 1px solid var(--line);
  background: linear-gradient(145deg, var(--line), var(--card)); box-shadow: 0 10px 30px var(--shadow);
  font-family: var(--serif); font-size: clamp(1rem, 4.2vw, 1.5rem); font-variant-numeric: tabular-nums; }
.board .blk { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 3px; padding: 3px; border-radius: 12px;
  background: var(--card); }
.board .cell { position: relative; aspect-ratio: 1; display: grid; place-items: center; border-radius: 8px; color: var(--ink);
  background: color-mix(in srgb, var(--line) 28%, var(--card)); cursor: default;
  transition: transform .18s cubic-bezier(.3,1.5,.5,1), box-shadow .18s ease; }
.board .cell:hover { transform: scale(1.16); z-index: 6; box-shadow: 0 8px 18px var(--shadow); }
.board .found { color: var(--green); font-weight: 700; background: color-mix(in srgb, var(--green) 11%, var(--card)); }
.board .doubt { background: var(--amber); }
.board .fixed { background: var(--rose); }
.board .try { background: var(--blue); }
.board .undo { background: var(--red); }

/* révélation de la grille résolue : les chiffres éclosent en vague diagonale, puis un reflet balaie la grille */
@keyframes reveal { 0% { opacity: 0; transform: scale(.2) rotate(-14deg); box-shadow: 0 0 0 0 rgba(60,190,110,.75); }
  55% { opacity: 1; transform: scale(1.28) rotate(3deg); box-shadow: 0 0 0 11px rgba(60,190,110,0); }
  100% { opacity: 1; transform: none; box-shadow: 0 0 0 0 rgba(60,190,110,0); } }
@keyframes shine { 0% { background-position: 130% 0; opacity: 1; } 85% { opacity: 1; } 100% { background-position: -30% 0; opacity: 0; } }
@keyframes glow { 40% { box-shadow: 0 0 0 7px color-mix(in srgb, var(--green) 38%, transparent), 0 10px 40px var(--shadow); } }
.board.pop .found { animation: reveal .75s cubic-bezier(.3,1.3,.5,1) var(--d) backwards; }
.board.pop { animation: glow 1.5s ease-out var(--end) backwards; }
.board.pop::after { content: ""; position: absolute; inset: 0; border-radius: inherit; pointer-events: none;
  background: linear-gradient(115deg, transparent 35%, rgba(255,255,255,.3) 50%, transparent 65%); background-size: 300% 100%;
  animation: shine 1.3s ease-out var(--end) both; }

/* onglets natifs */
[data-baseweb="tab"] { color: var(--muted); }
[data-baseweb="tab"][aria-selected="true"] { color: var(--green); }
[data-baseweb="tab-highlight"] { background-color: var(--green); }
"""

def _peer_css() -> str:
    """Au survol d'une case, sa ligne, sa colonne et son bloc se teintent (les 27 groupes de la grille)."""
    groups = [f"{kind}{i}" for kind in "rcb" for i in range(9)]
    selectors = ",\n".join(f".board:has(.{g}:hover) .{g}:not(:hover)" for g in groups)
    return selectors + " { box-shadow: inset 0 0 0 100px color-mix(in srgb, var(--green) 15%, transparent); }\n"


CHARTS_CSS += _peer_css()


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


def repair_banner_html(repaired: Sequence[tuple[int, int, int, int]]) -> str:
    """Gros avertissement : la lecture n'était pas un sudoku valide et des cases ont été changées.

    `repaired` : (ligne, colonne, valeur lue, valeur retenue), numérotées depuis 0 ; 0 = case vide.
    """
    if not repaired:
        return ""
    shown = lambda v: str(v) if v else "empty"
    items = "".join(f"<li>row {r + 1}, column {c + 1}: <b>{shown(a)}</b> → <b>{shown(b)}</b></li>" for r, c, a, b in repaired)
    n = len(repaired)
    return ('<div class="warn" role="alert"><div class="icon">⚠️</div><div>'
            f"<h3>This grid was corrected: the photo did not read as a valid sudoku</h3>"
            "<p>As read, the grid was impossible: a digit appeared twice in a row, column or block, or the grid did not "
            f"have exactly one solution. To make it solvable, {n} cell{'s were' if n > 1 else ' was'} changed to the "
            "classifier’s next best guess. <b>Check the cells below against your paper grid</b>: "
            "they are highlighted in orange.</p>"
            f"<ul>{items}</ul></div></div>")


def board_html(grid: Sequence[Sequence[int]], solution, doubtful: set, fixed: set, marks: Optional[dict] = None,
               pop: bool = False) -> str:
    """La grille dessinée : chiffres lus en noir, chiffres trouvés en vert, cases à vérifier surlignées.

    `marks` ({(ligne, col): classe CSS}) surligne des cases pendant les animations ;
    `pop` fait éclore les chiffres trouvés en vague diagonale, puis passe un reflet sur la grille.
    """
    marks = marks or {}
    found = [(r, c) for r in range(9) for c in range(9) if not grid[r][c] and solution and solution[r][c]]
    delay = {rc: k for k, rc in enumerate(sorted(found, key=lambda rc: (rc[0] + rc[1], rc[0])))}
    step = min(45.0, 1600 / max(len(found), 1))  # la révélation entière dure au plus ~1,6 s
    blocks: list[list[str]] = [[] for _ in range(9)]
    for r in range(9):
        for c in range(9):
            value = grid[r][c] or (solution[r][c] if solution else 0) or ""
            classes = ["cell", f"r{r}", f"c{c}", f"b{3 * (r // 3) + c // 3}"]
            tip = f"Row {r + 1}, column {c + 1}: "
            if (r, c) in delay:
                classes.append("found")
                tip += f"{value} found by the solver"
            else:
                tip += f"{value} on the paper" if value else "empty"
            if (r, c) in marks:
                classes.append(marks[(r, c)])
            elif (r, c) in fixed:
                classes.append("fixed")
                tip += " · corrected using the sudoku rules"
            elif (r, c) in doubtful:
                classes.append("doubt")
                tip += " · uncertain reading"
            style = f' style="--d:{delay[(r, c)] * step:.0f}ms"' if (r, c) in delay else ""
            blocks[3 * (r // 3) + c // 3].append(
                f'<div class="{" ".join(classes)}"{style} data-tip="{html.escape(tip)}">{value}</div>')
    body = "".join(f'<div class="blk">{"".join(cells)}</div>' for cells in blocks)
    end = f' style="--end:{len(found) * step + 300:.0f}ms"' if pop else ""
    return f'<div class="board{" pop" if pop else ""}"{end}>{body}</div>'


def _tier(conf: float) -> str:
    return "t3" if conf >= 0.98 else "t2" if conf >= 0.9 else "t1" if conf >= 0.8 else "lo" if conf >= 0.5 else "bad"


def heatmap_html(confidence: np.ndarray, grid: Sequence[Sequence[int]]) -> str:
    """9×9 : chaque case colorée selon la confiance du classifieur (les cases vides comptent aussi)."""
    cells = []
    for r in range(9):
        for c in range(9):
            conf = float(confidence[r, c])
            cells.append(f'<div class="{_tier(conf)}" style="--i:{r * 9 + c}" '
                         f'data-tip="Row {r + 1}, column {c + 1} · {conf:.1%} sure">{grid[r][c] or "·"}</div>')
    return (f'<div class="heat">{"".join(cells)}</div>'
            '<div class="heat-legend"><span>uncertain</span><i></i><span>very sure</span></div>')


HIST_EDGES = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.995, 1.0001]
HIST_LABELS = ["&lt;50", "60", "70", "80", "85", "90", "95", "98", "99.5", "100"]


def histogram_html(confidence: np.ndarray) -> str:
    """Répartition des 81 confiances. Hauteur en racine carrée : sinon les rares cases douteuses disparaissent."""
    counts, _ = np.histogram(np.asarray(confidence).ravel(), bins=HIST_EDGES)
    top = max(int(counts.max()), 1)
    cols = []
    for k, n in enumerate(counts):
        tone = "low" if k < 4 else "mid" if k < 6 else "good"
        h = 100 * math.sqrt(n / top)
        cols.append(f'<div><span class="v">{n if n else ""}</span>'
                    f'<i class="bar {tone}" style="--h:{h:.1f};--i:{k}"></i><span class="l">{HIST_LABELS[k]}</span></div>')
    return f'<div class="hist" role="img" aria-label="Confidence distribution">{"".join(cols)}</div>'


def gauge_html(value: float, label: str) -> str:
    """Anneau rempli à `value` (0 à 1), valeur au centre et libellé dessous (trop long pour tenir dans le trou)."""
    value = min(max(value, 0.0), 1.0)
    return (f'<div class="ring gauge" style="--p:{value * 100:.1f}" role="img" aria-label="{html.escape(label)} : {value:.0%}">'
            f'<div>{counter(f"{value * 100:.0f}%")}</div></div>'
            f'<span class="ring-label">{html.escape(label)}</span>')


PALETTE = {"c0": "var(--blue)", "c1": "var(--green)", "c2": "var(--amber)", "c3": "var(--rose)", "c4": "var(--muted)"}


def donut_html(parts: list[tuple[str, float, str]], center: str, unit: str) -> str:
    """Anneau en parts : (légende, valeur, classe de couleur c0..c4), avec la légende dessous."""
    total = sum(v for _, v, _ in parts) or 1
    stops, items, start = [], [], 0.0
    for label, v, cls in parts:
        end = start + 100 * v / total
        stops.append(f"{PALETTE[cls]} calc(var(--k) * {start:.2f}%) calc(var(--k) * {end:.2f}%)")
        items.append(f'<li><i class="{cls}"></i>{html.escape(label)}<span>{v:g}</span></li>')
        start = end
    gradient = f"conic-gradient({', '.join(stops)}, var(--line) 0)"
    ring = (f'<div class="ring donut" style="background:{gradient}">'
            f'<div>{counter(center)}<small>{html.escape(unit)}</small></div></div>')
    return ring + f'<ul class="legend-list">{"".join(items)}</ul>'


def stack_html(steps: list[tuple[str, float]]) -> str:
    """Barre empilée des durées (secondes) : (nom de l'étape, durée)."""
    steps = [(n, t) for n, t in steps if t and t > 0]
    total = sum(t for _, t in steps) or 1
    segs = "".join(f'<i class="c{k % 5}" style="--w:{100 * t / total:.2f}%;--i:{k}" data-tip="{html.escape(n)}: {t * 1000:.0f} ms ({t / total:.0%})"></i>'
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
    body = "".join(lines) or '<p class="note">No metrics found in MLflow for this model.</p>'
    head = f'<div class="model"><b>{html.escape(title)}</b><span>{html.escape(tag)}</span></div>'
    return panel("", head + body, note)
