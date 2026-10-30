"""Mode démo de l'application : on génère un lot de photos de sudokus, on le lance d'un coup, on ouvre un cas.

Le lot (valides, invalides, plusieurs grilles, photos dégradées…) vient de `sudoku_cv.demo_batch` ; ici il n'y a
que l'interface. Le cas ouvert repart dans le parcours habituel de l'application, comme une photo importée.
"""
from __future__ import annotations

import html
import io
import time
import zipfile

import cv2
import pandas as pd
import streamlit as st

from sudoku_cv.demo_batch import SPECS, Case, build_batch, outcome

SIZES = {"Small": 1, "Medium": 2, "Large": 3}  # nombre de passages sur la liste des cas
VERDICT_LABEL = {"ok": "Correct", "limit": "Limit reached", "miss": "Missed", "wrong": "Wrong answer"}
VERDICT_COLOR = {"ok": "--green-soft", "limit": "--amber", "miss": "--rose", "wrong": "--red"}
THUMB = 240

CSS = """
<style>
.verdict { display: inline-block; padding: .1em .6em; border-radius: 999px; font-size: .8rem; font-weight: 600; color: var(--ink); }
.v-ok { background: var(--green-soft); } .v-limit { background: var(--amber); }
.v-miss { background: var(--rose); } .v-wrong { background: var(--red); }
.demo-case { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: .9rem 1.2rem; margin: .8rem 0; }
.demo-case b { font-family: var(--serif); font-size: 1.15rem; }
.demo-case p { margin: .35rem 0 0; color: var(--muted); }
.thumb-cap { font-size: .8rem; margin: -.4rem 0 .3rem; color: var(--muted); }
</style>
"""


def _generate(seed: int, repeat: int) -> None:
    """Fabrique le lot et ses exports (images + manifeste) une fois pour toutes, dans la session."""
    cases = build_batch(seed, repeat)
    png = [cv2.imencode(".png", c.image)[1].tobytes() for c in cases]
    manifest = pd.DataFrame([{"file": f"{i + 1:02d}_{c.category.lower().replace(' ', '_')}.png", "category": c.category,
                              "expected": "solved" if c.expect == "solve" else "refused or corrected", "stress_test": c.stress,
                              "what_it_shows": c.note} for i, c in enumerate(cases)])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for name, data in zip(manifest["file"], png):
            z.writestr(name, data)
        z.writestr("manifest.csv", manifest.to_csv(index=False))
    st.session_state.demo = {"cases": cases, "png": png, "zip": buf.getvalue(), "seed": seed, "results": {}}
    st.session_state.demo_pick = None


def _run_all(pipe, settings: tuple) -> None:
    demo = st.session_state.demo
    results = {}
    bar = st.progress(0.0, text="Running the batch…")
    for i, case in enumerate(demo["cases"]):
        t = time.perf_counter()
        res = pipe.run(case.image)
        verdict, why = outcome(case, res)
        results[i] = {"verdict": verdict, "why": why, "status": res.status, "ms": (time.perf_counter() - t) * 1000}
        bar.progress((i + 1) / len(demo["cases"]), text=f"Running the batch… {i + 1}/{len(demo['cases'])}")
    bar.empty()
    demo["results"] = results
    demo["settings"] = settings


def _verdict_html(verdict: str) -> str:
    return f'<span class="verdict v-{verdict}">{VERDICT_LABEL[verdict]}</span>'


def _summary(results: dict) -> str:
    n = len(results)
    count = lambda v: sum(r["verdict"] == v for r in results.values())  # noqa: E731
    tiles = [(f"{count('ok')}/{n}", "correct", " good"), (str(count("limit")), "limits reached", ""),
             (str(count("miss")), "missed", ""), (str(count("wrong")), "wrong answers", "")]
    return '<div class="tiles">' + "".join(f'<div class="tile{c}"><b>{v}</b><span>{k}</span></div>'
                                           for v, k, c in tiles) + "</div>"


def _open(i: int) -> None:
    st.session_state.demo_pick = i


def show(pipe, settings: tuple, render) -> None:
    """Panneau du mode démo. `render(png, banner)` affiche une photo ; elle s'ouvre dans une modale, le lot reste derrière."""
    st.html(CSS)
    st.caption(f"Generated photos with a known answer: {len(SPECS)} situations per pass — clean, tilted, rotated, shadowed, "
               "several grids, invalid grids, and a few cases built to push the app to its limits. "
               "The download holds the images and a manifest.csv with the expected answer of each one.")
    c_seed, c_size, c_gen, _ = st.columns([1, 1, 1, 3], vertical_alignment="bottom")
    seed = c_seed.number_input("Seed", min_value=0, value=1, step=1, help="Same seed, same batch.")
    size = c_size.selectbox("Batch size", list(SIZES), format_func=lambda s: f"{s} ({SIZES[s] * len(SPECS)} photos)")
    if c_gen.button("Generate a new batch", width="stretch") or "demo" not in st.session_state:
        with st.spinner("Drawing the grids…"):
            _generate(int(seed), SIZES[size])
    demo = st.session_state.demo
    cases: list[Case] = demo["cases"]
    results = demo["results"] if demo.get("settings") == settings else {}

    run, save, _ = st.columns([1.3, 1.3, 3])
    if run.button("▶ Run the whole batch", type="primary", width="stretch"):
        _run_all(pipe, settings)
        results = demo["results"]
    save.download_button("⬇ Download the photos", demo["zip"], f"sudoku_demo_seed{demo['seed']}.zip", "application/zip",
                         width="stretch")

    if results:
        st.html(_summary(results))
        table = pd.DataFrame([{"Photo": c.name, "Expected": "solved" if c.expect == "solve" else "refused or corrected",
                               "Got": results[i]["status"], "Verdict": VERDICT_LABEL[results[i]["verdict"]],
                               "Why": results[i]["why"], "Time (ms)": round(results[i]["ms"])}
                              for i, c in enumerate(cases)])
        with st.expander("Results table"):
            st.dataframe(table, hide_index=True, width="stretch")

    with st.expander("Photos of the batch", expanded=True):
        for row in range(0, len(cases), 6):
            for col, i in zip(st.columns(6), range(row, min(row + 6, len(cases)))):
                case = cases[i]
                scale = THUMB / case.image.shape[1]
                col.image(cv2.cvtColor(cv2.resize(case.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                                       cv2.COLOR_BGR2RGB), width="stretch")
                col.html(f'<div class="thumb-cap">{html.escape(case.category)}'
                         + (f" · {_verdict_html(results[i]['verdict'])}" if i in results else "") + "</div>")
                col.button("Open", key=f"open_{i}", on_click=_open, args=(i,), width="stretch")

    pick = st.session_state.get("demo_pick")
    if pick is not None:
        st.session_state.demo_pick = None  # la modale ne se rouvre pas toute seule au rerun suivant
        case = cases[pick]
        st.dialog(case.name, width="large")(lambda: render(demo["png"][pick], lambda res: banner(case, res)))()


def banner(case: Case, res) -> None:
    """Ce que le cas ouvert cherche à montrer, et si l'application s'en est sortie."""
    verdict, why = outcome(case, res)
    expected = "a correct solution" if case.expect == "solve" else "a refusal or a corrected grid"
    st.html(f'<div class="demo-case">{_verdict_html(verdict)}'
            f'<p>{html.escape(case.note)} Expected: {expected}. {html.escape(why.capitalize())}.</p></div>')
