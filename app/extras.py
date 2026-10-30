"""Petits extras de l'interface.

Les scripts de `st.html` sont retirés par Streamlit : on passe par un composant `components.html`, dont l'iframe
(même origine que la page) branche ses écouteurs sur la page parente.
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

SCRIPT = r"""
(function () {
  const P = window.parent, D = P.document;
  if (P.__snapExtrasOff) P.__snapExtrasOff();  // un rechargement de l'iframe remplace l'ancien écouteur

  const SEQ = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'b', 'a'];
  const W = 640, H = 400, DURATION = 30000;
  let pos = 0, root = null, raf = 0, G = null;

  function onKey(e) {
    if (root) {
      if (e.key === 'Escape') close();
      else if (e.key === 'Enter' && G.state !== 'play') start();
      e.preventDefault();
      return;
    }
    const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    pos = k === SEQ[pos] ? pos + 1 : (k === SEQ[0] ? 1 : 0);
    if (pos === SEQ.length) { pos = 0; open(); }
  }
  D.addEventListener('keydown', onKey, true);
  P.__snapExtrasOff = () => { D.removeEventListener('keydown', onKey, true); close(); };

  const best = () => { try { return +P.localStorage.getItem('snapHuntBest') || 0; } catch (e) { return 0; } };
  const saveBest = (v) => { try { P.localStorage.setItem('snapHuntBest', String(v)); } catch (e) {} };

  function open() {
    root = D.createElement('div');
    root.style.cssText = 'position:fixed;inset:0;z-index:2147483000;display:grid;place-items:center;background:rgba(4,9,7,.94);' +
      'font-family:ui-monospace,Consolas,monospace;color:#7CFFB2;text-align:center';
    root.innerHTML = '<div><div style="display:flex;justify-content:space-between;width:' + W + 'px;max-width:92vw;margin:0 auto 8px;font-size:15px">' +
      '<span id="eg-score"></span><span id="eg-time"></span></div>' +
      '<canvas width="' + W + '" height="' + H + '" style="width:' + W + 'px;max-width:92vw;border:1px solid #1d5c3a;border-radius:10px;' +
      'cursor:crosshair;box-shadow:0 0 50px rgba(60,255,150,.16)"></canvas>' +
      '<div style="margin-top:10px;font-size:12px;opacity:.65">click the sudoku grids · avoid the decoys · Esc to quit</div></div>';
    D.body.appendChild(root);
    const cv = root.querySelector('canvas');
    G = { cv, ctx: cv.getContext('2d'), state: 'ready', objs: [], fx: [], score: 0, combo: 0, t0: 0, last: 0, spawnAt: 0, found: 0 };
    cv.addEventListener('pointerdown', onClick);
    raf = P.requestAnimationFrame(loop);
  }

  function close() {
    if (!root) return;
    P.cancelAnimationFrame(raf);
    root.remove();
    root = null;
  }

  function start() {
    Object.assign(G, { state: 'play', objs: [], fx: [], score: 0, combo: 0, found: 0, t0: performance.now(), spawnAt: 0 });
  }

  function spawn(now, elapsed) {
    const ramp = Math.min(elapsed / DURATION, 1);
    const s = 56 + Math.random() * 30;
    const grid = Math.random() < 0.62;
    G.objs.push({
      x: 20 + Math.random() * (W - s - 40), y: 20 + Math.random() * (H - s - 40), s, grid,
      kind: grid ? 'grid' : ['stain', 'thumb', 'shadow'][Math.floor(Math.random() * 3)],
      vx: (Math.random() - .5) * 50, vy: (Math.random() - .5) * 50, born: now, life: 2500 - ramp * 1000,
      digits: Array.from({ length: 16 }, () => [Math.floor(Math.random() * 9), Math.floor(Math.random() * 9), 1 + Math.floor(Math.random() * 9)]),
    });
    G.spawnAt = now + 720 - ramp * 340;
  }

  function onClick(e) {
    if (G.state !== 'play') { start(); return; }
    const r = G.cv.getBoundingClientRect();
    const x = (e.clientX - r.left) * W / r.width, y = (e.clientY - r.top) * H / r.height;
    for (let i = G.objs.length - 1; i >= 0; i--) {
      const o = G.objs[i];
      if (x >= o.x && x <= o.x + o.s && y >= o.y && y <= o.y + o.s) {
        const conf = (0.86 + Math.random() * 0.13).toFixed(2);
        if (o.grid) {
          G.combo = Math.min(G.combo + 1, 6);
          G.score += 10 + 2 * G.combo; G.found++;
          G.fx.push({ x: o.x, y: o.y, s: o.s, label: 'sudoku ' + conf, ok: true, t: performance.now() });
        } else {
          G.combo = 0; G.score = Math.max(0, G.score - 5);
          G.fx.push({ x: o.x, y: o.y, s: o.s, label: 'false positive ' + conf, ok: false, t: performance.now() });
        }
        G.objs.splice(i, 1);
        return;
      }
    }
    G.combo = 0;  // clic dans le vide : la série retombe
  }

  function drawObj(c, o, alpha) {
    c.save(); c.globalAlpha = alpha; c.translate(o.x, o.y);
    const s = o.s;
    if (o.kind === 'grid') {
      c.fillStyle = '#efece0'; c.fillRect(0, 0, s, s);
      c.strokeStyle = '#3b3b3b'; c.lineWidth = 1;
      for (let k = 0; k <= 9; k++) {
        c.lineWidth = k % 3 === 0 ? 1.8 : .6;
        c.beginPath(); c.moveTo(k * s / 9, 0); c.lineTo(k * s / 9, s); c.moveTo(0, k * s / 9); c.lineTo(s, k * s / 9); c.stroke();
      }
      c.fillStyle = '#222'; c.font = 'bold ' + Math.round(s / 11) + 'px serif'; c.textAlign = 'center'; c.textBaseline = 'middle';
      o.digits.forEach(([r, k, d]) => c.fillText(d, k * s / 9 + s / 18, r * s / 9 + s / 18));
    } else if (o.kind === 'stain') {
      c.fillStyle = 'rgba(120,72,34,.85)'; c.beginPath(); c.ellipse(s / 2, s / 2, s / 2, s / 2.4, .4, 0, 7); c.fill();
      c.strokeStyle = 'rgba(70,40,16,.9)'; c.lineWidth = 3; c.beginPath(); c.ellipse(s / 2, s / 2, s / 2.4, s / 2.9, .4, 0, 7); c.stroke();
    } else if (o.kind === 'thumb') {
      c.fillStyle = '#e0a58c'; c.beginPath(); c.roundRect(s * .22, s * .1, s * .56, s * .9, s * .26); c.fill();
      c.fillStyle = '#f2cfc0'; c.beginPath(); c.roundRect(s * .32, s * .16, s * .36, s * .26, s * .1); c.fill();
    } else {
      c.fillStyle = 'rgba(52,70,62,.92)'; c.strokeStyle = 'rgba(120,150,135,.8)'; c.lineWidth = 2; c.beginPath(); c.moveTo(0, s * .3); c.lineTo(s, 0); c.lineTo(s * .8, s); c.lineTo(s * .1, s * .85); c.closePath(); c.fill(); c.stroke();
    }
    c.restore();
  }

  function loop(now) {
    raf = P.requestAnimationFrame(loop);
    const c = G.ctx, dt = Math.min((now - (G.last || now)) / 1000, .05); G.last = now;
    // faux flux caméra : fond sombre, grille de repère, lignes de balayage
    c.fillStyle = '#07110c'; c.fillRect(0, 0, W, H);
    c.strokeStyle = 'rgba(80,255,160,.06)'; c.lineWidth = 1;
    for (let x = 0; x < W; x += 40) { c.beginPath(); c.moveTo(x, 0); c.lineTo(x, H); c.stroke(); }
    for (let y = 0; y < H; y += 40) { c.beginPath(); c.moveTo(0, y); c.lineTo(W, y); c.stroke(); }
    const sweep = (now / 9) % (H + 60) - 30;
    c.fillStyle = 'rgba(100,255,170,.07)'; c.fillRect(0, sweep, W, 26);

    const elapsed = G.state === 'play' ? now - G.t0 : 0;
    if (G.state === 'play') {
      if (elapsed >= DURATION) { G.state = 'over'; G.objs = []; if (G.score > best()) { G.newBest = true; saveBest(G.score); } else G.newBest = false; }
      else if (now >= G.spawnAt) spawn(now, elapsed);
    }
    G.objs = G.objs.filter(o => now - o.born < o.life);
    G.objs.forEach(o => {
      o.x = Math.max(4, Math.min(W - o.s - 4, o.x + o.vx * dt)); o.y = Math.max(4, Math.min(H - o.s - 4, o.y + o.vy * dt));
      const age = now - o.born, a = Math.min(1, age / 150, (o.life - age) / 300);
      drawObj(c, o, Math.max(a, 0));
    });
    G.fx = G.fx.filter(f => now - f.t < 800);
    G.fx.forEach(f => {
      const a = 1 - (now - f.t) / 800, col = f.ok ? '60,255,140' : '255,90,90';
      c.strokeStyle = 'rgba(' + col + ',' + a + ')'; c.lineWidth = 2.5; c.strokeRect(f.x - 3, f.y - 3, f.s + 6, f.s + 6);
      c.font = '12px ui-monospace,Consolas,monospace'; c.textAlign = 'left'; c.textBaseline = 'alphabetic';
      const tw = c.measureText(f.label).width + 10;
      c.fillStyle = 'rgba(' + col + ',' + a + ')'; c.fillRect(f.x - 3, f.y - 21, tw, 18);
      c.fillStyle = 'rgba(4,9,7,' + a + ')'; c.fillText(f.label, f.x + 2, f.y - 8);
    });

    root.querySelector('#eg-score').textContent = 'score ' + G.score + (G.combo > 1 ? '  x' + G.combo : '') + '   best ' + Math.max(best(), G.score);
    root.querySelector('#eg-time').textContent = G.state === 'play' ? (Math.max(0, (DURATION - elapsed) / 1000)).toFixed(1) + ' s' : '';

    if (G.state !== 'play') {
      c.fillStyle = 'rgba(4,9,7,.72)'; c.fillRect(0, 0, W, H);
      c.textAlign = 'center'; c.fillStyle = '#7CFFB2';
      c.font = 'bold 34px ui-monospace,Consolas,monospace';
      c.fillText(G.state === 'ready' ? 'SNAP HUNT' : 'SCAN COMPLETE', W / 2, 140);
      c.font = '15px ui-monospace,Consolas,monospace';
      if (G.state === 'ready') {
        c.fillText('find the sudoku grids before they fade', W / 2, 185);
        c.fillText('decoys (stains, fingers, shadows) are false positives', W / 2, 210);
      } else {
        c.fillText(G.found + ' grids detected · score ' + G.score + (G.newBest ? '  ★ new best' : ''), W / 2, 185);
      }
      c.fillStyle = 'rgba(124,255,178,' + (.55 + .45 * Math.sin(now / 280)) + ')';
      c.fillText('click or press Enter to ' + (G.state === 'ready' ? 'start' : 'play again'), W / 2, 262);
    }
  }
})();
"""


def install() -> None:
    """Branche l'extra sur la page. L'iframe est vide et sort du flux : elle ne décale rien."""
    with st.container(key="extras"):
        components.html(f"<script>{SCRIPT}</script>", height=0)
