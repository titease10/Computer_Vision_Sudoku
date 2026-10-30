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
  const W = 720, H = 460, TIME = 45, LIVES = 3, TAU = Math.PI * 2, MONO = 'ui-monospace,Consolas,monospace';
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const ease = (x) => x * x * (3 - 2 * x);
  let pos = 0, root = null, raf = 0, G = null, ac = null, muted = false, scan = null;

  // ---------- clavier : Konami code pour ouvrir, Échap pour fermer ----------
  function onKey(e) {
    if (root) {
      if (e.key === 'Escape') close();
      else if (e.key === 'Enter' && G.state !== 'play') start();
      else if (e.key === 'm' || e.key === 'M') muted = !muted;
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

  // ---------- sons (WebAudio, aucun fichier) ----------
  function tone(f, d, type, vol, slide) {
    if (muted || !ac) return;
    try {
      const o = ac.createOscillator(), g = ac.createGain(), t = ac.currentTime;
      o.type = type || 'sine'; o.frequency.setValueAtTime(f, t);
      if (slide) o.frequency.exponentialRampToValueAtTime(Math.max(30, f * slide), t + d);
      g.gain.setValueAtTime(vol || .05, t); g.gain.exponentialRampToValueAtTime(.0001, t + d);
      o.connect(g); g.connect(ac.destination); o.start(t); o.stop(t + d);
    } catch (e) {}
  }
  const sfx = {
    lock: () => { tone(880, .07, 'square', .03); setTimeout(() => tone(1320, .09, 'square', .03), 70); },
    snap: (c) => { tone(520 + 520 * c, .12, 'triangle', .07); setTimeout(() => tone(780 + 600 * c, .16, 'triangle', .06), 60); },
    perfect: () => [660, 880, 1320, 1760].forEach((f, i) => setTimeout(() => tone(f, .12, 'triangle', .06), i * 55)),
    bad: () => { tone(160, .35, 'sawtooth', .07, .4); },
    whiff: () => tone(200, .08, 'square', .03, .6),
    level: () => [440, 660, 880].forEach((f, i) => setTimeout(() => tone(f, .14, 'square', .04), i * 90)),
    over: () => [440, 330, 220].forEach((f, i) => setTimeout(() => tone(f, .25, 'sawtooth', .05), i * 160)),
  };

  // ---------- ouverture / fermeture ----------
  function open() {
    root = D.createElement('div');
    root.style.cssText = 'position:fixed;inset:0;z-index:2147483000;display:grid;place-items:center;background:rgba(2,6,4,.96);' +
      'font-family:' + MONO + ';color:#7CFFB2;text-align:center;user-select:none;-webkit-user-select:none';
    root.innerHTML = '<div><canvas style="width:' + W + 'px;max-width:94vw;aspect-ratio:' + W + '/' + H + ';border:1px solid #1d6a42;' +
      'border-radius:14px;cursor:none;touch-action:none;box-shadow:0 0 70px rgba(60,255,150,.2)"></canvas>' +
      '<div style="margin-top:10px;font-size:12px;opacity:.6">move to aim · hold on a target to lock it · click to snap · M mute · Esc quit</div></div>';
    D.body.appendChild(root);
    const cv = root.querySelector('canvas'), dpr = Math.min(P.devicePixelRatio || 1, 2);
    cv.width = W * dpr; cv.height = H * dpr;
    G = { cv, ctx: cv.getContext('2d'), dpr, rx: W / 2, ry: H / 2, mx: W / 2, my: H / 2, moved: 0 };
    cv.addEventListener('pointermove', (e) => aim(e));
    cv.addEventListener('pointerdown', (e) => { aim(e); if (G.state === 'play') snap(); else start(); });
    scan = D.createElement('canvas'); scan.width = 4; scan.height = 3;
    const sc = scan.getContext('2d'); sc.fillStyle = 'rgba(0,0,0,.16)'; sc.fillRect(0, 2, 4, 1);
    reset('ready');
    raf = P.requestAnimationFrame(loop);
  }

  function close() {
    if (!root) return;
    P.cancelAnimationFrame(raf);
    root.remove(); root = null;
  }

  function aim(e) {
    const r = G.cv.getBoundingClientRect();
    G.mx = (e.clientX - r.left) * W / r.width; G.my = (e.clientY - r.top) * H / r.height; G.moved = performance.now();
  }

  function reset(state) {
    Object.assign(G, { state, t: 0, last: 0, targets: [], parts: [], floats: [], score: 0, combo: 0, lives: LIVES, found: 0, fp: 0,
      confSum: 0, level: 1, spawnIn: .3, shake: 0, flash: 0, flashCol: '80,255,150', time: TIME, newBest: false, whiffs: 0, escaped: 0 });
  }

  function start() {
    if (!ac) { try { ac = new (P.AudioContext || P.webkitAudioContext)(); } catch (e) {} }
    if (ac && ac.state === 'suspended') ac.resume();
    reset('play');
    G.floats.push({ x: W / 2, y: H / 2, text: 'GO!', col: '124,255,178', life: 1, size: 54, vy: -20 });
  }

  function end() {
    G.state = 'over';
    G.newBest = G.score > best();
    if (G.newBest) saveBest(G.score);
    sfx.over();
  }

  // ---------- cibles ----------
  function spawn() {
    const lv = G.level, grid = Math.random() < .56;
    let kind = 'grid';
    if (!grid) {
      const lookalike = Math.random() < Math.min(.12 + .08 * lv, .6);
      kind = lookalike ? (Math.random() < .5 ? 'chess' : 'cross') : ['stain', 'thumb', 'shadow'][Math.floor(Math.random() * 3)];
    }
    const s = 66 + Math.random() * 34, sp = 26 + 9 * lv + Math.random() * 30, a = Math.random() * TAU;
    let x, y, tries = 0;
    do {
      x = 14 + Math.random() * (W - s - 28); y = 70 + Math.random() * (H - s - 90); tries++;
    } while (tries < 16 && G.targets.some((o) => !o.dead && Math.abs(o.x + o.s / 2 - x - s / 2) < (o.s + s) / 2 + 14 && Math.abs(o.y + o.s / 2 - y - s / 2) < (o.s + s) / 2 + 14));
    G.targets.push({
      kind, grid, look: kind === 'chess' || kind === 'cross', s, x, y,
      vx: Math.cos(a) * sp, vy: Math.sin(a) * sp, age: 0, dead: false, life: Math.max(3.4, 6 - .32 * lv), lock: 0, conf: 0, noise: 0, nt: 0,
      ph: Math.random() * TAU, pinged: false,
      digits: Array.from({ length: 24 }, () => [Math.floor(Math.random() * 9), Math.floor(Math.random() * 9), 1 + Math.floor(Math.random() * 9)]),
      black: Array.from({ length: 49 }, () => Math.random() < .27),
    });
    G.spawnIn = Math.max(.4, .95 - .07 * lv) * (.75 + Math.random() * .5);
  }

  const inside = (t) => Math.hypot(t.x + t.s / 2 - G.rx, t.y + t.s / 2 - G.ry) < 52 + t.s * .1;

  function snap() {
    const cands = G.targets.filter((t) => !t.dead && inside(t)).sort((a, b) => b.lock - a.lock);
    const t = cands[0], cx = G.rx, cy = G.ry;
    if (!t) {
      G.combo = 0; G.score = Math.max(0, G.score - 5); G.whiffs++;
      G.floats.push({ x: cx, y: cy - 30, text: 'no detection', col: '170,190,180', life: .8, size: 15, vy: -30 });
      sfx.whiff();
      return;
    }
    const x = t.x + t.s / 2, y = t.y + t.s / 2;
    t.dead = true;
    if (t.grid) {
      const mult = 1 + Math.min(G.combo, 4) * .5, perfect = t.conf >= .96;
      const pts = Math.round((15 + 85 * t.conf * t.conf) * mult) + (perfect ? 25 : 0);
      G.score += pts; G.combo++; G.found++; G.confSum += t.conf;
      G.floats.push({ x, y: y - 10, text: '+' + pts, col: perfect ? '255,215,90' : '124,255,178', life: 1.1, size: perfect ? 34 : 26, vy: -55 });
      if (perfect) G.floats.push({ x, y: y + 18, text: 'PERFECT', col: '255,215,90', life: 1.1, size: 15, vy: -40 });
      burst(x, y, perfect ? '255,215,90' : '90,255,160', perfect ? 34 : 20);
      G.flash = .18; G.flashCol = perfect ? '255,215,90' : '80,255,150';
      perfect ? sfx.perfect() : sfx.snap(t.conf);
      const lv = 1 + Math.floor(G.found / 6);
      if (lv > G.level) { G.level = lv; G.floats.push({ x: W / 2, y: 120, text: 'LEVEL ' + lv, col: '124,255,178', life: 1.5, size: 40, vy: -10 }); setTimeout(sfx.level, 250); }
    } else {
      G.fp++; G.lives--; G.combo = 0; G.score = Math.max(0, G.score - 40); G.shake = .4; G.flash = .3; G.flashCol = '255,70,70';
      G.floats.push({ x, y: y - 10, text: 'FALSE POSITIVE', col: '255,100,100', life: 1.3, size: 22, vy: -40 });
      G.floats.push({ x, y: y + 16, text: t.look ? 'not a sudoku!' : '-40', col: '255,160,160', life: 1.3, size: 14, vy: -30 });
      burst(x, y, '255,90,90', 22);
      sfx.bad();
    }
  }

  function burst(x, y, col, n) {
    for (let i = 0; i < n; i++) {
      const a = Math.random() * TAU, v = 60 + Math.random() * 220;
      G.parts.push({ x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life: .5 + Math.random() * .5, max: 1, col, s: 2 + Math.random() * 4 });
    }
  }

  // ---------- mise à jour ----------
  function update(dt, now) {
    G.t += dt;
    if (G.state === 'ready') { G.mx = W / 2 + Math.cos(G.t * .7) * W * .34; G.my = H / 2 + Math.sin(G.t * 1.15) * H * .26; }
    G.rx += (G.mx - G.rx) * Math.min(1, dt * 16); G.ry += (G.my - G.ry) * Math.min(1, dt * 16);
    if (G.state === 'play') {
      G.time -= dt;
      if (G.time <= 0 || G.lives <= 0) end();
    }
    if (G.state !== 'over') {
      G.spawnIn -= dt;
      if (G.spawnIn <= 0 && G.targets.length < Math.min(3 + G.level, 7)) spawn(); else if (G.spawnIn <= 0) G.spawnIn = .3;
    }
    G.targets.forEach((t) => {
      t.nt -= dt;
      if (G.state !== 'over') {
        t.age += dt;
        t.x += t.vx * dt + Math.sin(G.t * 1.7 + t.ph) * 8 * dt; t.y += t.vy * dt + Math.cos(G.t * 1.3 + t.ph) * 8 * dt;
        if (t.x < 8 || t.x > W - t.s - 8) { t.vx *= -1; t.x = clamp(t.x, 8, W - t.s - 8); }
        if (t.y < 64 || t.y > H - t.s - 8) { t.vy *= -1; t.y = clamp(t.y, 64, H - t.s - 8); }
      }
      t.lock = clamp(t.lock + (inside(t) ? dt / .5 : -dt / .9), 0, 1);
      if (t.nt <= 0) { t.noise = (Math.random() - .5) * .04; t.nt = .09; }
      const cap = t.grid ? .99 : (t.look ? .94 : .58), base = t.grid ? .3 : .16;
      t.conf = clamp(base + (cap - base) * ease(t.lock) + t.noise, 0, .995);
      if (t.lock > .97 && !t.pinged && G.state === 'play') { t.pinged = true; sfx.lock(); }
      if (t.lock < .5) t.pinged = false;
      if (t.age > t.life && !t.dead) { t.dead = true; if (t.grid && G.state === 'play') { G.combo = 0; G.escaped++; } }
    });
    if (G.state !== 'over') {
      const live = G.targets.filter((t) => !t.dead);
      for (let i = 0; i < live.length; i++) for (let j = i + 1; j < live.length; j++) {
        const a = live[i], b = live[j], dx = b.x + b.s / 2 - a.x - a.s / 2, dy = b.y + b.s / 2 - a.y - a.s / 2, m = (a.s + b.s) / 2;
        if (Math.abs(dx) < m && Math.abs(dy) < m) {
          if (Math.abs(dx) >= Math.abs(dy)) {
            const k = dx >= 0 ? 1 : -1; a.x -= k * 2; b.x += k * 2;
            if ((b.vx - a.vx) * k < 0) { const v = a.vx; a.vx = b.vx; b.vx = v; }
          } else {
            const k = dy >= 0 ? 1 : -1; a.y -= k * 2; b.y += k * 2;
            if ((b.vy - a.vy) * k < 0) { const v = a.vy; a.vy = b.vy; b.vy = v; }
          }
        }
      }
    }
    G.targets = G.targets.filter((t) => !t.dead);
    G.parts.forEach((p) => { p.life -= dt; p.x += p.vx * dt; p.y += p.vy * dt; p.vx *= .96; p.vy *= .96; });
    G.parts = G.parts.filter((p) => p.life > 0);
    G.floats.forEach((f) => { f.life -= dt; f.y += f.vy * dt; });
    G.floats = G.floats.filter((f) => f.life > 0);
    G.shake = Math.max(0, G.shake - dt); G.flash = Math.max(0, G.flash - dt);
  }

  // ---------- dessin des cibles ----------
  function drawTarget(c, t, alpha) {
    c.save(); c.globalAlpha = alpha;
    c.translate(t.x + t.s / 2, t.y + t.s / 2); c.rotate(Math.sin(G.t * .6 + t.ph) * .1);
    const s = t.s, h = s / 2;
    c.shadowColor = 'rgba(0,0,0,.6)'; c.shadowBlur = 14; c.shadowOffsetY = 5;
    if (t.kind === 'grid' || t.kind === 'cross' || t.kind === 'chess') {
      c.fillStyle = '#f1eee2'; c.fillRect(-h, -h, s, s); c.shadowColor = 'transparent';
      if (t.kind === 'grid') {
        for (let k = 0; k <= 9; k++) {
          c.strokeStyle = k % 3 === 0 ? '#1f1f1f' : '#8a8a80'; c.lineWidth = k % 3 === 0 ? 2 : .7;
          c.beginPath(); c.moveTo(-h + k * s / 9, -h); c.lineTo(-h + k * s / 9, h); c.moveTo(-h, -h + k * s / 9); c.lineTo(h, -h + k * s / 9); c.stroke();
        }
        c.fillStyle = '#1c1c1c'; c.font = 'bold ' + Math.round(s / 10.5) + 'px Georgia,serif'; c.textAlign = 'center'; c.textBaseline = 'middle';
        t.digits.forEach(([r, k, d]) => c.fillText(d, -h + k * s / 9 + s / 18, -h + r * s / 9 + s / 18 + 1));
      } else if (t.kind === 'chess') {
        for (let r = 0; r < 8; r++) for (let k = 0; k < 8; k++) { c.fillStyle = (r + k) % 2 ? '#3a3a3a' : '#e9e6da'; c.fillRect(-h + k * s / 8, -h + r * s / 8, s / 8 + .5, s / 8 + .5); }
        c.strokeStyle = '#1f1f1f'; c.lineWidth = 2; c.strokeRect(-h, -h, s, s);
        c.fillStyle = '#c9a227'; c.font = Math.round(s / 5) + 'px serif'; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText('♞', 0, 0);
      } else {
        for (let r = 0; r < 7; r++) for (let k = 0; k < 7; k++) {
          if (t.black[r * 7 + k]) { c.fillStyle = '#202020'; c.fillRect(-h + k * s / 7, -h + r * s / 7, s / 7 + .5, s / 7 + .5); }
          else if ((r * 7 + k) % 5 === 0) { c.fillStyle = '#333'; c.font = Math.round(s / 11) + 'px sans-serif'; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText('ABCDEFGHIJ'[(r + k) % 10], -h + k * s / 7 + s / 14, -h + r * s / 7 + s / 14); }
        }
        c.strokeStyle = '#555'; c.lineWidth = .8;
        for (let k = 0; k <= 7; k++) { c.beginPath(); c.moveTo(-h + k * s / 7, -h); c.lineTo(-h + k * s / 7, h); c.moveTo(-h, -h + k * s / 7); c.lineTo(h, -h + k * s / 7); c.stroke(); }
      }
    } else if (t.kind === 'stain') {
      c.fillStyle = 'rgba(122,74,36,.92)'; c.beginPath(); c.ellipse(0, 0, h, h * .86, .4, 0, TAU); c.fill(); c.shadowColor = 'transparent';
      c.strokeStyle = 'rgba(70,40,16,.95)'; c.lineWidth = 4; c.beginPath(); c.ellipse(0, 0, h * .8, h * .66, .4, 0, TAU); c.stroke();
      c.fillStyle = 'rgba(190,140,90,.35)'; c.beginPath(); c.ellipse(-h * .25, -h * .2, h * .25, h * .15, .4, 0, TAU); c.fill();
    } else if (t.kind === 'thumb') {
      c.fillStyle = '#e1a78e'; c.beginPath(); c.roundRect(-s * .28, -h, s * .56, s, s * .28); c.fill(); c.shadowColor = 'transparent';
      c.fillStyle = '#f4d2c3'; c.beginPath(); c.roundRect(-s * .17, -h * .88, s * .34, s * .26, s * .1); c.fill();
      c.strokeStyle = 'rgba(150,90,70,.4)'; c.lineWidth = 1; for (let k = 0; k < 4; k++) { c.beginPath(); c.arc(0, h * .3 + k * 4, s * .16, .2, Math.PI - .2); c.stroke(); }
    } else {
      c.fillStyle = 'rgba(60,82,72,.95)'; c.strokeStyle = 'rgba(130,165,148,.85)'; c.lineWidth = 2; c.beginPath();
      c.moveTo(-h, -h * .4); c.lineTo(h, -h); c.lineTo(h * .6, h); c.lineTo(-h * .8, h * .7); c.closePath(); c.fill(); c.stroke();
    }
    c.restore();
  }

  function drawBox(c, t) {
    if (t.lock < .06 || t.dead) return;
    const hue = Math.round(120 * clamp((t.conf - .35) / .6, 0, 1)), col = 'hsl(' + hue + ',95%,58%)';
    const x = t.x - 5, y = t.y - 5, s = t.s + 10, l = Math.min(20, s * .3);
    c.save(); c.strokeStyle = col; c.lineWidth = 2.6; c.shadowColor = col; c.shadowBlur = 10;
    [[x, y, 1, 1], [x + s, y, -1, 1], [x, y + s, 1, -1], [x + s, y + s, -1, -1]].forEach(([px, py, dx, dy]) => {
      c.beginPath(); c.moveTo(px + dx * l, py); c.lineTo(px, py); c.lineTo(px, py + dy * l); c.stroke();
    });
    c.globalAlpha = .25 + .35 * t.lock; c.lineWidth = 1; c.strokeRect(x, y, s, s); c.globalAlpha = 1; c.shadowBlur = 0;
    const label = 'sudoku ' + t.conf.toFixed(2); c.font = 'bold 12px ' + MONO;
    const w = c.measureText(label).width + 12, ly = y > 82 ? y - 20 : y + s + 3;
    c.fillStyle = col; c.fillRect(x, ly, w, 18); c.fillStyle = '#04100a'; c.textAlign = 'left'; c.textBaseline = 'middle'; c.fillText(label, x + 6, ly + 9.5);
    c.restore();
  }

  function drawReticle(c, now) {
    let ml = 0; G.targets.forEach((t) => { if (!t.dead && inside(t)) ml = Math.max(ml, t.lock); });
    const R = 44, x = G.rx, y = G.ry, ready = ml > .95, col = ready ? '255,215,90' : '170,255,205';
    c.save(); c.translate(x, y);
    c.strokeStyle = 'rgba(' + col + ',.95)'; c.lineWidth = 2.4; c.shadowColor = 'rgba(' + col + ',.9)'; c.shadowBlur = 8;
    const o = ready ? 3 + Math.sin(now / 70) * 2 : 0;
    [[-1, -1], [1, -1], [-1, 1], [1, 1]].forEach(([dx, dy]) => {
      c.beginPath(); c.moveTo(dx * (R + o), dy * (R + o - 14)); c.lineTo(dx * (R + o), dy * (R + o)); c.lineTo(dx * (R + o - 14), dy * (R + o)); c.stroke();
    });
    c.beginPath(); c.moveTo(-8, 0); c.lineTo(8, 0); c.moveTo(0, -8); c.lineTo(0, 8); c.stroke();
    if (ml > .02) { c.strokeStyle = 'rgba(' + col + ',1)'; c.lineWidth = 3.5; c.beginPath(); c.arc(0, 0, R - 10, -Math.PI / 2, -Math.PI / 2 + TAU * ml); c.stroke(); }
    if (ready) { c.shadowBlur = 0; c.fillStyle = 'rgba(255,215,90,.95)'; c.font = 'bold 13px ' + MONO; c.textAlign = 'center'; c.fillText('SNAP!', 0, R + 24); }
    c.restore();
  }

  // ---------- interface (HUD, écrans) ----------
  function glowText(c, text, x, y, size, col, align, blur) {
    c.save(); c.font = 'bold ' + size + 'px ' + MONO; c.textAlign = align || 'left'; c.textBaseline = 'alphabetic';
    c.shadowColor = 'rgba(' + col + ',.85)'; c.shadowBlur = blur === undefined ? 12 : blur; c.fillStyle = 'rgb(' + col + ')'; c.fillText(text, x, y); c.restore();
  }

  function drawHud(c, now) {
    glowText(c, String(G.score).padStart(5, '0'), 18, 38, 30, '234,255,243');
    if (G.combo > 1) glowText(c, 'x' + (1 + Math.min(G.combo - 1, 4) * .5).toFixed(1) + ' combo ' + G.combo, 20, 56, 12, '255,215,90', 'left', 6);
    const bw = 230, bx = W / 2 - bw / 2, f = clamp(G.time / TIME, 0, 1);
    c.fillStyle = 'rgba(255,255,255,.1)'; c.beginPath(); c.roundRect(bx, 20, bw, 9, 5); c.fill();
    c.fillStyle = f > .25 ? '#58f09a' : '#ff5a5a'; c.beginPath(); c.roundRect(bx, 20, Math.max(8, bw * f), 9, 5); c.fill();
    glowText(c, Math.max(0, G.time).toFixed(1) + 's', W / 2, 50, 12, '170,255,205', 'center', 0);
    for (let i = 0; i < LIVES; i++) {
      const x = W - 24 - i * 26; c.fillStyle = i < G.lives ? '#ff5a7a' : 'rgba(255,255,255,.12)';
      c.save(); c.translate(x, 30); c.beginPath(); c.moveTo(0, 6); c.bezierCurveTo(-13, -4, -5, -12, 0, -4); c.bezierCurveTo(5, -12, 13, -4, 0, 6); c.fill(); c.restore();
    }
    glowText(c, 'LV ' + G.level, W - 18, 56, 12, '124,255,178', 'right', 4);
    // légendes « flux caméra »
    c.fillStyle = 'rgba(124,255,178,.55)'; c.font = '11px ' + MONO; c.textAlign = 'left'; c.fillText('CAM-01 · YOLO11n · conf ≥ 0.25', 14, H - 12);
    if (Math.floor(now / 500) % 2) { c.fillStyle = '#ff4d4d'; c.beginPath(); c.arc(W - 62, H - 16, 4, 0, TAU); c.fill(); }
    c.fillStyle = 'rgba(124,255,178,.55)'; c.textAlign = 'right'; c.fillText('REC', W - 16, H - 12);
  }

  function panel(c, w, h) {
    const x = (W - w) / 2, y = (H - h) / 2;
    c.fillStyle = 'rgba(3,12,8,.66)'; c.beginPath(); c.roundRect(x, y, w, h, 16); c.fill();
    c.strokeStyle = 'rgba(80,255,160,.45)'; c.lineWidth = 1.5; c.stroke();
  }

  function drawOverlay(c, now) {
    const pulse = .55 + .45 * Math.sin(now / 280);
    if (G.state === 'ready') {
      panel(c, 520, 290);
      const g = c.createLinearGradient(0, 100, 0, 160); g.addColorStop(0, '#d8ffe9'); g.addColorStop(1, '#41e88a');
      c.save(); c.font = 'bold 62px ' + MONO; c.textAlign = 'center'; c.shadowColor = 'rgba(60,255,150,.9)'; c.shadowBlur = 24; c.fillStyle = g; c.fillText('SNAP HUNT', W / 2, 168); c.restore();
      c.fillStyle = '#9fdcb9'; c.font = '14px ' + MONO; c.textAlign = 'center';
      c.fillText('hold the viewfinder on a sudoku grid until it locks, then click', W / 2, 205);
      c.fillText('chessboards, crosswords, fingers and stains fool the detector', W / 2, 228);
      c.fillText('3 lives · 45 seconds · faster every 6 grids', W / 2, 251);
      c.fillStyle = 'rgba(255,215,90,' + pulse + ')'; c.font = 'bold 17px ' + MONO; c.fillText('click or press Enter to start', W / 2, 296);
      if (best()) { c.fillStyle = '#6fbf91'; c.font = '13px ' + MONO; c.fillText('best ' + best(), W / 2, 322); }
    } else if (G.state === 'over') {
      panel(c, 500, 310);
      glowText(c, G.lives <= 0 ? 'SIGNAL LOST' : 'SCAN COMPLETE', W / 2, 140, 38, G.lives <= 0 ? '255,100,100' : '124,255,178', 'center', 20);
      glowText(c, String(G.score), W / 2, 200, 56, '234,255,243', 'center', 18);
      if (G.newBest) glowText(c, '★ NEW BEST ★', W / 2, 228, 16, '255,215,90', 'center', 10);
      const acc = G.found + G.fp ? Math.round(100 * G.found / (G.found + G.fp)) : 0, avg = G.found ? (G.confSum / G.found).toFixed(2) : '—';
      c.fillStyle = '#9fdcb9'; c.font = '14px ' + MONO; c.textAlign = 'center';
      c.fillText(G.found + ' grids · ' + G.fp + ' false positives · precision ' + acc + '% · avg conf ' + avg, W / 2, 258);
      c.fillText('best ' + best(), W / 2, 280);
      c.fillStyle = 'rgba(255,215,90,' + pulse + ')'; c.font = 'bold 16px ' + MONO; c.fillText('click or press Enter to play again', W / 2, 318);
    }
  }

  // ---------- boucle ----------
  function loop(now) {
    raf = P.requestAnimationFrame(loop);
    const dt = Math.min((now - (G.last || now)) / 1000, .05); G.last = now;
    update(dt, now);
    const c = G.ctx; c.setTransform(G.dpr, 0, 0, G.dpr, 0, 0);
    // fond : dégradé, repères, bruit capteur, balayage
    const bg = c.createLinearGradient(0, 0, 0, H); bg.addColorStop(0, '#08201a'); bg.addColorStop(1, '#0e3024');
    c.fillStyle = bg; c.fillRect(0, 0, W, H);
    c.save();
    if (G.shake > 0) c.translate((Math.random() - .5) * 14 * G.shake, (Math.random() - .5) * 14 * G.shake);
    c.strokeStyle = 'rgba(80,255,160,.05)'; c.lineWidth = 1;
    for (let x = 0; x <= W; x += 45) { c.beginPath(); c.moveTo(x, 0); c.lineTo(x, H); c.stroke(); }
    for (let y = 0; y <= H; y += 45) { c.beginPath(); c.moveTo(0, y); c.lineTo(W, y); c.stroke(); }
    c.fillStyle = 'rgba(160,255,200,.1)'; for (let i = 0; i < 46; i++) c.fillRect(Math.random() * W, Math.random() * H, 1.4, 1.4);
    const sw = (now / 8) % (H + 80) - 40; const sg = c.createLinearGradient(0, sw - 30, 0, sw + 30);
    sg.addColorStop(0, 'rgba(90,255,170,0)'); sg.addColorStop(.5, 'rgba(90,255,170,.09)'); sg.addColorStop(1, 'rgba(90,255,170,0)'); c.fillStyle = sg; c.fillRect(0, sw - 30, W, 60);

    G.targets.forEach((t) => drawTarget(c, t, G.state === 'over' ? .55 : clamp(Math.min(t.age / .2, (t.life - t.age) / .35), 0, 1)));
    G.targets.forEach((t) => drawBox(c, t));
    G.parts.forEach((p) => { c.fillStyle = 'rgba(' + p.col + ',' + clamp(p.life * 2, 0, 1) + ')'; c.fillRect(p.x, p.y, p.s, p.s); });
    G.floats.forEach((f) => { glowText(c, f.text, f.x, f.y, f.size, f.col, 'center', 8); });
    if (G.state !== 'over') drawReticle(c, now);
    c.restore();
    if (G.flash > 0) { c.fillStyle = 'rgba(' + G.flashCol + ',' + G.flash * .8 + ')'; c.fillRect(0, 0, W, H); }
    if (G.state !== 'ready') drawHud(c, now);
    drawOverlay(c, now);
    // effet écran cathodique : lignes, vignette
    c.fillStyle = c.createPattern(scan, 'repeat'); c.fillRect(0, 0, W, H);
    const v = c.createRadialGradient(W / 2, H / 2, H * .35, W / 2, H / 2, W * .65); v.addColorStop(0, 'rgba(0,0,0,0)'); v.addColorStop(1, 'rgba(0,0,0,.38)');
    c.fillStyle = v; c.fillRect(0, 0, W, H);
    root.dataset.state = G.state; root.dataset.score = G.score; root.dataset.lives = G.lives; root.dataset.level = G.level; root.dataset.combo = G.combo;
  }
})();
"""


def install() -> None:
    """Branche l'extra sur la page. L'iframe est vide et sort du flux : elle ne décale rien."""
    with st.container(key="extras"):
        components.html(f"<script>{SCRIPT}</script>", height=0)
