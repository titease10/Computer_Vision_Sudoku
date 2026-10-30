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
  const W = 720, H = 460, TAU = Math.PI * 2, MONO = 'ui-monospace,Consolas,monospace';
  const START_TIME = 25, MAX_TIME = 30, MISS_COST = 2.5, HIT_GAIN = .5, TOP = 84;
  const FONTS = ['Georgia,serif', 'Arial,Helvetica,sans-serif', '"Courier New",monospace', 'Impact,Haettenschweiler,sans-serif',
    '"Times New Roman",serif', '"Trebuchet MS",sans-serif', 'Verdana,sans-serif', '"Palatino Linotype",serif', '"Comic Sans MS",cursive'];
  const SIMILAR = { 1: [7, 4, 2], 2: [7, 3, 5], 3: [8, 5, 2], 4: [9, 1, 7], 5: [6, 3, 8], 6: [9, 5, 8], 7: [1, 9, 2], 8: [3, 6, 9], 9: [6, 4, 7] };
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const pickOf = (a) => a[Math.floor(Math.random() * a.length)];
  const backOut = (x) => { const c = 1.9; return 1 + (c + 1) * Math.pow(x - 1, 3) + c * Math.pow(x - 1, 2); };
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

  const tops = () => { try { return JSON.parse(P.localStorage.getItem('digitDashTop') || '[]'); } catch (e) { return []; } };
  const saveTop = (score) => {
    const list = tops().concat(score).sort((a, b) => b - a).slice(0, 5);
    try { P.localStorage.setItem('digitDashTop', JSON.stringify(list)); } catch (e) {}
    return list;
  };

  // ---------- sons (WebAudio, aucun fichier) : la série monte dans une gamme pentatonique ----------
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
  const SCALE = [0, 2, 4, 7, 9, 12, 14, 16, 19, 21, 24, 26, 28];
  const note = (i) => 261.63 * Math.pow(2, SCALE[Math.min(i, SCALE.length - 1)] / 12);
  const sfx = {
    hit: (combo) => { tone(note(combo), .16, 'triangle', .08); tone(note(combo) * 2, .1, 'sine', .03); },
    clear: () => [0, 4, 7, 12, 16].forEach((n, i) => setTimeout(() => tone(261.63 * Math.pow(2, n / 12) * 2, .14, 'triangle', .06), i * 55)),
    bad: () => { tone(150, .3, 'sawtooth', .07, .45); tone(110, .3, 'square', .04, .5); },
    tick: () => tone(1000, .04, 'square', .03),
    start: () => [0, 7, 12].forEach((n, i) => setTimeout(() => tone(261.63 * Math.pow(2, n / 12), .12, 'square', .04), i * 70)),
    over: () => [440, 392, 330, 262].forEach((f, i) => setTimeout(() => tone(f, .28, 'sawtooth', .05), i * 170)),
  };

  // ---------- ouverture / fermeture ----------
  function open() {
    root = D.createElement('div');
    root.style.cssText = 'position:fixed;inset:0;z-index:2147483000;display:grid;place-items:center;background:rgba(2,6,4,.96);' +
      'font-family:' + MONO + ';color:#7CFFB2;text-align:center;user-select:none;-webkit-user-select:none';
    root.innerHTML = '<div><canvas style="width:' + W + 'px;max-width:94vw;aspect-ratio:' + W + '/' + H + ';border:1px solid #1d6a42;' +
      'border-radius:14px;cursor:pointer;touch-action:none;box-shadow:0 0 70px rgba(60,255,150,.2)"></canvas>' +
      '<div style="margin-top:10px;font-size:12px;opacity:.6">click the digit shown at the top · M mute · Esc quit</div></div>';
    D.body.appendChild(root);
    const cv = root.querySelector('canvas'), dpr = Math.min(P.devicePixelRatio || 1, 2);
    cv.width = W * dpr; cv.height = H * dpr;
    G = { cv, ctx: cv.getContext('2d'), dpr, mx: -99, my: -99 };
    cv.addEventListener('pointermove', (e) => aim(e));
    cv.addEventListener('pointerdown', (e) => { aim(e); if (G.state === 'play') click(G.mx, G.my); else start(); });
    cv.addEventListener('pointerleave', () => { G.mx = -99; G.my = -99; });
    scan = D.createElement('canvas'); scan.width = 4; scan.height = 3;
    const sc = scan.getContext('2d'); sc.fillStyle = 'rgba(0,0,0,.14)'; sc.fillRect(0, 2, 4, 1);
    reset('ready');
    G.round = 3; newRound();
    raf = P.requestAnimationFrame(loop);
  }

  function close() {
    if (!root) return;
    P.cancelAnimationFrame(raf);
    root.remove(); root = null;
  }

  function aim(e) {
    const r = G.cv.getBoundingClientRect();
    G.mx = (e.clientX - r.left) * W / r.width; G.my = (e.clientY - r.top) * H / r.height;
  }

  function reset(state) {
    Object.assign(G, { state, t: 0, last: 0, round: 1, score: 0, combo: 0, bestCombo: 0, time: START_TIME, playT: 0, hits: 0, miss: 0, rounds: 0,
      confused: {}, parts: [], floats: [], boxes: [], shake: 0, flash: 0, flashCol: '80,255,150', tiles: [], target: 0, need: 0, total: 0,
      introT: 0, clearT: 0, roundT: 0, demoT: 0, mistakes: 0, tickAt: 99, newBest: false, rank: 0 });
  }

  function start() {
    if (!ac) { try { ac = new (P.AudioContext || P.webkitAudioContext)(); } catch (e) {} }
    if (ac && ac.state === 'suspended') ac.resume();
    reset('play');
    newRound();
    sfx.start();
  }

  function end() {
    G.state = 'over';
    const before = tops()[0] || 0, list = saveTop(G.score);
    G.rank = list.indexOf(G.score) + 1;
    G.newBest = G.score > before;
    G.floats.length = 0;
    sfx.over();
  }

  // ---------- manches ----------
  function newRound() {
    const r = G.round;
    const rows = Math.min(4 + Math.floor((r - 1) / 3), 7), cols = Math.round(rows * 1.6);
    const s = Math.floor(Math.min(680 / cols, (H - TOP - 14) / rows)), bw = cols * s, bh = rows * s;
    let target; do { target = 1 + Math.floor(Math.random() * 9); } while (target === G.target);
    const n = rows * cols, k = Math.min(3 + Math.floor(r / 3), 7, Math.floor(n / 4));
    const order = Array.from({ length: n }, (_, i) => i).sort(() => Math.random() - .5), isTarget = new Set(order.slice(0, k));
    const similarP = Math.min(.25 + .06 * r, .75), rotMax = Math.min(4 + 3 * r, 38) * Math.PI / 180;
    G.tiles = [];
    for (let i = 0; i < n; i++) {
      let d = target;
      if (!isTarget.has(i)) { do { d = Math.random() < similarP ? pickOf(SIMILAR[target]) : 1 + Math.floor(Math.random() * 9); } while (d === target); }
      const rr = Math.floor(i / cols), cc = i % cols;
      G.tiles.push({ r: rr, c: cc, digit: d, rot: (Math.random() - .5) * 2 * rotMax, font: pickOf(FONTS), weight: Math.random() < .5 ? 'bold' : 'normal',
        italic: Math.random() < .2, sc: .82 + Math.random() * .26, blur: r >= 5 && Math.random() < Math.min(.07 * (r - 4), .5) ? .8 + Math.random() * 1.2 : 0,
        dots: r >= 7 && Math.random() < .4 ? Array.from({ length: 9 }, () => [Math.random(), Math.random()]) : null,
        delay: (rr + cc) * .03, found: false, wrong: 0, wobble: 0 });
    }
    Object.assign(G, { target, need: k, total: k, rows, cols, s, ox: (W - bw) / 2, oy: TOP + (H - TOP - 10 - bh) / 2, introT: .35 + (rows + cols) * .03,
      clearT: 0, roundT: 0, mistakes: 0 });
  }

  function tileAt(x, y) {
    const c = Math.floor((x - G.ox) / G.s), r = Math.floor((y - G.oy) / G.s);
    if (c < 0 || r < 0 || c >= G.cols || r >= G.rows) return null;
    return G.tiles[r * G.cols + c];
  }

  function click(x, y) {
    if (G.state !== 'play' || G.introT > 0 || G.clearT > 0) return;
    const t = tileAt(x, y);
    if (!t || t.found) return;
    const cx = G.ox + t.c * G.s + G.s / 2, cy = G.oy + t.r * G.s + G.s / 2;
    if (t.digit === G.target) {
      t.found = true; G.need--; G.hits++; G.combo++; G.bestCombo = Math.max(G.bestCombo, G.combo);
      G.time = Math.min(MAX_TIME, G.time + HIT_GAIN);
      const mult = Math.min(1 + Math.floor(G.combo / 3) * .5, 5), pts = Math.round(10 * mult);
      G.score += pts;
      G.floats.push({ x: cx, y: cy - 6, text: '+' + pts, col: '124,255,178', life: .8, size: 18 + Math.min(G.combo, 12), vy: -50 });
      G.boxes.push({ x: cx, y: cy, s: G.s, label: G.target + ' ' + (.9 + Math.random() * .09).toFixed(2), ok: true, t: G.t });
      burst(cx, cy, '90,255,160', 10 + Math.min(G.combo, 10));
      sfx.hit(G.combo - 1);
      if (G.combo > 0 && G.combo % 5 === 0) G.floats.push({ x: W / 2, y: TOP + 20, text: G.combo + ' IN A ROW', col: '255,215,90', life: 1, size: 26, vy: -20 });
      if (G.need === 0) roundClear();
    } else {
      t.wrong = .55; t.wobble = .4; G.miss++; G.mistakes++; G.combo = 0; G.time -= MISS_COST; G.shake = .35; G.flash = .25; G.flashCol = '255,70,70';
      const key = G.target + '>' + t.digit; G.confused[key] = (G.confused[key] || 0) + 1;
      G.floats.push({ x: cx, y: cy - 6, text: '-' + MISS_COST + 's', col: '255,110,110', life: .9, size: 22, vy: -40 });
      G.boxes.push({ x: cx, y: cy, s: G.s, label: t.digit + '? ' + (.2 + Math.random() * .25).toFixed(2), ok: false, t: G.t });
      burst(cx, cy, '255,90,90', 14);
      sfx.bad();
      if (G.time <= 0) end();
    }
  }

  function roundClear() {
    const bonus = 50 + 10 * G.round + (G.mistakes === 0 ? 100 : 0), gain = Math.max(1.5, 4 - .25 * G.round);
    G.score += bonus; G.rounds++; G.time = Math.min(MAX_TIME, G.time + gain); G.clearT = .6;
    G.floats.push({ x: W / 2, y: H / 2 - 10, text: 'ROUND CLEAR  +' + bonus, col: '124,255,178', life: 1.1, size: 30, vy: -25 });
    if (G.mistakes === 0) G.floats.push({ x: W / 2, y: H / 2 + 22, text: 'FLAWLESS', col: '255,215,90', life: 1.1, size: 20, vy: -25 });
    G.floats.push({ x: W / 2, y: TOP - 4, text: '+' + gain.toFixed(1) + 's', col: '150,215,255', life: 1, size: 18, vy: -20 });
    G.flash = .18; G.flashCol = '80,255,150';
    G.tiles.forEach((t) => { if (t.digit === G.target) burst(G.ox + t.c * G.s + G.s / 2, G.oy + t.r * G.s + G.s / 2, '124,255,178', 6); });
    sfx.clear();
  }

  function burst(x, y, col, n) {
    for (let i = 0; i < n; i++) {
      const a = Math.random() * TAU, v = 60 + Math.random() * 200;
      G.parts.push({ x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life: .4 + Math.random() * .5, col, s: 2 + Math.random() * 3.5 });
    }
  }

  // ---------- mise à jour ----------
  function update(dt) {
    G.t += dt; G.roundT += dt;
    if (G.state === 'play') {
      if (G.clearT > 0) { G.clearT -= dt; if (G.clearT <= 0) { G.round++; newRound(); } }
      else if (G.introT > 0) G.introT -= dt;
      else {
        G.time -= dt; G.playT += dt;
        const sec = Math.ceil(G.time);
        if (G.time < 5 && sec !== G.tickAt) { G.tickAt = sec; sfx.tick(); }
        if (G.time >= 5) G.tickAt = 99;
        if (G.time <= 0) { G.time = 0; end(); }
      }
    } else if (G.state === 'ready') {  // démo : le tableau se résout tout seul derrière le titre
      G.demoT += dt;
      if (G.demoT > .7 && G.roundT > 1) {
        G.demoT = 0;
        const left = G.tiles.filter((t) => t.digit === G.target && !t.found);
        if (left.length) { const t = pickOf(left); t.found = true; burst(G.ox + t.c * G.s + G.s / 2, G.oy + t.r * G.s + G.s / 2, '90,255,160', 10); }
        else { G.round = 2 + Math.floor(Math.random() * 6); newRound(); }
      }
    }
    G.tiles.forEach((t) => { t.wrong = Math.max(0, t.wrong - dt); t.wobble = Math.max(0, t.wobble - dt); });
    G.parts.forEach((p) => { p.life -= dt; p.x += p.vx * dt; p.y += p.vy * dt; p.vx *= .95; p.vy *= .95; });
    G.parts = G.parts.filter((p) => p.life > 0);
    G.floats.forEach((f) => { f.life -= dt; f.y += f.vy * dt; });
    G.floats = G.floats.filter((f) => f.life > 0);
    G.boxes = G.boxes.filter((b) => G.t - b.t < .7);
    G.shake = Math.max(0, G.shake - dt); G.flash = Math.max(0, G.flash - dt);
  }

  // ---------- dessin ----------
  const hue = () => (G.round * 47) % 360;

  function drawTile(c, t, hover, pulse) {
    const s = G.s, pad = 3, p = clamp((G.roundT - t.delay) / .25, 0, 1);
    if (p <= 0) return;
    const sc = backOut(p) * (hover ? 1.07 : 1), cx = G.ox + t.c * s + s / 2, cy = G.oy + t.r * s + s / 2, w = s - 2 * pad;
    c.save(); c.translate(cx, cy); c.scale(sc, sc);
    if (t.wobble > 0) c.rotate(Math.sin(t.wobble * 40) * .12 * t.wobble / .4);
    const over = G.state === 'over' && t.digit === G.target && !t.found;
    const g = c.createLinearGradient(0, -w / 2, 0, w / 2);
    if (t.found) { g.addColorStop(0, 'rgba(70,255,150,.30)'); g.addColorStop(1, 'rgba(40,200,110,.18)'); }
    else if (t.wrong > 0) { g.addColorStop(0, 'rgba(255,80,80,.55)'); g.addColorStop(1, 'rgba(200,40,40,.35)'); }
    else if (over) { g.addColorStop(0, 'rgba(255,215,90,' + (.3 + .3 * pulse) + ')'); g.addColorStop(1, 'rgba(255,170,40,.18)'); }
    else { g.addColorStop(0, hover ? '#1f5547' : '#15382f'); g.addColorStop(1, hover ? '#18463a' : '#0f2a24'); }
    c.shadowColor = hover && !t.found ? 'hsla(' + hue() + ',90%,65%,.7)' : 'rgba(0,0,0,.35)'; c.shadowBlur = hover && !t.found ? 14 : 5; c.shadowOffsetY = 2;
    c.fillStyle = g; c.beginPath(); c.roundRect(-w / 2, -w / 2, w, w, Math.max(5, w * .16)); c.fill(); c.shadowColor = 'transparent';
    c.strokeStyle = t.found ? 'rgba(124,255,178,.7)' : 'hsla(' + hue() + ',60%,60%,' + (hover ? .8 : .22) + ')'; c.lineWidth = 1.4; c.stroke();
    c.save(); c.rotate(t.rot);
    if (t.blur) c.filter = 'blur(' + t.blur.toFixed(1) + 'px)';
    c.font = (t.italic ? 'italic ' : '') + t.weight + ' ' + Math.round(w * .66 * t.sc) + 'px ' + t.font;
    c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillStyle = t.found ? 'rgba(190,255,215,.32)' : '#eafff4'; c.fillText(String(t.digit), 0, 1);
    c.restore();
    if (t.dots) { c.fillStyle = 'rgba(255,255,255,.4)'; t.dots.forEach(([x, y]) => c.fillRect((x - .5) * w, (y - .5) * w, 1.6, 1.6)); }
    if (t.found) {
      c.strokeStyle = 'rgba(150,255,190,.95)'; c.lineWidth = Math.max(2, w * .06); c.lineCap = 'round'; c.lineJoin = 'round';
      c.beginPath(); c.moveTo(-w * .2, w * .02); c.lineTo(-w * .05, w * .17); c.lineTo(w * .22, -w * .14); c.stroke();
    }
    c.restore();
  }

  function glowText(c, text, x, y, size, col, align, blur) {
    c.save(); c.font = 'bold ' + size + 'px ' + MONO; c.textAlign = align || 'left'; c.textBaseline = 'alphabetic';
    c.shadowColor = 'rgba(' + col + ',.85)'; c.shadowBlur = blur === undefined ? 12 : blur; c.fillStyle = 'rgb(' + col + ')'; c.fillText(text, x, y); c.restore();
  }

  function drawHud(c, now) {
    const h = hue(), acc = 'hsl(' + h + ',90%,66%)';
    glowText(c, String(G.score).padStart(5, '0'), 18, 38, 30, '234,255,243');
    const mult = Math.min(1 + Math.floor(G.combo / 3) * .5, 5);
    glowText(c, G.combo ? 'x' + mult.toFixed(1) + '  streak ' + G.combo : 'streak 0', 20, 56, 12, G.combo ? '255,215,90' : '120,160,140', 'left', G.combo ? 6 : 0);
    // la cible : le chiffre à trouver
    c.save(); c.translate(W / 2 - 14, 34);
    c.fillStyle = 'rgba(255,255,255,.07)'; c.beginPath(); c.roundRect(-30, -28, 60, 56, 14); c.fill();
    c.strokeStyle = acc; c.lineWidth = 2.2; c.shadowColor = acc; c.shadowBlur = 14; c.stroke();
    c.font = 'bold 44px Georgia,serif'; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillStyle = acc; c.fillText(String(G.target), 0, 3); c.restore();
    glowText(c, 'FIND', W / 2 - 62, 24, 12, '170,255,205', 'right', 0);
    glowText(c, 'x' + G.need, W / 2 + 28, 44, 24, G.need === 1 ? '255,215,90' : '234,255,243', 'left', 6);
    glowText(c, 'ROUND ' + G.round, W - 18, 28, 14, '170,255,205', 'right', 0);
    const low = G.time < 6;
    glowText(c, G.time.toFixed(1) + 's', W - 18, 54, 20, low ? '255,100,100' : '124,255,178', 'right', low ? 10 : 4);
    // barre de temps
    const bx = 14, bw = W - 28, f = clamp(G.time / MAX_TIME, 0, 1);
    c.fillStyle = 'rgba(255,255,255,.1)'; c.beginPath(); c.roundRect(bx, 66, bw, 9, 5); c.fill();
    const bar = c.createLinearGradient(bx, 0, bx + bw, 0);
    bar.addColorStop(0, low ? '#ff5a5a' : '#37e08a'); bar.addColorStop(1, low ? '#ff9a5a' : '#9dffcf');
    c.save(); if (low) c.globalAlpha = .65 + .35 * Math.sin(now / 90);
    c.fillStyle = bar; c.beginPath(); c.roundRect(bx, 66, Math.max(10, bw * f), 9, 5); c.fill(); c.restore();
    c.fillStyle = 'rgba(124,255,178,.5)'; c.font = '11px ' + MONO; c.textAlign = 'left'; c.fillText('DIGIT CLASSIFIER · you are the model', 14, H - 4);
  }

  function panel(c, w, h) {
    const x = (W - w) / 2, y = (H - h) / 2;
    c.fillStyle = 'rgba(3,12,8,.9)'; c.beginPath(); c.roundRect(x, y, w, h, 16); c.fill();
    c.strokeStyle = 'rgba(80,255,160,.45)'; c.lineWidth = 1.5; c.stroke();
  }

  function mostConfused() {
    let bestK = null, bestN = 0;
    for (const k in G.confused) if (G.confused[k] > bestN) { bestN = G.confused[k]; bestK = k; }
    if (!bestK) return null;
    const [t, d] = bestK.split('>');
    return 'you mistook ' + d + ' for ' + t + ' x' + bestN;
  }

  function drawOverlay(c, now) {
    const pulse = .55 + .45 * Math.sin(now / 280);
    if (G.state === 'ready') {
      panel(c, 560, 300);
      const g = c.createLinearGradient(0, 110, 0, 170); g.addColorStop(0, '#d8ffe9'); g.addColorStop(1, '#41e88a');
      c.save(); c.font = 'bold 58px ' + MONO; c.textAlign = 'center'; c.shadowColor = 'rgba(60,255,150,.9)'; c.shadowBlur = 24; c.fillStyle = g; c.fillText('DIGIT DASH', W / 2, 165); c.restore();
      c.fillStyle = '#b6e8cc'; c.font = '15px ' + MONO; c.textAlign = 'center';
      ['be the classifier: click every digit shown at the top', 'each hit adds time · each miss costs ' + MISS_COST + 's',
        'rotated, blurred, noisy fonts... and look-alikes (6 / 9!)'].forEach((t, i) => c.fillText(t, W / 2, 205 + i * 24));
      c.fillStyle = 'rgba(255,215,90,' + pulse + ')'; c.font = 'bold 18px ' + MONO; c.fillText('click or press Enter', W / 2, 308);
      const top = tops(); if (top.length) { c.fillStyle = '#6fbf91'; c.font = '13px ' + MONO; c.fillText('best ' + top[0], W / 2, 332); }
    } else if (G.state === 'over') {
      panel(c, 580, 370);
      glowText(c, "TIME'S UP", W / 2, 118, 34, '255,215,90', 'center', 18);
      glowText(c, String(G.score), W / 2, 178, 58, '234,255,243', 'center', 18);
      if (G.newBest) glowText(c, '★ NEW BEST ★', W / 2, 205, 16, '255,215,90', 'center', 10);
      else if (G.rank > 0) glowText(c, 'top ' + G.rank + ' run', W / 2, 205, 15, '124,255,178', 'center', 6);
      const tot = G.hits + G.miss, acc = tot ? Math.round(100 * G.hits / tot) : 0;
      c.fillStyle = '#9fdcb9'; c.font = '14px ' + MONO; c.textAlign = 'center';
      c.fillText(G.rounds + ' rounds · accuracy ' + acc + '% · best streak ' + G.bestCombo + ' · ' + Math.floor(G.playT) + 's', W / 2, 236);
      const conf = mostConfused();
      c.fillStyle = conf ? '#ff9a9a' : '#9dffcf'; c.fillText(conf || 'no confusion at all: a perfect classifier', W / 2, 260);
      c.fillStyle = '#6fbf91'; c.font = '13px ' + MONO;
      c.fillText(tops().map((v, i) => (i + 1) + '. ' + v).join('   '), W / 2, 292);
      c.fillStyle = 'rgba(255,215,90,' + pulse + ')'; c.font = 'bold 17px ' + MONO; c.fillText('click or press Enter to retry', W / 2, 336);
    }
  }

  function drawBoxFx(c) {
    G.boxes.forEach((b) => {
      const a = 1 - (G.t - b.t) / .7, col = b.ok ? '60,255,140' : '255,90,90', x = b.x - b.s / 2 - 2, y = b.y - b.s / 2 - 2, s = b.s + 4, l = s * .28;
      c.save(); c.strokeStyle = 'rgba(' + col + ',' + a + ')'; c.lineWidth = 2.5; c.shadowColor = 'rgba(' + col + ',.9)'; c.shadowBlur = 8;
      [[x, y, 1, 1], [x + s, y, -1, 1], [x, y + s, 1, -1], [x + s, y + s, -1, -1]].forEach(([px, py, dx, dy]) => {
        c.beginPath(); c.moveTo(px + dx * l, py); c.lineTo(px, py); c.lineTo(px, py + dy * l); c.stroke();
      });
      c.shadowBlur = 0; c.font = 'bold 11px ' + MONO; const w = c.measureText(b.label).width + 10, ly = y > TOP + 4 ? y - 16 : y + s + 1;
      c.fillStyle = 'rgba(' + col + ',' + a + ')'; c.fillRect(x, ly, w, 15); c.fillStyle = 'rgba(4,16,10,' + a + ')'; c.textAlign = 'left'; c.textBaseline = 'middle'; c.fillText(b.label, x + 5, ly + 8);
      c.restore();
    });
  }

  function loop(now) {
    raf = P.requestAnimationFrame(loop);
    const dt = Math.min((now - (G.last || now)) / 1000, .05); G.last = now;
    update(dt);
    const c = G.ctx; c.setTransform(G.dpr, 0, 0, G.dpr, 0, 0);
    const bg = c.createLinearGradient(0, 0, 0, H); bg.addColorStop(0, '#07201b'); bg.addColorStop(1, '#0d2f27');
    c.fillStyle = bg; c.fillRect(0, 0, W, H);
    c.save();
    if (G.shake > 0) c.translate((Math.random() - .5) * 12 * G.shake, (Math.random() - .5) * 12 * G.shake);
    c.fillStyle = 'rgba(160,255,200,.08)'; for (let i = 0; i < 40; i++) c.fillRect(Math.random() * W, Math.random() * H, 1.3, 1.3);
    const hover = G.state === 'play' && G.introT <= 0 && G.clearT <= 0 ? tileAt(G.mx, G.my) : null, pulse = .5 + .5 * Math.sin(now / 200);
    c.save(); if (G.state === 'ready') c.globalAlpha = .4;
    G.tiles.forEach((t) => drawTile(c, t, t === hover, pulse));
    c.restore();
    drawBoxFx(c);
    G.parts.forEach((p) => { c.fillStyle = 'rgba(' + p.col + ',' + clamp(p.life * 2, 0, 1) + ')'; c.fillRect(p.x, p.y, p.s, p.s); });
    G.floats.forEach((f) => glowText(c, f.text, f.x, f.y, f.size, f.col, 'center', 8));
    c.restore();
    if (G.flash > 0) { c.fillStyle = 'rgba(' + G.flashCol + ',' + G.flash * .7 + ')'; c.fillRect(0, 0, W, H); }
    if (G.state !== 'ready') drawHud(c, now);
    drawOverlay(c, now);
    c.fillStyle = c.createPattern(scan, 'repeat'); c.fillRect(0, 0, W, H);
    const v = c.createRadialGradient(W / 2, H / 2, H * .4, W / 2, H / 2, W * .66); v.addColorStop(0, 'rgba(0,0,0,0)'); v.addColorStop(1, 'rgba(0,0,0,.3)');
    c.fillStyle = v; c.fillRect(0, 0, W, H);
    root.dataset.state = G.state; root.dataset.score = G.score; root.dataset.round = G.round; root.dataset.combo = G.combo; root.dataset.time = G.time.toFixed(1);
    root._g = G; root._dbg = { update, click, start };  // accès pour tester la difficulté sans attendre le temps réel
  }
})();
"""


def install() -> None:
    """Branche l'extra sur la page. L'iframe est vide et sort du flux : elle ne décale rien."""
    with st.container(key="extras"):
        components.html(f"<script>{SCRIPT}</script>", height=0)
