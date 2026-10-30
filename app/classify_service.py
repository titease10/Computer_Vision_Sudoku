"""Petit service HTTP de classification de chiffres.

Le jeu (JavaScript, dans le navigateur) y envoie les images d'un tableau de chiffres, et le vrai classifieur du
projet répond. Le modèle reste donc en Python, chargé une seule fois par l'application.

    POST /classify   {"images": ["data:image/png;base64,...", ...]}
                  -> {"pred": [...], "conf": [...], "ms": temps d'inférence, "n": ..., "model": ...}
    GET  /health     -> {"ready": true|false}

Par défaut le service n'écoute que sur 127.0.0.1 (SNAP_AI_HOST / SNAP_AI_PORT pour changer, par exemple dans Docker).
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np

from sudoku_cv.cells import preprocess_cell

HOST = os.environ.get("SNAP_AI_HOST", "127.0.0.1")
PORT = int(os.environ.get("SNAP_AI_PORT", "8512"))
MAX_IMAGES = 200
MAX_BODY = 8 * 1024 * 1024

_state: dict = {"classifier": None}
_lock = threading.Lock()  # un seul appel au modèle à la fois


def attach(classifier) -> None:
    """Branche le classifieur chargé par l'application (objet avec `predict_probs`)."""
    _state["classifier"] = classifier


def classify(images: list[str]) -> dict:
    """Images PNG (data URL ou base64) de chiffres foncés sur fond clair -> classe et confiance de chacune."""
    clf = _state["classifier"]
    if clf is None:
        raise RuntimeError("le classifieur n'est pas encore chargé")
    if not 0 < len(images) <= MAX_IMAGES:
        raise ValueError(f"entre 1 et {MAX_IMAGES} images attendues")
    cells = []
    for s in images:
        raw = base64.b64decode(s.split(",", 1)[-1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("image illisible")
        cells.append(preprocess_cell(img))
    t0 = time.perf_counter()
    with _lock:
        probs = clf.predict_probs(cells)
    ms = (time.perf_counter() - t0) * 1000
    return {"pred": probs.argmax(axis=1).tolist(), "conf": np.round(probs.max(axis=1), 3).tolist(),
            "ms": round(ms, 1), "n": len(cells), "model": "YOLO11n-cls"}


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: Optional[dict] = None) -> None:
        body = json.dumps(payload).encode() if payload is not None else b""
        self.send_response(code)
        self.send_header("Access-Control-Allow-Origin", "*")  # la page et le service n'ont pas le même port
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        if body:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send(204)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {"ready": _state["classifier"] is not None})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/classify":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY:
                self._send(413, {"error": "corps trop gros"})
                return
            body = json.loads(self.rfile.read(length))
            self._send(200, classify(body["images"]))
        except RuntimeError as exc:
            self._send(503, {"error": str(exc)})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})

    def log_message(self, *args) -> None:  # pas de ligne de log par requête
        pass


def start(host: str = HOST, port: int = PORT) -> Optional[ThreadingHTTPServer]:
    """Lance le service dans un thread. None si le port est déjà pris (par exemple par une autre instance)."""
    try:
        server = ThreadingHTTPServer((host, port), _Handler)
    except OSError:
        return None
    threading.Thread(target=server.serve_forever, daemon=True, name="classify-service").start()
    return server
