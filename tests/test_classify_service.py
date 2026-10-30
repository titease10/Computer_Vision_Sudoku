import base64
import json
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

import classify_service


class FakeClassifier:
    """Lit un « chiffre » dans la luminosité de l'image : niveau 52 -> 2, niveau 130 -> 5."""

    def predict_probs(self, cells):
        out = np.zeros((len(cells), 10), np.float32)
        for k, c in enumerate(cells):
            out[k, int(c.mean() // 26) % 10] = 1.0
        return out


def png(level: int) -> str:
    ok, buf = cv2.imencode(".png", np.full((20, 20, 3), level, np.uint8))
    assert ok
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


@pytest.fixture(autouse=True)
def fake_model():
    classify_service.attach(FakeClassifier())
    yield
    classify_service.attach(None)


def test_classify_returns_class_and_confidence_of_each_image():
    out = classify_service.classify([png(52), png(130)])
    assert out["pred"] == [2, 5] and out["conf"] == [1.0, 1.0] and out["n"] == 2 and out["ms"] >= 0


def test_classify_rejects_bad_input_and_missing_model():
    with pytest.raises(ValueError):
        classify_service.classify([])
    with pytest.raises(ValueError):
        classify_service.classify(["data:image/png;base64," + base64.b64encode(b"pas une image").decode()])
    classify_service.attach(None)
    with pytest.raises(RuntimeError):
        classify_service.classify([png(52)])


def test_http_roundtrip_with_cors():
    server = classify_service.start("127.0.0.1", 0)  # port 0 : le système en choisit un libre
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        req = urllib.request.Request(base + "/classify", data=json.dumps({"images": [png(78)]}).encode(),
                                     headers={"Content-Type": "text/plain"})
        with urllib.request.urlopen(req) as r:
            assert r.headers["Access-Control-Allow-Origin"] == "*"
            assert json.loads(r.read())["pred"] == [3]
        with urllib.request.urlopen(base + "/health") as r:
            assert json.loads(r.read()) == {"ready": True}
        bad = urllib.request.Request(base + "/classify", data=b"{}", headers={"Content-Type": "text/plain"})
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(bad)
        assert err.value.code == 400
    finally:
        server.shutdown()
