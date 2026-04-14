from __future__ import annotations

from fastapi.testclient import TestClient

from dualstream.api import app


def test_root_and_static_assets_served() -> None:
    client = TestClient(app)

    root = client.get("/")
    assert root.status_code == 200
    assert "DualStream Local UI" in root.text

    js = client.get("/static/app.js")
    assert js.status_code == 200
    assert 'postJSON("/preflight/generate"' in js.text


def test_ui_uses_same_origin_relative_api_urls() -> None:
    client = TestClient(app)

    js = client.get("/static/app.js")
    assert "http://127.0.0.1" not in js.text
    assert 'postJSON("/generate"' in js.text


def test_preflight_endpoint_available() -> None:
    client = TestClient(app)
    response = client.post("/preflight/generate", json={"prompt": "x", "model": "gpt2", "offline": False})
    assert response.status_code == 200
    data = response.json()
    assert data["kind"] == "generate"
