"""Same-origin SPA serving: assets, client-route fallback, API/health untouched."""

from __future__ import annotations

from test_api_support import make_client, zero_spend  # noqa: F401


def test_spa_fallback_and_assets(tmp_path):
    static = tmp_path / "dist"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
    (static / "assets" / "app-123.js").write_text("console.log(1)", encoding="utf-8")
    (static / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
    client = make_client(tmp_path, static_dir=str(static))

    root = client.get("/")
    assert root.status_code == 200 and "id=root" in root.text
    assert "id=root" in client.get("/sessions/abc").text  # client-side route
    assert client.get("/assets/app-123.js").text == "console.log(1)"
    assert client.get("/favicon.svg").text == "<svg/>"
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/api/auth/mode").json() == {"mode": "none"}
    missing_api = client.get("/api/does-not-exist")
    assert missing_api.status_code == 404 and missing_api.json()["code"] == "not_found"
    assert "nope" not in client.get("/..%2Fsecret.txt").text


def test_no_static_dir_means_api_only(tmp_path):
    client = make_client(tmp_path)  # static_dir points at a missing folder
    assert client.get("/").status_code == 404
    assert client.get("/health").json() == {"status": "ok"}


def test_cors_allows_dev_origin_without_credentials(tmp_path):
    client = make_client(tmp_path)
    response = client.options(
        "/api/sessions",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-credentials" not in response.headers
    assert "authorization" in response.headers["access-control-allow-headers"].lower()
