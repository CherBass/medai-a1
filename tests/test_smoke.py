from fastapi.testclient import TestClient

from src.api.main import app


def test_app_metadata() -> None:
    assert app.title == "medai-a1"
    assert app.version == "0.1.0"


def test_healthz_ok() -> None:
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
