"""Deep health-check tests (audit task B3)."""

from fastapi.testclient import TestClient
from sqlalchemy import text

from app.database import engine
from app.main import app, _migration_head

client = TestClient(app)


def test_health_ok_reports_matching_migration():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["migration"]["head"] is not None
    assert body["migration"]["current"] == body["migration"]["head"]
    assert body["version"] not in ("", "unknown")


def test_health_503_on_migration_mismatch():
    head = _migration_head()
    assert head is not None
    with engine.begin() as conn:
        conn.execute(text("UPDATE alembic_version SET version_num = 'deadbeef'"))
    try:
        resp = client.get("/api/health")
        assert resp.status_code == 503
        assert resp.json()["status"] == "error"
    finally:
        with engine.begin() as conn:
            conn.execute(text("UPDATE alembic_version SET version_num = :v"), {"v": head})
    assert client.get("/api/health").status_code == 200
