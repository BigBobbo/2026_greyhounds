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


def test_busy_timeout_pragma_set():
    """Audit B4: concurrent writers must queue, not fail instantly."""
    from sqlalchemy import text

    with engine.connect() as conn:
        timeout = conn.execute(text("PRAGMA busy_timeout")).fetchone()[0]
    assert timeout == 30000


def test_concurrent_writers_do_not_lock():
    """Two threads inserting through the app engine must both succeed."""
    import threading

    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE IF NOT EXISTS _writelock_test (id INTEGER PRIMARY KEY, v TEXT)")
        )

    errors: list[Exception] = []

    def writer(tag: str):
        try:
            for i in range(30):
                with engine.begin() as conn:
                    conn.execute(
                        text("INSERT INTO _writelock_test (v) VALUES (:v)"),
                        {"v": f"{tag}-{i}"},
                    )
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(t,)) for t in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent writes failed: {errors}"
    with engine.begin() as conn:
        n = conn.execute(text("SELECT count(*) FROM _writelock_test")).fetchone()[0]
        conn.execute(text("DROP TABLE _writelock_test"))
    assert n == 60
