"""API-key authentication tests (audit task A2).

The dependency short-circuits before any route logic runs, so these tests
do not need a populated database: a 401 proves the guard fired, and any
non-401 status proves the request was allowed through.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture
def client():
    # No context manager: avoids running the lifespan (scheduler startup).
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def with_api_key(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "test-secret-key")


PROTECTED_SAMPLES = [
    ("GET", "/api/tracks/"),
    ("POST", "/api/bankroll/reset"),
    ("DELETE", "/api/training/experiments/1"),
    ("POST", "/api/scraping/trigger"),
    ("GET", "/api/predictions/history"),
]


@pytest.mark.parametrize("method,path", PROTECTED_SAMPLES)
def test_protected_routes_reject_missing_key(client, with_api_key, method, path):
    resp = client.request(method, path)
    assert resp.status_code == 401, f"{method} {path} returned {resp.status_code}"


@pytest.mark.parametrize("method,path", PROTECTED_SAMPLES)
def test_protected_routes_reject_wrong_key(client, with_api_key, method, path):
    resp = client.request(method, path, headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401


def test_correct_key_passes_auth(client, with_api_key):
    resp = client.get("/api/tracks/", headers={"X-API-Key": "test-secret-key"})
    assert resp.status_code != 401


def test_health_is_open(client, with_api_key):
    resp = client.get("/api/health")
    assert resp.status_code == 200


def test_auth_disabled_when_key_unset(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "")
    resp = client.get("/api/tracks/")
    assert resp.status_code != 401


def test_no_get_route_mutates_state():
    """Audit I1: every known state-changing route must not be GET.

    GETs are crawlable, prefetchable, and cacheable; these routes run
    models, persist predictions, or launch background jobs.
    """
    mutating_paths = {
        "/api/predictions/race/{race_id}": "POST",
        "/api/predictions/by-date": "POST",
        "/api/predictions/best-bets": "POST",
        "/api/predictions/upcoming": "POST",
        "/api/predictions/race/{race_id}/ensemble": "POST",
        "/api/predictions/race/{race_id}/combos": "POST",
        "/api/features/materialize": "POST",
    }
    from app.main import app as _app

    routes = {}
    for route in _app.routes:
        methods = getattr(route, "methods", None)
        if methods:
            routes.setdefault(route.path, set()).update(methods)

    for path, method in mutating_paths.items():
        assert path in routes, f"route {path} missing"
        assert "GET" not in routes[path], f"{path} still accepts GET"
        assert method in routes[path]

    assert "/api/features/start-materialize" not in routes
