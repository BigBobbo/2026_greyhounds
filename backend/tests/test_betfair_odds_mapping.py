"""Unit tests for the Betfair odds-capture mapping logic (no network)."""

from datetime import date, time, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from scraping.betfair_odds import (
    imminent,
    market_local_date_time,
    match_market_to_race,
    normalise_venue,
    parse_runner_trap,
    snapshot_rows,
)


def test_imminent_window_excludes_started_and_distant_markets():
    now = datetime.now(timezone.utc)

    def market(minutes_from_now):
        start = now + timedelta(minutes=minutes_from_now)
        return {"marketStartTime": start.isoformat().replace("+00:00", "Z")}

    markets = [market(-30), market(5), market(90), market(240)]
    picked = imminent(markets, 120)
    assert len(picked) == 2  # the +5 and +90 only
    assert imminent(markets, 10) == [markets[1]]
    assert imminent(markets, 0) == []


class Row:
    def __init__(self, id, race_time, race_number, track_name):
        self.id = id
        self.race_time = race_time
        self.race_number = race_number
        self.track_name = track_name


class Entry:
    def __init__(self, trap, dog_id):
        self.trap = trap
        self.dog_id = dog_id


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    yield db
    db.close()


@pytest.fixture
def seeded_race(db_session):
    """One 20:14 Irish race with six traps. August is UTC+1 in Dublin, so
    the equivalent Betfair marketStartTime is 19:14 UTC."""
    from app.models.dog import Dog
    from app.models.race import Race
    from app.models.race_entry import RaceEntry
    from app.models.track import Track

    track = Track(name="Shelbourne Park", code="SHP", location="Dublin")
    db_session.add(track)
    db_session.flush()

    race = Race(track_id=track.id, race_date=date(2026, 8, 20),
                race_time=time(20, 14), race_number=5, distance_m=525,
                status="scheduled")
    db_session.add(race)
    db_session.flush()

    entries = []
    for trap in range(1, 7):
        dog = Dog(name=f"DOG {trap}")
        db_session.add(dog)
        db_session.flush()
        entry = RaceEntry(race_id=race.id, dog_id=dog.id, trap=trap)
        db_session.add(entry)
        entries.append(entry)
    db_session.commit()

    return SimpleNamespace(
        id=race.id,
        track_name="Shelbourne Park",
        market_start_iso="2026-08-20T19:14:00.000Z",
    ), entries


def test_ingest_maps_agent_payload_to_snapshots(db_session, seeded_race):
    """The agent forwards venue/time/runner-name/price; the server does
    all the matching. One market in -> one snapshot per priced runner."""
    from scraping.betfair_odds import ingest_snapshots
    from app.models.odds import OddsSnapshot

    race, entries = seeded_race
    payload = [{
        "market_id": "1.999",
        "venue": race.track_name,
        "market_start_time": race.market_start_iso,
        "runners": [
            {"runner_name": f"{e.trap}. Dog {e.trap}", "price": 2.0 + e.trap}
            for e in entries
        ] + [{"runner_name": "9. Not In This Race", "price": 5.0}],
    }]
    result = ingest_snapshots(db_session, payload)

    assert result["markets_matched"] == 1
    assert result["markets_unmatched"] == 0
    # the trap-9 runner has no matching entry and is dropped
    assert result["snapshots_written"] == len(entries)

    rows = db_session.query(OddsSnapshot).all()
    assert {r.dog_id for r in rows} == {e.dog_id for e in entries}
    assert all(r.bookmaker == "betfair_exchange" for r in rows)
    assert all(not r.is_sp for r in rows)
    by_dog = {r.dog_id: r for r in rows}
    first = entries[0]
    assert by_dog[first.dog_id].odds_decimal == 2.0 + first.trap
    assert abs(by_dog[first.dog_id].implied_prob
               - 1 / (2.0 + first.trap)) < 1e-9


def test_ingest_reports_unmatched_markets(db_session, seeded_race):
    """A venue we don't run reports back rather than failing silently —
    that's how a track-name mismatch becomes visible."""
    from scraping.betfair_odds import ingest_snapshots

    race, _ = seeded_race
    result = ingest_snapshots(db_session, [{
        "venue": "Nowhere Park",
        "market_start_time": race.market_start_iso,
        "runners": [{"runner_name": "1. Ghost", "price": 3.0}],
    }])
    assert result["snapshots_written"] == 0
    assert result["markets_unmatched"] == 1
    assert "Nowhere Park" in result["unmatched"][0]


def test_ingest_skips_invalid_prices(db_session, seeded_race):
    from scraping.betfair_odds import ingest_snapshots

    race, entries = seeded_race
    result = ingest_snapshots(db_session, [{
        "venue": race.track_name,
        "market_start_time": race.market_start_iso,
        "runners": [
            {"runner_name": f"{entries[0].trap}. A", "price": 1.0},
            {"runner_name": f"{entries[1].trap}. B", "price": 0},
            {"runner_name": f"{entries[2].trap}. C", "price": 4.5},
        ],
    }])
    assert result["snapshots_written"] == 1


def test_trap_parsing():
    assert parse_runner_trap("1. Ballymac Star") == 1
    assert parse_runner_trap(" 6 . Slippy Maska") == 6
    assert parse_runner_trap("Ballymac Star") is None


def test_venue_aliases():
    assert normalise_venue("Curraheen") == "Curraheen Park"
    assert normalise_venue("Shelbourne Park ") == "Shelbourne Park"
    assert normalise_venue("Youghal") == "Youghal"


def test_market_time_converts_to_irish_local():
    # 19:14 UTC in July = 20:14 Irish summer time
    d, hhmm = market_local_date_time("2026-07-20T19:14:00.000Z")
    assert hhmm == "20:14"
    assert str(d) == "2026-07-20"


def test_match_by_venue_and_time():
    races = [
        Row(1, time(20, 14), 5, "Shelbourne Park"),
        Row(2, time(20, 30), 6, "Shelbourne Park"),
        Row(3, time(20, 14), 4, "Youghal"),
    ]
    market = {
        "event": {"venue": "Shelbourne"},
        "marketStartTime": "2026-07-20T19:14:00.000Z",
    }
    assert match_market_to_race(market, races).id == 1


def test_match_tolerates_small_time_drift():
    races = [Row(1, time(20, 16), 5, "Youghal")]
    market = {
        "event": {"venue": "Youghal"},
        "marketStartTime": "2026-07-20T19:14:00.000Z",  # 20:14 local
    }
    assert match_market_to_race(market, races).id == 1


def test_no_match_across_venues():
    races = [Row(1, time(20, 14), 5, "Kilkenny")]
    market = {
        "event": {"venue": "Youghal"},
        "marketStartTime": "2026-07-20T19:14:00.000Z",
    }
    assert match_market_to_race(market, races) is None


def test_snapshot_rows_map_traps_and_prices():
    catalogue = {"runners": [
        {"selectionId": 101, "runnerName": "1. Fast Dog"},
        {"selectionId": 102, "runnerName": "2. Slow Dog"},
        {"selectionId": 103, "runnerName": "3. No Price"},
    ]}
    book = {"runners": [
        {"selectionId": 101, "status": "ACTIVE",
         "ex": {"availableToBack": [{"price": 3.5, "size": 20}]}},
        {"selectionId": 102, "status": "ACTIVE",
         "ex": {"availableToBack": [{"price": 6.0, "size": 8}]}},
        {"selectionId": 103, "status": "ACTIVE", "ex": {"availableToBack": []}},
    ]}
    entries = {1: Entry(1, 501), 2: Entry(2, 502), 3: Entry(3, 503)}
    rows = snapshot_rows(book, catalogue, race_id=42, entries_by_trap=entries,
                         scraped_at=datetime(2026, 7, 20, 19, 0))
    assert len(rows) == 2
    assert rows[0]["race_id"] == 42
    assert rows[0]["dog_id"] == 501
    assert rows[0]["odds_decimal"] == 3.5
    assert abs(rows[0]["implied_prob"] - 1 / 3.5) < 1e-9
    assert rows[1]["dog_id"] == 502
    assert rows[0]["is_sp"] is False
