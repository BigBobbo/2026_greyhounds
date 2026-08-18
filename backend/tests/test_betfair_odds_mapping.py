"""Unit tests for the Betfair odds-capture mapping logic (no network)."""

from datetime import time, datetime, timedelta, timezone

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
