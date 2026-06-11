"""Golden tests for the GRI scraper parsers (audit tasks E1/E2/E3/E14/E15).

Fixture provenance: grireland.ie is NOT reachable from this sandbox (egress
allowlist returns 403 "Host not in allowlist"), so the HTML files under
tests/fixtures/ are SYNTHETIC. They were constructed to exactly match the
markup scraping/gri_scraper.py documents and parses: <h4> "Race N" headers,
igb-tbl tables with the documented column order, <img alt="Trap N"> trap
encoding, viewresults-pedigree-* spans, GRI's malformed missing-</td>
greyhound cell, and the ASP.NET search form with its track dropdown.
"""

import asyncio
from datetime import date, time
from pathlib import Path

import httpx
import pytest
from bs4 import BeautifulSoup

from scraping.gri_scraper import (
    ParseStructureError,
    ScrapeFetchError,
    _parse_card_header,
    _parse_race_header,
    _parse_result_table,
    _parse_sp_decimal,
    parse_card_form_page,
    parse_card_page,
    parse_results_page,
    scrape_results,
)

FIXTURES = Path(__file__).parent / "fixtures"
RACE_DATE = date(2026, 6, 5)


def load_fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


@pytest.fixture(scope="module")
def results_races() -> list[dict]:
    return parse_results_page(
        load_fixture("results_spk_2026-06-05.html"), "SPK", RACE_DATE
    )


def _by_trap(race: dict) -> dict[int, dict]:
    return {e["trap"]: e for e in race["entries"]}


# ---------------------------------------------------------------------------
# Results page golden assertions (E3)
# ---------------------------------------------------------------------------


def test_results_race_count(results_races):
    assert len(results_races) == 2


def test_results_race1_header(results_races):
    race = results_races[0]
    assert race["race_number"] == 1
    # "The 600 Final" sponsor number must NOT win over the real 525 distance
    assert race["distance_m"] == 525
    assert race["grade"] == "A2"
    assert race["race_type"] == "flat"
    assert race["track_code"] == "SPK"
    assert race["race_date"] == RACE_DATE
    assert race["going"] == "-10"
    assert race["prize_money"] == 220.0
    assert len(race["entries"]) == 6


def test_results_winner_row_with_malformed_markup(results_races):
    """The winner's greyhound cell is missing its closing </td> (real GRI
    quirk) — every column must still be attributed correctly."""
    winner = _by_trap(results_races[0])[4]
    assert winner["finish_position"] == 1
    assert winner["dog_name"] == "BALLYMAC VISION"
    assert winner["sire_name"] == "LAUGHIL BLAKE"
    # Mixed-case link text in the fixture — must be uppercased
    assert winner["dam_name"] == "BALLYMAC ARRA"
    assert winner["prize_money"] == 220.0
    assert winner["weight_kg"] == 32.5
    assert winner["win_time"] == 28.92
    assert winner["finish_time"] == 28.92
    assert winner["going"] == "-10"
    assert winner["starting_price"] == "5/2F"
    assert winner["sp_decimal"] == 3.5  # 5/2 favourite
    assert winner["grade_at_entry"] == "A2"
    assert winner["comment"] == "QAw,Crd1"
    assert winner.get("beaten_distance") is None  # winner has empty "By"


def test_results_sp_decimal_conversions(results_races):
    by_trap = _by_trap(results_races[0])
    assert by_trap[4]["sp_decimal"] == 3.5    # 5/2F
    assert by_trap[1]["sp_decimal"] == 4.0    # 3/1
    assert by_trap[2]["sp_decimal"] == 4.5    # 7/2
    assert by_trap[3]["sp_decimal"] == 4.33   # 10/3
    assert by_trap[5]["sp_decimal"] == 9.0    # 8/1
    assert by_trap[6]["sp_decimal"] == 7.0    # 6/1


def test_results_finish_positions_and_times(results_races):
    by_trap = _by_trap(results_races[0])
    assert by_trap[1]["finish_position"] == 2
    assert by_trap[1]["finish_time"] == 29.10
    assert by_trap[1]["beaten_distance"] == 2.5
    assert by_trap[1]["weight_kg"] == 27.8
    assert by_trap[2]["finish_position"] == 3
    assert by_trap[2]["finish_time"] == 29.18


def test_results_non_finisher(results_races):
    """Trap 6 has no finish position and no est time — both must be absent,
    everything else still parsed."""
    dnf = _by_trap(results_races[0])[6]
    assert dnf["dog_name"] == "KILMORE DASHER"
    assert dnf.get("finish_position") is None
    assert dnf.get("finish_time") is None
    assert dnf["sp_decimal"] == 7.0
    assert dnf["comment"] == "Stopped,DNF"


def test_results_pedigree_scoped_to_row(results_races):
    """E15: trap 1's row has NO dam span. It must get dam=None — and trap 2
    (the next dog in the table) must keep its OWN dam, not have it stolen by
    the document-order find_next walk the old parser used."""
    by_trap = _by_trap(results_races[0])
    assert by_trap[1]["sire_name"] == "DROOPYS SYDNEY"
    assert by_trap[1].get("dam_name") is None
    assert by_trap[2]["sire_name"] == "BALLYMAC BEST"
    assert by_trap[2]["dam_name"] == "COOLAVANNY MISS"


def test_results_marathon_race(results_races):
    """E14: 1010-yard marathon must pass the widened distance gate."""
    race = results_races[1]
    assert race["race_number"] == 2
    assert race["distance_m"] == 1010
    assert race["grade"] == "M1"
    assert len(race["entries"]) == 6
    winner = _by_trap(race)[2]
    assert winner["starting_price"] == "evens"
    assert winner["sp_decimal"] == 2.0
    assert winner["win_time"] == 61.45  # marathon times pass the 15-70s gate


# ---------------------------------------------------------------------------
# SP decimal conversion (direct unit tests)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sp_text,expected",
    [
        ("evens", 2.0),
        ("Evens", 2.0),
        ("evs", 2.0),
        ("EVS", 2.0),
        ("5/2", 3.5),
        ("5/2F", 3.5),
        ("10/3J", 4.33),
        ("4/1", 5.0),
        ("1/2", 1.5),
        ("0/1", None),       # decimal <= 1.0 is non-sensical
        ("garbage", None),
        ("No Price", None),
        ("", None),
        ("-", None),
    ],
)
def test_parse_sp_decimal(sp_text, expected):
    assert _parse_sp_decimal(sp_text) == expected


# ---------------------------------------------------------------------------
# Header distance parsing (E14)
# ---------------------------------------------------------------------------


def test_results_header_prefers_last_distance():
    info = _parse_race_header("Race 1 - The 600 Final 525 (Grade : A2) Flat 525")
    assert info["distance_m"] == 525
    assert info["race_number"] == 1
    assert info["grade"] == "A2"


def test_results_header_marathon_distance():
    info = _parse_race_header("Race 4 - Night Owl Marathon 1010 (Grade : M1) Flat 1010")
    assert info["distance_m"] == 1010


def test_card_header_marathon_distance():
    info = _parse_card_header(
        "Race 2 SHELBOURNE STAYERS 20:08 Approx. (1010 Yds. Flat) (Race Grade : S3)"
    )
    assert info["distance_m"] == 1010
    assert info["race_time"] == time(20, 8)


def test_card_header_fallback_prefers_last_distance():
    info = _parse_card_header("Race 3 The 600 Final 20:00 Approx. 525 Flat")
    assert info["distance_m"] == 525


# ---------------------------------------------------------------------------
# Loud failure on markup/network problems (E1)
# ---------------------------------------------------------------------------


def test_empty_day_returns_empty_list():
    html = load_fixture("results_empty_day.html")
    assert parse_results_page(html, "SPK", RACE_DATE) == []
    assert parse_card_page(html, "SPK", RACE_DATE) == []


def test_broken_structure_raises():
    html = load_fixture("results_broken_structure.html")
    with pytest.raises(ParseStructureError):
        parse_results_page(html, "SPK", RACE_DATE)
    with pytest.raises(ParseStructureError):
        parse_card_page(html, "SPK", RACE_DATE)


def test_non_gri_page_raises():
    html = "<html><body><h1>503 Service Unavailable</h1></body></html>"
    with pytest.raises(ParseStructureError):
        parse_results_page(html, "SPK", RACE_DATE)


def test_race_headers_without_tables_raises():
    html = (
        "<html><body>"
        "<select id='ContentPlaceHolder1_ddlTrack'><option>SPK</option></select>"
        "<h4>Race 1 - Sprint 525 (Grade : A4) Flat 525</h4>"
        "</body></html>"
    )
    with pytest.raises(ParseStructureError):
        parse_results_page(html, "SPK", RACE_DATE)


def _run(coro):
    return asyncio.run(coro)


def test_scrape_results_raises_on_non_200():
    async def go():
        transport = httpx.MockTransport(lambda req: httpx.Response(500))
        async with httpx.AsyncClient(transport=transport) as client:
            await scrape_results("SPK", RACE_DATE, client)

    with pytest.raises(ScrapeFetchError):
        _run(go())


def test_scrape_results_raises_on_network_error():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await scrape_results("SPK", RACE_DATE, client)

    with pytest.raises(ScrapeFetchError):
        _run(go())


def test_scrape_results_parses_fixture_over_http():
    html = load_fixture("results_spk_2026-06-05.html")

    async def go():
        transport = httpx.MockTransport(lambda req: httpx.Response(200, text=html))
        async with httpx.AsyncClient(transport=transport) as client:
            return await scrape_results("SPK", RACE_DATE, client)

    races = _run(go())
    assert len(races) == 2
    assert races[0]["distance_m"] == 525


# ---------------------------------------------------------------------------
# Header-keyed column parsing + sanity ranges (E2)
# ---------------------------------------------------------------------------


def test_extra_column_does_not_shift_values():
    """A new 'Split' column inserted between Wt. and Win Time must not shift
    any value attribution (the old tail-count parser would have shifted
    everything before it by one)."""
    races = parse_results_page(
        load_fixture("results_extra_column.html"), "SPK", RACE_DATE
    )
    assert len(races) == 1
    by_trap = _by_trap(races[0])

    winner = by_trap[3]
    assert winner["dog_name"] == "QUICK SPLIT"
    assert winner["weight_kg"] == 31.2
    assert winner["win_time"] == 29.31
    assert winner["finish_time"] == 29.31
    assert winner["starting_price"] == "2/1F"
    assert winner["sp_decimal"] == 3.0
    assert winner["grade_at_entry"] == "A4"
    assert winner["comment"] == "QAw,ALd"
    assert winner["prize_money"] == 185.0

    second = by_trap[6]
    assert second["weight_kg"] == 28.6
    assert second["beaten_distance"] == 3.0
    assert second["sp_decimal"] == 5.0
    assert second["comment"] == "Crd1,RanOn"


def test_renamed_headers_raise_structure_error():
    html = """
    <table class="igb-tbl">
      <tr><th>Foo</th><th>Bar</th><th>Baz</th><th>Qux</th></tr>
      <tr><td>1.</td><td><img alt="Trap 1"/></td>
          <td><a href="x">SOME DOG</a></td><td>5/2</td></tr>
    </table>
    """
    table = BeautifulSoup(html, "html.parser").find("table")
    with pytest.raises(ParseStructureError):
        _parse_result_table(table)


def test_headerless_table_falls_back_to_legacy_positions(caplog):
    html = """
    <table class="igb-tbl">
      <tr><td>Results</td></tr>
      <tr>
        <td>1.</td>
        <td><img alt="Trap 2"/></td>
        <td><a href="x">OLD STYLE DOG</a></td>
        <td><span class="viewresults-pedigree-sire"><a href="s">SIRE X</a></span></td>
        <td><span class="viewresults-pedigree-dam"><a href="d">DAM X</a></span></td>
        <td>150</td><td>29.3</td><td>28.99</td><td></td><td>-20</td>
        <td>28.99</td><td>4/1</td><td>A5</td><td>EvAw</td>
      </tr>
    </table>
    """
    table = BeautifulSoup(html, "html.parser").find("table")
    with caplog.at_level("WARNING"):
        entries = _parse_result_table(table)
    assert any("legacy" in rec.message for rec in caplog.records)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["dog_name"] == "OLD STYLE DOG"
    assert entry["trap"] == 2
    assert entry["weight_kg"] == 29.3
    assert entry["win_time"] == 28.99
    assert entry["sp_decimal"] == 5.0
    assert entry["grade_at_entry"] == "A5"
    assert entry["comment"] == "EvAw"
    assert entry["sire_name"] == "SIRE X"
    assert entry["dam_name"] == "DAM X"


def test_out_of_range_values_become_none(caplog):
    html = """
    <table class="igb-tbl">
      <tr>
        <th>Pos.</th><th>Trap</th><th>Greyhound</th><th>Prize</th><th>Wt.</th>
        <th>Win Time</th><th>By</th><th>Going</th><th>Est Time</th>
        <th>SP.</th><th>Grade</th><th>Comm.</th>
      </tr>
      <tr>
        <td>1.</td>
        <td><img alt="Trap 1"/></td>
        <td><a href="x">GARBAGE ROW</a></td>
        <td>100</td>
        <td>99.0</td>
        <td>9.2</td>
        <td></td>
        <td>N</td>
        <td>75.0</td>
        <td>0/1</td>
        <td>A1</td>
        <td>Ok</td>
      </tr>
    </table>
    """
    table = BeautifulSoup(html, "html.parser").find("table")
    with caplog.at_level("WARNING"):
        entries = _parse_result_table(table)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.get("weight_kg") is None      # 99.0 kg out of 20-40
    assert entry.get("win_time") is None       # 9.2 s out of 15-70
    assert entry.get("finish_time") is None    # 75.0 s out of 15-70
    assert entry.get("sp_decimal") is None     # 0/1 -> 1.0, not > 1.0
    # Sanity-checked fields are dropped, not entire rows
    assert entry["dog_name"] == "GARBAGE ROW"
    assert any("out-of-range" in rec.message.lower() for rec in caplog.records)


# ---------------------------------------------------------------------------
# Card page + card form golden assertions (E3)
# ---------------------------------------------------------------------------


def test_card_page_golden():
    races = parse_card_page(
        load_fixture("card_spk_2026-06-12.html"), "SPK", date(2026, 6, 12)
    )
    assert len(races) == 2

    race1 = races[0]
    assert race1["race_number"] == 1
    assert race1["race_time"] == time(19, 40)
    assert race1["distance_m"] == 525
    assert race1["grade"] == "A3"
    assert race1["status"] == "scheduled"
    assert [e["trap"] for e in race1["entries"]] == [1, 2, 3, 4, 5, 6]
    assert race1["entries"][0]["dog_name"] == "CLONBRIEN ROCKET"
    assert race1["entries"][5]["dog_name"] == "SKYWALKER HOPE"

    race2 = races[1]
    assert race2["race_number"] == 2
    assert race2["distance_m"] == 1010  # marathon card distance
    assert race2["race_time"] == time(20, 8)
    assert len(race2["entries"]) == 6


def test_card_form_page_golden():
    by_trap = parse_card_form_page(load_fixture("card_form_spk_r1.html"))
    assert set(by_trap.keys()) == {1, 2}

    t1 = by_trap[1]
    assert t1["owner_name"] == "Mr John Murphy"
    assert t1["trainer_name"] == "Pat Guiry"
    # Sire/dam uppercased consistently with the results parser
    assert t1["sire_name"] == "DROOPYS SYDNEY"
    assert t1["dam_name"] == "CLONBRIEN TREACLE"
    assert t1["best_time"] == 28.55

    t2 = by_trap[2]
    assert t2["owner_name"] == "Tyrur Syndicate"
    assert t2["trainer_name"] == "P J Fahy"
    assert t2["sire_name"] == "BALLYMAC BEST"
    assert t2["dam_name"] == "TYRUR PEGGY SUE"
    assert t2["best_time"] == 28.91
