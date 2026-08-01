"""Tests for the greyhound race-comment parser."""

import pytest

from ml.race_comments import parse_race_comment


def test_empty_inputs():
    for v in (None, "", "   "):
        p = parse_race_comment(v)
        assert p["is_early_pace"] is False
        assert p["led_bends"] == set()
        assert p["trouble_bends"] == set()
        assert p["finished_well"] is False


def test_non_string_input_is_safe():
    # pandas can pass floats (NaN) through — parser must not crash
    p = parse_race_comment(float("nan"))
    assert p["is_early_pace"] is False


def test_classic_early_pace_led_first_bend():
    p = parse_race_comment("EP,Ld 1")
    assert p["is_early_pace"] is True
    assert p["led_bends"] == {1}
    assert p["is_mid_pace"] is False
    assert p["is_late_pace"] is False


def test_running_style_tokens_are_whole_words():
    # "epic" should NOT match EP running style
    p = parse_race_comment("ran an epic race")
    assert p["is_early_pace"] is False
    # Similarly "comp" shouldn't trip MP
    p2 = parse_race_comment("competitive run")
    assert p2["is_mid_pace"] is False


def test_mid_and_late_pace():
    assert parse_race_comment("MP, finished well")["is_mid_pace"] is True
    assert parse_race_comment("MidP pressed on")["is_mid_pace"] is True
    assert parse_race_comment("LP closer, RnOn late")["is_late_pace"] is True


def test_led_at_multiple_bends():
    p = parse_race_comment("QAw Ld1 Ld2 Ld3 Clr")
    assert p["led_bends"] == {1, 2, 3}
    assert p["quick_away"] is True
    assert p["cleared_field"] is True


def test_led_at_half_and_three_quarter_maps_to_bends():
    p = parse_race_comment("Ld½ fd")
    assert p["led_bends"] == {2}
    assert p["faded"] is True

    p2 = parse_race_comment("Ld 3/4 Clr")
    assert p2["led_bends"] == {3}


def test_disputed_lead():
    assert parse_race_comment("Disp Ld, Ld 2")["disputed_lead"] is True
    assert parse_race_comment("DisW")["disputed_lead"] is True


def test_break_quality_tokens():
    p = parse_race_comment("QAw Ld-1")
    assert p["quick_away"] is True
    assert p["slow_away"] is False

    p2 = parse_race_comment("SAw Ck1 Crd2")
    assert p2["slow_away"] is True
    assert p2["quick_away"] is False
    assert p2["trouble_bends"] == {1, 2}

    p3 = parse_race_comment("Awk Stb")
    assert p3["awkward_start"] is True


def test_trouble_with_and_without_bend():
    # With bend number
    assert parse_race_comment("Ck 3 Fd")["trouble_bends"] == {3}
    # Without bend number — flag the generic indicator instead
    p = parse_race_comment("Crowded, Faded")
    assert p["trouble_bends"] == set()
    assert p["trouble_unspecified"] is True
    assert p["faded"] is True


def test_stamina_markers():
    p = parse_race_comment("MP, RnOn late, Stayed")
    assert p["finished_well"] is True
    assert p["faded"] is False

    p2 = parse_race_comment("EP led then Wknd, tired late")
    assert p2["finished_well"] is False
    assert p2["faded"] is True


def test_cleared_field():
    assert parse_race_comment("QAw Ld-4 Clr")["cleared_field"] is True
    assert parse_race_comment("Won by clear 3 lengths")["cleared_field"] is True


def test_rail_and_wide():
    p = parse_race_comment("Rls, Ld 2")
    assert p["railed"] is True
    p2 = parse_race_comment("W/R, Crd 1")
    assert p2["ran_wide"] is True
    # Both can appear (unusually) — parser shouldn't conflict
    p3 = parse_race_comment("Rls then Wd late")
    assert p3["railed"] is True and p3["ran_wide"] is True


def test_fell_and_waited():
    assert parse_race_comment("Ld1 Fell 2")["fell"] is True
    assert parse_race_comment("Wtd behind leaders")["waited"] is True
    assert parse_race_comment("SCd at bend 3")["short_of_room"] is True


def test_case_insensitive_and_punctuation_tolerant():
    # All-caps and mixed case both work
    p = parse_race_comment("ep,LD1,CLR")
    assert p["is_early_pace"] is True
    assert p["led_bends"] == {1}
    assert p["cleared_field"] is True


def test_unknown_tokens_are_ignored():
    # Arbitrary free-text shouldn't raise or produce false positives
    p = parse_race_comment("gamely finished in the middle group of runners")
    # None of the primary flags should trip on generic prose
    assert p["is_early_pace"] is False
    assert p["is_mid_pace"] is False
    assert p["is_late_pace"] is False
    assert p["led_bends"] == set()


def test_real_world_examples():
    # A handful of realistic-looking combinations
    examples = [
        ("EP Ld1-4 Clr", {"is_early_pace", "led_bends:1", "led_bends:2",
                          "led_bends:3", "led_bends:4", "cleared_field"}),
        ("SAw Bmp1 Crd2 Fd", {"slow_away", "trouble_bends:1",
                              "trouble_bends:2", "faded"}),
        ("MP Ck3 RnOn", {"is_mid_pace", "trouble_bends:3", "finished_well"}),
        ("LP Wtd Fin Wl", {"is_late_pace", "waited", "finished_well"}),
    ]
    for raw, expected in examples:
        parsed = parse_race_comment(raw)
        actual = set()
        for k, v in parsed.items():
            if isinstance(v, bool) and v:
                actual.add(k)
            elif isinstance(v, set):
                for b in v:
                    actual.add(f"{k}:{b}")
        missing = expected - actual
        assert not missing, f"For '{raw}': missing flags {missing}"


# --- Beaten-margin parsing (close finishes carry the most information) ---

from scraping.gri_scraper import parse_beaten_margin


class TestBeatenMargin:
    def test_named_margins(self):
        assert parse_beaten_margin("SH") == 0.05
        assert parse_beaten_margin("HD") == 0.1
        assert parse_beaten_margin("NK") == 0.25
        assert parse_beaten_margin("DH") == 0.0

    def test_distance_is_unquantified(self):
        assert parse_beaten_margin("DIS") is None
        assert parse_beaten_margin("DIST") is None

    def test_vulgar_fractions(self):
        assert parse_beaten_margin("½") == 0.5
        assert parse_beaten_margin("1½") == 1.5
        assert parse_beaten_margin("2¾") == 2.75
        assert parse_beaten_margin("¼") == 0.25

    def test_ascii_fractions(self):
        assert parse_beaten_margin("1 1/2") == 1.5
        assert parse_beaten_margin("3/4") == 0.75

    def test_plain_numbers_and_length_suffix(self):
        assert parse_beaten_margin("3") == 3.0
        assert parse_beaten_margin("2.5") == 2.5
        assert parse_beaten_margin("4L") == 4.0

    def test_garbage_is_none(self):
        assert parse_beaten_margin("") is None
        assert parse_beaten_margin("N/A") is None
