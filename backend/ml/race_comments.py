"""Structured parser for greyhound race comments.

GRI-style race comments encode rich qualitative information about each
run — running style, break quality, trip shape (which bend the dog led
at, where trouble happened), stamina (finished well / faded), and rail
vs wide preference.  Historically this feature set only parsed a
handful of front-runner and trouble tokens; this module expands parsing
to the full industry vocabulary so downstream aggregates can expose
running-style rates, bend-by-bend trouble counts, stamina markers, and
more.

The parser returns a simple dict of boolean flags and a set of bend
numbers, which the feature builder aggregates per dog over the last N
races.

Comment abbreviations vary slightly across data sources (GRI /
GreyhoundData / Timeform).  The regex dictionary here accepts common
synonyms (`Ld1`, `Ld 1`, `Led 1`; `Ck`, `Chk`, `Checked`; etc.) and is
case-insensitive.  Unknown tokens are silently ignored.
"""

from __future__ import annotations

import re
from typing import TypedDict


class ParsedComment(TypedDict):
    # Running style markers (from training-stats section of form lines)
    is_early_pace: bool        # EP
    is_mid_pace: bool          # MP / MidP
    is_late_pace: bool         # LP / closer
    # Break quality at the boxes
    quick_away: bool           # QAw, Fl T, F/T, Qk Aw
    slow_away: bool            # SAw, Slw Aw, Missed Break
    awkward_start: bool        # Awk, Stb, Stumbled
    # Trip shape — which bends did the dog actually lead at?
    led_bends: set[int]        # {1,2,3,4}; "½" maps to 2, "¾" to 3
    disputed_lead: bool        # Disp Ld, DisW
    # Stamina markers
    finished_well: bool        # Fin Wl, RnOn, Ran On, Stay, Stayed, Kpt On
    faded: bool                # Fd, Wknd, Weakened, Tired
    cleared_field: bool        # Clr (won by daylight)
    # Trouble markers — bend-by-bend
    trouble_bends: set[int]    # {1..4} for Ck/Bmp/Crd/Hmp/Baulk at bend N
    trouble_unspecified: bool  # trouble tokens without a bend marker
    # Positioning preference
    railed: bool               # Rls, Rails, RlsTo
    ran_wide: bool             # W/R, Wd, Wide
    # Edge cases
    fell: bool                 # Fell
    waited: bool               # Wtd (trapped behind)
    short_of_room: bool        # SCd, ShtCd


def _empty() -> ParsedComment:
    return ParsedComment(
        is_early_pace=False,
        is_mid_pace=False,
        is_late_pace=False,
        quick_away=False,
        slow_away=False,
        awkward_start=False,
        led_bends=set(),
        disputed_lead=False,
        finished_well=False,
        faded=False,
        cleared_field=False,
        trouble_bends=set(),
        trouble_unspecified=False,
        railed=False,
        ran_wide=False,
        fell=False,
        waited=False,
        short_of_room=False,
    )


# --- Regex fragments ---------------------------------------------------------
# Bend suffix: accepts a digit 1-4, ½/¾ (treated as 2/3), or literal "str"
# (stretch, i.e. after the final bend, treated as 4).
_BEND_SUFFIX = r"\s*(?:(1)|(2)|(3)|(4)|(½|1/2|half)|(¾|3/4)|(str|stretch))"


def _bend_from_groups(groups: tuple) -> int | None:
    if groups[0]:
        return 1
    if groups[1]:
        return 2
    if groups[2]:
        return 3
    if groups[3]:
        return 4
    if groups[4]:  # half
        return 2
    if groups[5]:  # 3/4
        return 3
    if groups[6]:  # stretch
        return 4
    return None


# Running-style — standalone whole-word tokens at the start of the comment
_EP_RE = re.compile(r"(?:^|[\s,;/])ep(?:[\s,;/]|$)", re.IGNORECASE)
_MP_RE = re.compile(r"(?:^|[\s,;/])m(?:id)?p(?:[\s,;/]|$)", re.IGNORECASE)
_LP_RE = re.compile(r"(?:^|[\s,;/])lp(?:[\s,;/]|$)", re.IGNORECASE)

# Break quality
_QUICK_AWAY_RE = re.compile(
    r"(?:^|[\s,;/])(?:q\s*aw|q\.?a\.?w|quick\s*aw(?:ay)?|fl(?:ew)?\s*t|f/t|qk\s*aw)",
    re.IGNORECASE,
)
_SLOW_AWAY_RE = re.compile(
    r"(?:^|[\s,;/])(?:s\s*aw|s\.?a\.?w|slow\s*aw(?:ay)?|slw\s*aw|missed\s*br)",
    re.IGNORECASE,
)
_AWKWARD_RE = re.compile(
    r"(?:^|[\s,;/])(?:awk|stb|stumbl|stumbled)",
    re.IGNORECASE,
)

# Led at bend — "Ld 1", "Ld1", "Led 2", "Led-3", "Ld½"
_LED_BEND_RE = re.compile(
    r"\b(?:ld|led)" + _BEND_SUFFIX,
    re.IGNORECASE,
)
# Led across a range — "Ld1-4" meaning led from bend 1 to bend 4 inclusive.
_LED_RANGE_RE = re.compile(
    r"\b(?:ld|led)\s*([1-4])\s*-\s*([1-4])\b",
    re.IGNORECASE,
)
# Disputed lead
_DISPUTED_RE = re.compile(
    r"(?:disp\s*(?:ld|lead)|\bdisw\b|dispute(?:d)?\s*lead)",
    re.IGNORECASE,
)

# Stamina
_FINISH_WELL_RE = re.compile(
    r"(?:fin\s*wl|fin\.?\s*well|ran?\s*on|rnon|\bstay(?:ed)?\b|kpt\s*on|kept\s*on)",
    re.IGNORECASE,
)
_FADED_RE = re.compile(
    r"(?:\bfd\b|\bfad(?:ed)?\b|\bwknd\b|weakened|\btird\b|\btired\b|\bwk\b|no\s*ext)",
    re.IGNORECASE,
)
_CLEARED_RE = re.compile(
    r"(?:\bclr\b|\bclear\b|by\s*(?:daylight|lengths?))",
    re.IGNORECASE,
)

# Trouble — captures bend number if present.  Tokens: ck/chk/checked,
# bmp/bumped, crd/crowded, hmp/hamp(ered), blk/baulk(ed)
_TROUBLE_WITH_BEND_RE = re.compile(
    r"\b(?:ck|chk|checked|bmp|bumped|crd|crowded|hmp|hamp(?:ered)?|"
    r"blk|baulk(?:ed)?)" + _BEND_SUFFIX,
    re.IGNORECASE,
)
_TROUBLE_ANY_RE = re.compile(
    r"\b(?:ck|chk|checked|bmp|bumped|crd|crowded|hmp|hamp(?:ered)?|"
    r"blk|baulk(?:ed)?)\b",
    re.IGNORECASE,
)

# Positioning
_RAILED_RE = re.compile(r"\b(?:rls|rls\s*to|railed?|rails)\b", re.IGNORECASE)
_WIDE_RE = re.compile(r"(?:\bw/r\b|\bwd\b|\bwide\b)", re.IGNORECASE)

# Edge cases
_FELL_RE = re.compile(r"\bfell\b", re.IGNORECASE)
_WAITED_RE = re.compile(r"\bwtd\b|\bwaited\b", re.IGNORECASE)
_SHORT_OF_ROOM_RE = re.compile(
    r"(?:\bscd\b|\bshtcd\b|\bsht\.?\s*crd\b|short\s*of\s*room)",
    re.IGNORECASE,
)


def parse_race_comment(raw: object) -> ParsedComment:
    """Parse a raw comment string into structured flags.

    Handles None / empty / non-string input gracefully by returning
    an empty-flags dict.
    """
    out = _empty()
    if raw is None:
        return out
    text = str(raw).strip()
    if not text:
        return out

    if _EP_RE.search(text):
        out["is_early_pace"] = True
    if _MP_RE.search(text):
        out["is_mid_pace"] = True
    if _LP_RE.search(text):
        out["is_late_pace"] = True

    if _QUICK_AWAY_RE.search(text):
        out["quick_away"] = True
    if _SLOW_AWAY_RE.search(text):
        out["slow_away"] = True
    if _AWKWARD_RE.search(text):
        out["awkward_start"] = True

    # Ranges first (so "Ld1-4" fills {1,2,3,4}); then single bends to pick up
    # any Ld-token not caught by the range expression.
    for m in _LED_RANGE_RE.finditer(text):
        lo, hi = int(m.group(1)), int(m.group(2))
        if hi >= lo:
            for b in range(lo, hi + 1):
                out["led_bends"].add(b)
    for m in _LED_BEND_RE.finditer(text):
        bend = _bend_from_groups(m.groups())
        if bend is not None:
            out["led_bends"].add(bend)
    if _DISPUTED_RE.search(text):
        out["disputed_lead"] = True

    if _FINISH_WELL_RE.search(text):
        out["finished_well"] = True
    if _FADED_RE.search(text):
        out["faded"] = True
    if _CLEARED_RE.search(text):
        out["cleared_field"] = True

    for m in _TROUBLE_WITH_BEND_RE.finditer(text):
        bend = _bend_from_groups(m.groups())
        if bend is not None:
            out["trouble_bends"].add(bend)
    if _TROUBLE_ANY_RE.search(text) and not out["trouble_bends"]:
        # Trouble word with no bend number — still useful as a binary flag
        out["trouble_unspecified"] = True

    if _RAILED_RE.search(text):
        out["railed"] = True
    if _WIDE_RE.search(text):
        out["ran_wide"] = True

    if _FELL_RE.search(text):
        out["fell"] = True
    if _WAITED_RE.search(text):
        out["waited"] = True
    if _SHORT_OF_ROOM_RE.search(text):
        out["short_of_room"] = True

    return out


# Names of the per-dog aggregated features emitted by the batch pipeline
# (aggregation lives in ml/race_features.py).  Exposed here so the UI can
# list them and the test suite can assert against a canonical set.
COMMENT_FEATURE_NAMES = [
    # Running-style (fraction of last 10 runs marked with the style token)
    "running_style_ep_rate_last10",
    "running_style_mp_rate_last10",
    "running_style_lp_rate_last10",
    # Break quality
    "quick_away_rate_last10",
    "slow_away_rate_last10",
    "awkward_start_rate_last10",
    # Trip shape — rate of leading at each bend, plus a trip trend flag
    "led_at_bend1_rate_last10",
    "led_at_bend2_rate_last10",
    "led_at_bend3_rate_last10",
    "led_at_bend4_rate_last10",
    "disputed_lead_rate_last10",
    # Stamina
    "finish_well_rate_last10",
    "faded_rate_last10",
    # Location-specific trouble
    "trouble_bend1_rate_last10",
    "trouble_bend2_rate_last10",
    "trouble_bend3_rate_last10",
    "trouble_bend4_rate_last10",
    # Positioning
    "railed_rate_last10",
    "ran_wide_rate_last10",
    # Clean winning
    "clear_win_rate_last10",
]
