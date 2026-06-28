"""
FIFA World Cup 2026 — Match Predictor & Dashboard
=================================================
Streamlit app that predicts every match in the 2026 FIFA World Cup
(Canada / Mexico / USA) using historical World Cup data.

Files expected in the same folder as app.py (or repo root):
    - world-cup-Data-all-matches-2.csv          (historical match results, 1930-)
    - WCup_2026_4.2.5_en.xlsx                   (2026 draw / fixture file)
    - world_cup_2026_all_nations_qualifiers.csv (2026 qualifying campaign results)
    - results_2026.csv                          (optional, results as they come in)

The results_2026.csv has columns:
    match_no,home_team,away_team,home_score,away_score
It can also be uploaded live from the sidebar (no file edit needed).

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling — adapts to both light and dark Streamlit themes.
# We use CSS variables so colours flip automatically with @media
# (prefers-color-scheme), keeping good contrast in either mode.
st.markdown(
    """
    <style>
    /* ---- Light mode (default) ---- */
    :root {
        --wc-heading: #0a1f3d;       /* deep navy — readable on white */
        --wc-accent:  #c8102e;       /* World Cup red */
        --wc-card-bg: rgba(10,31,61,0.04);
        --wc-card-bd: rgba(10,31,61,0.12);
        --wc-team-a:  #1f4e8c;       /* navy blue */
        --wc-team-b:  #c8102e;       /* red */
        --wc-draw:    #6b7280;       /* neutral grey */
        --wc-bar-text:#ffffff;
    }
    /* ---- Dark mode override ---- */
    @media (prefers-color-scheme: dark) {
        :root {
            --wc-heading: #f5d76e;   /* warm gold on dark background */
            --wc-accent:  #ff5470;
            --wc-card-bg: rgba(255,255,255,0.04);
            --wc-card-bd: rgba(255,255,255,0.10);
            --wc-team-a:  #4a90e2;
            --wc-team-b:  #ff5470;
            --wc-draw:    #9ca3af;
            --wc-bar-text:#0a1f3d;
        }
    }

    h1, h2, h3 { color: var(--wc-heading) !important; }

    .metric-card {
        background: var(--wc-card-bg);
        border: 1px solid var(--wc-card-bd);
        border-radius: 12px;
        padding: 16px;
    }
    .prob-bar { height: 26px; border-radius: 6px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants — file paths (relative so they work on Streamlit Cloud / GitHub)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
HIST_CSV = ROOT / "world-cup-Data-all-matches-2.csv"
DRAW_XLSX = ROOT / "WCup_2026_4.2.5_en.xlsx"
RESULTS_CSV = ROOT / "results_2026.csv"
QUALIFIERS_CSV = ROOT / "world_cup_2026_all_nations_qualifiers.csv"

RESULTS_TEMPLATE_COLUMNS = [
    "match_no", "home_team", "away_team", "home_score", "away_score",
]

# ---- Historical name resolution ----
#
# Some 2026 teams played at past World Cups under one or more *different*
# names — either because the country was renamed (Zaire → DR Congo) or
# because it succeeded a larger state that has since dissolved
# (Czechoslovakia → Czech Republic + Slovakia, USSR → Russia, etc.).
#
# `HISTORICAL_NAMES` maps each 2026-draw team name to *every* name it has
# played under in the historical dataset. The first entry is the team's
# own modern name in the CSV; subsequent entries are predecessor states
# whose World Cup record is inherited.
#
# Successor relationships follow FIFA's own treatment of national-team
# records (e.g. Germany inherits West Germany's record; the modern
# Russia federation inherits the Soviet Union's; Serbia inherits
# Yugoslavia and Serbia and Montenegro).
HISTORICAL_NAMES: dict[str, list[str]] = {
    # ---- Spelling-only renames (one-to-one) ----
    "Rep. of Korea":   ["South Korea"],
    "USA":             ["USA", "United States"],
    "IR Iran":         ["Iran"],
    "Bosnia/Herzeg.":  ["Bosnia and Herzegovina", "Bosnia & Herz."],
    "Ivory Coast":     ["Ivory Coast"],
    "Turkey":          ["Turkey"],          # Türkiye in some sources
    # ---- Successor-state inheritances ----
    "Czech Rep.":      ["Czech Republic", "Czechoslovakia"],
    "Germany":         ["Germany", "West Germany"],
    "Russia":          ["Russia", "Soviet Union"],
    "Serbia":          ["Serbia", "Serbia and Montenegro", "Yugoslavia"],
    "DR Congo":        ["DR Congo", "Zaire"],
    # ---- True debutants (kept here for completeness) ----
    "Cape Verde":      ["Cape Verde"],
    "Curaçao":         ["Curaçao"],
    "Jordan":          ["Jordan"],
    "Uzbekistan":      ["Uzbekistan"],
}

# Teams making their genuine FIFA World Cup debut in 2026 (no historical
# data under any name). Used for documentation only — the prediction
# engine detects debutants automatically by checking the dataset.
DEBUTANTS_2026: set[str] = {
    "Cape Verde", "Curaçao", "Jordan", "Uzbekistan",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Qualifying-campaign data (2026 cycle)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_qualifiers() -> pd.DataFrame:
    """Return a tidy dataframe of 2026 qualifying matches.

    The raw CSV lists each match from one team's perspective. Some matches
    appear twice (once for each side) which we de-duplicate, then we
    explode again into one row per team-match (long format), matching
    the layout used by `load_history`.
    """
    if not QUALIFIERS_CSV.exists():
        return pd.DataFrame(
            columns=["team", "opponent", "gf", "ga", "result", "region"]
        )

    raw = pd.read_csv(QUALIFIERS_CSV)

    # Parse "Team A vs Team B" + "x-y" into structured columns
    parsed = raw["Match"].str.split(" vs ", n=1, expand=True)
    raw = raw.assign(
        home=parsed[0].str.strip(),
        away=parsed[1].str.strip(),
    )
    # Strip any trailing notation like ' (4-1 pen)' or ' (aet)' — we keep
    # the regulation-time scoreline, which is what matters for goal averages.
    clean = raw["Scoreline"].astype(str).str.split("(").str[0].str.strip()
    score = clean.str.split("-", n=1, expand=True)
    raw["home_score"] = pd.to_numeric(score[0], errors="coerce")
    raw["away_score"] = pd.to_numeric(score[1], errors="coerce")
    raw = raw.dropna(subset=["home_score", "away_score"])
    raw["home_score"] = raw["home_score"].astype(int)
    raw["away_score"] = raw["away_score"].astype(int)

    # Normalise spelling differences between qualifier files and the 2026 draw
    # so qualifier_profile() lookups succeed for every team.
    QUAL_NAME_FIXES = {
        "Bosnia & Herz.": "Bosnia and Herzegovina",
    }
    for col in ["home", "away"]:
        raw[col] = raw[col].replace(QUAL_NAME_FIXES)

    # De-duplicate: a unique match is the unordered pair + scoreline
    raw["_pair"] = raw.apply(
        lambda r: tuple(sorted([r["home"], r["away"]])) + (r["home_score"], r["away_score"]),
        axis=1,
    )
    unique = raw.drop_duplicates(subset=["_pair"]).copy()

    # Long format: two rows per match (one for each team)
    home_rows = pd.DataFrame({
        "team":     unique["home"],
        "opponent": unique["away"],
        "gf":       unique["home_score"],
        "ga":       unique["away_score"],
        "region":   unique["Region"],
    })
    away_rows = pd.DataFrame({
        "team":     unique["away"],
        "opponent": unique["home"],
        "gf":       unique["away_score"],
        "ga":       unique["home_score"],
        "region":   unique["Region"],
    })
    tidy = pd.concat([home_rows, away_rows], ignore_index=True)
    tidy["result"] = np.select(
        [tidy["gf"] > tidy["ga"], tidy["gf"] < tidy["ga"]],
        ["W", "L"],
        default="D",
    )
    return tidy


def qualifier_profile(qual: pd.DataFrame, team: str) -> dict | None:
    """Return {matches, avg_gf, avg_ga} for a team in qualifying, or None.

    Looks up under every name the team has played under (so e.g.
    'Rep. of Korea' resolves to 'South Korea' in the qualifying CSV).
    """
    names = hist_names(team)
    df = qual[qual["team"].isin(names)]
    if df.empty:
        return None
    return {
        "matches": int(len(df)),
        "avg_gf": float(df["gf"].mean()),
        "avg_ga": float(df["ga"].mean()),
        "wins":   int((df["result"] == "W").sum()),
        "draws":  int((df["result"] == "D").sum()),
        "losses": int((df["result"] == "L").sum()),
        "games":  df.sort_index().reset_index(drop=True),
    }


@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    """Return tidy historical match dataframe with one row per team-match.

    The CSV has one row per match (home/away). We "explode" it so each
    team appears once per match — easier for averaging goals scored /
    conceded and head-to-head lookups.
    """
    df = pd.read_csv(HIST_CSV)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["year"] = df["match_date"].dt.year

    home = df[
        [
            "year", "tournament_id", "stage_name",
            "home_team_name", "away_team_name",
            "home_team_score", "away_team_score",
        ]
    ].copy()
    home.columns = ["year", "tournament_id", "stage", "team", "opponent", "gf", "ga"]
    home["venue"] = "home"

    away = df[
        [
            "year", "tournament_id", "stage_name",
            "away_team_name", "home_team_name",
            "away_team_score", "home_team_score",
        ]
    ].copy()
    away.columns = ["year", "tournament_id", "stage", "team", "opponent", "gf", "ga"]
    away["venue"] = "away"

    tidy = pd.concat([home, away], ignore_index=True)
    tidy["result"] = np.select(
        [tidy["gf"] > tidy["ga"], tidy["gf"] < tidy["ga"]],
        ["W", "L"],
        default="D",
    )
    return tidy


def load_results_2026(uploaded=None) -> pd.DataFrame:
    """Return a clean dataframe of 2026 results.

    Source priority:
      1. an uploaded file from the sidebar (Streamlit UploadedFile)
      2. results_2026.csv next to app.py
      3. empty dataframe with the right schema
    """
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
    else:
        return pd.DataFrame(columns=RESULTS_TEMPLATE_COLUMNS)

    # Validate columns and coerce types
    missing = [c for c in RESULTS_TEMPLATE_COLUMNS if c not in df.columns]
    if missing:
        st.sidebar.error(
            f"Results CSV is missing column(s): {missing}. "
            f"Required: {RESULTS_TEMPLATE_COLUMNS}"
        )
        return pd.DataFrame(columns=RESULTS_TEMPLATE_COLUMNS)
    df = df[RESULTS_TEMPLATE_COLUMNS].copy()
    df["match_no"] = pd.to_numeric(df["match_no"], errors="coerce").astype("Int64")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").astype("Int64")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])
    return df


def merge_results_into_history(
    hist: pd.DataFrame, results: pd.DataFrame
) -> pd.DataFrame:
    """Append 2026 results to the historical (tidy) dataframe.

    `hist` is the long-format dataframe produced by `load_history` —
    one row per team-match. We add two rows per 2026 result so the
    prediction engine treats those games like any other World Cup match.

    Each team is stored under its primary historical name so the lookup
    keys stay consistent (e.g. a 2026 result for “Rep. of Korea” is
    written as “South Korea”, matching the historical CSV).
    """
    if results.empty:
        return hist
    rows = []
    for _, r in results.iterrows():
        h, a = r["home_team"], r["away_team"]
        hg, ag = int(r["home_score"]), int(r["away_score"])
        h_canon = hist_name(h)  # primary historical spelling
        a_canon = hist_name(a)
        rows.append({
            "year": 2026, "tournament_id": "WC-2026", "stage": "Group",
            "team": h_canon, "opponent": a_canon, "gf": hg, "ga": ag,
            "venue": "home",
            "result": "W" if hg > ag else ("L" if hg < ag else "D"),
        })
        rows.append({
            "year": 2026, "tournament_id": "WC-2026", "stage": "Group",
            "team": a_canon, "opponent": h_canon, "gf": ag, "ga": hg,
            "venue": "away",
            "result": "W" if ag > hg else ("L" if ag < hg else "D"),
        })
    return pd.concat([hist, pd.DataFrame(rows)], ignore_index=True)


@st.cache_data(show_spinner=False)
def load_draw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (groups_df, fixtures_df) parsed from the 2026 draw spreadsheet."""
    # ---- Groups sheet ----
    g_raw = pd.read_excel(DRAW_XLSX, sheet_name="Groups", header=None)
    rows = []
    current_group = None
    for _, r in g_raw.iterrows():
        slot, num, name = r[1], r[2], r[3]
        if isinstance(name, str) and len(name) == 1 and name.isalpha():
            current_group = name
            continue
        if isinstance(slot, str) and current_group and isinstance(name, str):
            rows.append({"group": current_group, "slot": slot, "team": name.strip()})
    groups = pd.DataFrame(rows)

    # ---- Matches sheet ----
    m_raw = pd.read_excel(DRAW_XLSX, sheet_name="Matches", header=None)
    fixtures = []
    for _, r in m_raw.iterrows():
        no = r[1]
        if not (isinstance(no, (int, float)) and pd.notna(no) and float(no).is_integer()):
            continue
        fixtures.append(
            {
                "match_no": int(no),
                "slot1": r[2],
                "slot2": r[3],
                "date_local": r[4],
                "venue": r[7],
                "team1": r[8] if isinstance(r[8], str) else None,
                "team2": r[9] if isinstance(r[9], str) else None,
                "stage_label": r[12] if isinstance(r[12], str) else "Group",
            }
        )
    fixtures = pd.DataFrame(fixtures)
    return groups, fixtures


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------
def hist_names(team: str) -> list[str]:
    """Return every historical name a 2026 team played under.

    For most teams this is just the team's own name. For successor states
    it includes predecessor names (e.g. Russia → [Russia, Soviet Union]).
    """
    return HISTORICAL_NAMES.get(team, [team])


def hist_name(team: str) -> str:
    """Return the team's primary historical name (for display purposes)."""
    return hist_names(team)[0]


def team_history(hist: pd.DataFrame, team: str) -> pd.DataFrame:
    """Slice of the tidy history dataframe for a team and all its predecessors."""
    names = hist_names(team)
    return hist[hist["team"].isin(names)]


def is_debutant(team: str, hist: pd.DataFrame) -> bool:
    """A team is a debutant if no matches exist under any of its names."""
    return team_history(hist, team).empty


# ---------------------------------------------------------------------------
# Statistics — team strength
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def team_stats(_hist: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-team historical World Cup statistics.

    Stats are keyed by **2026 team name** so successor states inherit
    the records of their predecessors (e.g. Germany's row covers both
    Germany and West Germany matches).
    """
    # Build a reverse lookup: every historical name → modern team name.
    rev: dict[str, str] = {}
    for modern, names in HISTORICAL_NAMES.items():
        for n in names:
            rev[n] = modern
    # Any team not in the mapping keeps its own name.
    df = _hist.copy()
    df["team_modern"] = df["team"].map(lambda t: rev.get(t, t))

    g = df.groupby("team_modern").agg(
        matches=("gf", "size"),
        wins=("result", lambda s: (s == "W").sum()),
        draws=("result", lambda s: (s == "D").sum()),
        losses=("result", lambda s: (s == "L").sum()),
        goals_for=("gf", "sum"),
        goals_against=("ga", "sum"),
        avg_gf=("gf", "mean"),
        avg_ga=("ga", "mean"),
        first_year=("year", "min"),
        last_year=("year", "max"),
    ).rename_axis("team").reset_index()
    g["win_pct"] = (g["wins"] / g["matches"] * 100).round(1)
    g["goal_diff"] = g["goals_for"] - g["goals_against"]
    return g


def recent_form(hist: pd.DataFrame, team: str, n: int = 10) -> pd.DataFrame:
    """Last n matches for a team (most recent first), including matches
    played under any predecessor name.
    """
    df = team_history(hist, team).copy()
    if df.empty:
        return df
    cols = ["year", "team", "opponent", "gf", "ga", "result", "stage"]
    return (
        df.sort_values("year", ascending=False)
        .head(n)[cols]
        .rename(columns={"team": "played as"})
    )


def head_to_head(hist: pd.DataFrame, team_a: str, team_b: str) -> dict:
    """Return summary of every World Cup meeting between two teams,
    including matches played under predecessor names.
    """
    a_names = hist_names(team_a)
    b_names = hist_names(team_b)
    df = hist[hist["team"].isin(a_names) & hist["opponent"].isin(b_names)]
    if df.empty:
        return {"played": 0}
    return {
        "played": len(df),
        "wins_a": int((df["result"] == "W").sum()),
        "draws": int((df["result"] == "D").sum()),
        "wins_b": int((df["result"] == "L").sum()),
        "avg_gf_a": float(df["gf"].mean()),
        "avg_gf_b": float(df["ga"].mean()),
        "matches": df.sort_values("year", ascending=False),
    }


# ---------------------------------------------------------------------------
# Prediction engine — Poisson model with head-to-head adjustment
# ---------------------------------------------------------------------------
# Tournament-wide averages used as a baseline (and as the fallback for
# debutant teams). Computed once from the historical dataset.
def baseline_goals(hist: pd.DataFrame) -> tuple[float, float]:
    return float(hist["gf"].mean()), float(hist["ga"].mean())


def _team_attack_defense(
    hist: pd.DataFrame,
    qual: pd.DataFrame | None,
    team: str,
    base_gf: float,
    base_ga: float,
    qualifier_weight: float,
    debutant_strength: float,
) -> tuple[float, float, dict]:
    """Compute (attack, defense, flags) for a single team.

    Combines World-Cup history with 2026 qualifying form using a
    weighted blend. Falls back to baseline for true debutants.
    """
    flags = {"is_debutant": False, "used_qualifiers": False}

    # ---- World Cup history component ----
    h = team_history(hist, team)
    if h.empty:
        wc_attack = base_gf * debutant_strength
        wc_defense = base_ga / debutant_strength
        flags["is_debutant"] = True
    else:
        wc_attack = float(h["gf"].mean())
        wc_defense = float(h["ga"].mean())

    # ---- 2026 qualifying component ----
    if qual is not None and not qual.empty and qualifier_weight > 0:
        prof = qualifier_profile(qual, team)
    else:
        prof = None

    if prof is None:
        return wc_attack, wc_defense, flags

    flags["used_qualifiers"] = True
    flags["qualifier_matches"] = prof["matches"]

    # Effective weight scales with the size of the qualifying sample.
    # A team with 1 qualifier match shouldn't outweigh decades of WC data.
    # k=4: with 4 qualifiers, full `qualifier_weight` applies; less below.
    k = 4
    sample_factor = prof["matches"] / (prof["matches"] + k)
    w = qualifier_weight * sample_factor

    # Debutants — the qualifying sample is the *only* real signal we have,
    # so override the baseline fallback entirely with their qualifier form
    # (still using sample_factor so a single match doesn't fully dominate).
    if flags["is_debutant"]:
        # Blend qualifying form with baseline by sample size
        attack = (
            sample_factor * prof["avg_gf"]
            + (1 - sample_factor) * (base_gf * debutant_strength)
        )
        defense = (
            sample_factor * prof["avg_ga"]
            + (1 - sample_factor) * (base_ga / debutant_strength)
        )
        return attack, defense, flags

    # Established team — weighted average of WC history and qualifying form
    attack = (1 - w) * wc_attack + w * prof["avg_gf"]
    defense = (1 - w) * wc_defense + w * prof["avg_ga"]
    return attack, defense, flags


def expected_goals(
    hist: pd.DataFrame,
    team_a: str,
    team_b: str,
    h2h_weight: float = 0.20,
    debutant_strength: float = 0.70,
    qual: pd.DataFrame | None = None,
    qualifier_weight: float = 0.75,
) -> tuple[float, float, dict]:
    """Return (lambda_a, lambda_b, info) — expected goals for each team.

    Method:
      1. Start from each team's historical attack (avg goals scored)
         and defense (avg goals conceded) at the World Cup.
      2. Blend in their 2026 qualifying form, weighted by sample size
         (more qualifying matches = more influence; capped by
         `qualifier_weight`).
      3. Combine teams: team A's xG = mean(A.attack, B.defense).
      4. If the two teams have met at past World Cups, blend the
         head-to-head goal averages in (weight = h2h_weight).
      5. For debutants with no history *and* no qualifier data, fall
         back to the tournament-wide baseline scaled by
         `debutant_strength`.
    """
    base_gf, base_ga = baseline_goals(hist)
    info: dict = {"debutant_a": False, "debutant_b": False,
                  "used_qualifiers_a": False, "used_qualifiers_b": False}

    a_attack, a_defense, fa = _team_attack_defense(
        hist, qual, team_a, base_gf, base_ga, qualifier_weight, debutant_strength,
    )
    b_attack, b_defense, fb = _team_attack_defense(
        hist, qual, team_b, base_gf, base_ga, qualifier_weight, debutant_strength,
    )
    info["debutant_a"] = fa["is_debutant"]
    info["debutant_b"] = fb["is_debutant"]
    info["used_qualifiers_a"] = fa["used_qualifiers"]
    info["used_qualifiers_b"] = fb["used_qualifiers"]
    info["qualifier_matches_a"] = fa.get("qualifier_matches", 0)
    info["qualifier_matches_b"] = fb.get("qualifier_matches", 0)

    # Blend attack vs opponent defense (each scaled by tournament average)
    lam_a = (a_attack + b_defense) / 2
    lam_b = (b_attack + a_defense) / 2

    # Head-to-head adjustment
    h2h = head_to_head(hist, team_a, team_b)
    info["h2h"] = h2h
    if h2h["played"] >= 1:
        lam_a = (1 - h2h_weight) * lam_a + h2h_weight * h2h["avg_gf_a"]
        lam_b = (1 - h2h_weight) * lam_b + h2h_weight * h2h["avg_gf_b"]

    # Floor at a small positive value so Poisson works
    lam_a = max(lam_a, 0.2)
    lam_b = max(lam_b, 0.2)
    return lam_a, lam_b, info


def match_probabilities(
    lam_a: float, lam_b: float, max_goals: int = 8
) -> tuple[float, float, float, np.ndarray]:
    """Return (P(A wins), P(draw), P(B wins), score-grid)."""
    a = np.array([poisson.pmf(i, lam_a) for i in range(max_goals + 1)])
    b = np.array([poisson.pmf(i, lam_b) for i in range(max_goals + 1)])
    grid = np.outer(a, b)  # rows = team A goals, cols = team B goals
    p_a = float(np.tril(grid, -1).sum())  # team A scores more
    p_draw = float(np.trace(grid))
    p_b = float(np.triu(grid, 1).sum())
    return p_a, p_draw, p_b, grid


def likely_scoreline(grid: np.ndarray) -> tuple[int, int, float]:
    i, j = np.unravel_index(np.argmax(grid), grid.shape)
    return int(i), int(j), float(grid[i, j])


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def prob_bar(p_a: float, p_draw: float, p_b: float, team_a: str, team_b: str):
    """Render a horizontal stacked probability bar.

    Colours come from CSS variables defined at the top of the app, so the
    bar stays high-contrast in both light and dark Streamlit themes.
    """
    a_pct = p_a * 100
    d_pct = p_draw * 100
    b_pct = p_b * 100
    html = f"""
    <div style="display:flex;width:100%;border-radius:8px;overflow:hidden;
                font-weight:600;color:#fff;height:32px;
                box-shadow:0 1px 2px rgba(0,0,0,0.08);">
      <div style="width:{a_pct}%;background:var(--wc-team-a);display:flex;
                  align-items:center;justify-content:center;font-size:13px;
                  white-space:nowrap;overflow:hidden;">
        {team_a} {a_pct:.1f}%
      </div>
      <div style="width:{d_pct}%;background:var(--wc-draw);display:flex;
                  align-items:center;justify-content:center;font-size:13px;
                  white-space:nowrap;overflow:hidden;">
        Draw {d_pct:.1f}%
      </div>
      <div style="width:{b_pct}%;background:var(--wc-team-b);display:flex;
                  align-items:center;justify-content:center;font-size:13px;
                  white-space:nowrap;overflow:hidden;">
        {team_b} {b_pct:.1f}%
      </div>
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


def predict_card(
    hist: pd.DataFrame,
    team_a: str,
    team_b: str,
    qual: pd.DataFrame | None = None,
    qualifier_weight: float = 0.75,
):
    """Render a full prediction card for one match."""
    lam_a, lam_b, info = expected_goals(
        hist, team_a, team_b, qual=qual, qualifier_weight=qualifier_weight,
    )
    p_a, p_d, p_b, grid = match_probabilities(lam_a, lam_b)
    sa, sb, p_score = likely_scoreline(grid)

    prob_bar(p_a, p_d, p_b, team_a, team_b)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{team_a} xG", f"{lam_a:.2f}")
    c2.metric(f"{team_b} xG", f"{lam_b:.2f}")
    c3.metric("Most likely score", f"{sa} – {sb}")
    c4.metric("Score probability", f"{p_score*100:.1f}%")

    flags = []
    if info.get("debutant_a"):
        if info.get("used_qualifiers_a"):
            flags.append(
                f"🆕 {team_a} is a World Cup debutant — projection based on "
                f"{info['qualifier_matches_a']} qualifying match(es)."
            )
        else:
            flags.append(
                f"⚠️ {team_a} is a World Cup debutant — using baseline projection."
            )
    elif info.get("used_qualifiers_a"):
        flags.append(
            f"📈 {team_a}: {info['qualifier_matches_a']} qualifying match(es) "
            "blended into projection."
        )
    if info.get("debutant_b"):
        if info.get("used_qualifiers_b"):
            flags.append(
                f"🆕 {team_b} is a World Cup debutant — projection based on "
                f"{info['qualifier_matches_b']} qualifying match(es)."
            )
        else:
            flags.append(
                f"⚠️ {team_b} is a World Cup debutant — using baseline projection."
            )
    elif info.get("used_qualifiers_b"):
        flags.append(
            f"📈 {team_b}: {info['qualifier_matches_b']} qualifying match(es) "
            "blended into projection."
        )
    if info["h2h"]["played"] == 0:
        flags.append("ℹ️ No prior World Cup meetings between these teams.")
    else:
        h = info["h2h"]
        flags.append(
            f"📊 Head-to-head: {h['played']} prior meetings — "
            f"{team_a} {h['wins_a']}W / {h['draws']}D / {h['wins_b']}L "
            f"(avg goals {h['avg_gf_a']:.2f}-{h['avg_gf_b']:.2f})."
        )
    for f in flags:
        st.caption(f)


# ===========================================================================
# Main app
# ===========================================================================
def main() -> None:
    if not HIST_CSV.exists() or not DRAW_XLSX.exists():
        st.error(
            "Missing data files. Place `world-cup-Data-all-matches-2.csv` "
            "and `WCup_2026_4.2.5_en.xlsx` next to app.py."
        )
        st.stop()

    base_hist = load_history()
    groups, fixtures = load_draw()
    qualifiers = load_qualifiers()

    # --- Sidebar nav ---
    st.sidebar.title("⚽ World Cup 2026")
    st.sidebar.caption("Canada · Mexico · USA")
    page = st.sidebar.radio(
        "Navigate",
        ["🏟️ Dashboard", "🔮 Match Predictor", "👥 Team Profiles",
         "🥇 Tournament Simulator", "🏆 Knockout Simulator",
         "📈 Team Stats",
         "📡 Live Results", "📐 Methodology"],
    )

    st.sidebar.divider()
    st.sidebar.markdown("### ⚙️ Model settings")
    use_qualifiers = st.sidebar.toggle(
        "Blend in 2026 qualifying form",
        value=True,
        help="When on, each team's 2026 qualifying-campaign results are "
             "blended with their World Cup history. Especially helpful for "
             "debutants who have no World Cup data.",
    )
    qualifier_weight = st.sidebar.slider(
        "Qualifier weight",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Maximum influence qualifying form can have (the actual weight "
             "also scales with sample size). Default 0.75 was calibrated "
             "against the first 28 played 2026 matches — see Methodology.",
        disabled=not use_qualifiers,
    )
    st.sidebar.caption(
        "💡 Defaults calibrated against the 56 played 2026 matches "
        "(group stage complete). Calibrated config nails 62.5% of "
        "outcomes vs 53.6% for the original defaults. Flip on "
        "‘Use 2026 results in the model’ below for another small boost."
    )

    st.sidebar.markdown("### 📡 2026 results feed")
    uploaded = st.sidebar.file_uploader(
        "Upload results CSV",
        type=["csv"],
        help=("Columns: match_no, home_team, away_team, home_score, away_score. "
              "Re-upload any time to refresh predictions."),
    )
    results = load_results_2026(uploaded)
    use_live = st.sidebar.toggle(
        "Use 2026 results in the model",
        value=True,
        help="When on, completed 2026 matches feed back into team form for "
             "later-round predictions.",
    )

    # Merge live results in if requested
    hist = merge_results_into_history(base_hist, results) if use_live else base_hist
    stats = team_stats(hist)

    # Stash settings in a single object passed down to predict_card etc.
    qual_in_use = qualifiers if use_qualifiers else None
    qual_w = qualifier_weight if use_qualifiers else 0.0

    st.sidebar.divider()
    st.sidebar.metric("Historical matches", f"{len(base_hist)//2:,}")
    st.sidebar.metric("Teams in history", f"{base_hist['team'].nunique()}")
    st.sidebar.metric("Qualifier matches", f"{len(qualifiers)//2 if not qualifiers.empty else 0}")
    st.sidebar.metric("2026 results loaded", f"{len(results)}")
    st.sidebar.metric("2026 fixtures", f"{len(fixtures)}")

    # --- Pages ---
    if page == "🏟️ Dashboard":
        page_dashboard(hist, groups, fixtures, results, qual_in_use, qual_w)
    elif page == "🔮 Match Predictor":
        page_predictor(hist, groups, qual_in_use, qual_w)
    elif page == "👥 Team Profiles":
        page_profiles(hist, stats, groups, qualifiers)
    elif page == "🥇 Tournament Simulator":
        page_simulator(hist, fixtures, groups, results, qual_in_use, qual_w)
    elif page == "🏆 Knockout Simulator":
        page_knockout_simulator(results)
    elif page == "📈 Team Stats":
        page_team_stats()
    elif page == "📡 Live Results":
        page_live_results(base_hist, fixtures, results, qual_in_use, qual_w)
    else:
        page_methodology(hist, qualifiers)


# ---------------------------------------------------------------------------
# Page: Dashboard — fixture list with predictions
# ---------------------------------------------------------------------------
def page_dashboard(
    hist: pd.DataFrame,
    groups: pd.DataFrame,
    fixtures: pd.DataFrame,
    results: pd.DataFrame,
    qual: pd.DataFrame | None,
    qualifier_weight: float,
):
    st.title("🏟️ World Cup 2026 — Fixture Dashboard")
    st.caption("Browse every scheduled match. Predictions appear automatically "
               "for fixtures with both teams confirmed. Played matches show the "
               "actual scoreline alongside the prediction.")

    # Build a lookup: match_no -> (home_score, away_score)
    results_by_no = {
        int(r["match_no"]): (int(r["home_score"]), int(r["away_score"]))
        for _, r in results.iterrows()
        if pd.notna(r["match_no"])
    }

    # Filters
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        stage_options = ["All"] + sorted(
            fixtures["stage_label"].dropna().unique().tolist()
        )
        stage = st.selectbox("Stage", stage_options)
    with c2:
        group_options = ["All"] + sorted(groups["group"].unique().tolist())
        group_filter = st.selectbox("Group (group stage only)", group_options)
    with c3:
        search = st.text_input("Search team", "")

    df = fixtures.copy()
    if stage != "All":
        df = df[df["stage_label"] == stage]
    if group_filter != "All":
        teams_in_group = set(groups[groups["group"] == group_filter]["team"])
        df = df[
            df["team1"].isin(teams_in_group) | df["team2"].isin(teams_in_group)
        ]
    if search:
        s = search.lower()
        df = df[
            df["team1"].fillna("").str.lower().str.contains(s)
            | df["team2"].fillna("").str.lower().str.contains(s)
        ]

    st.markdown(f"**{len(df)} matches**")

    for _, m in df.iterrows():
        with st.container(border=True):
            top = st.columns([1, 4, 2])
            top[0].markdown(f"**Match {m['match_no']}**  \n{m['stage_label']}")
            if m["team1"] and m["team2"]:
                top[1].markdown(f"### {m['team1']} 🆚 {m['team2']}")
            else:
                top[1].markdown(f"### {m['slot1']} 🆚 {m['slot2']}")
                top[1].caption("Knockout placeholder — opponents TBD")
            try:
                top[2].markdown(
                    f"📅 {pd.to_datetime(m['date_local']).strftime('%a %d %b %Y · %H:%M')}"
                    f"  \n📍 {m['venue']}"
                )
            except Exception:
                top[2].markdown(f"📍 {m['venue']}")

            if m["team1"] and m["team2"]:
                actual = results_by_no.get(int(m["match_no"]))
                if actual is not None:
                    a_h, a_a = actual
                    st.success(
                        f"✅ **Played:** {m['team1']} {a_h} – {a_a} {m['team2']}"
                    )
                predict_card(hist, m["team1"], m["team2"], qual, qualifier_weight)


# ---------------------------------------------------------------------------
# Page: Match Predictor — pick any two teams
# ---------------------------------------------------------------------------
def page_predictor(
    hist: pd.DataFrame,
    groups: pd.DataFrame,
    qual: pd.DataFrame | None,
    qualifier_weight: float,
):
    st.title("🔮 Match Predictor")
    st.caption("Pick any two 2026 teams to see win/draw/loss probabilities, "
               "expected goals and head-to-head history.")

    teams_2026 = sorted(groups["team"].dropna().unique().tolist())
    c1, c2 = st.columns(2)
    team_a = c1.selectbox("Team A", teams_2026, index=0)
    team_b = c2.selectbox(
        "Team B", teams_2026, index=min(1, len(teams_2026) - 1)
    )

    if team_a == team_b:
        st.warning("Pick two different teams.")
        return

    st.divider()
    predict_card(hist, team_a, team_b, qual, qualifier_weight)

    # --- Score-grid heatmap ---
    lam_a, lam_b, _ = expected_goals(
        hist, team_a, team_b, qual=qual, qualifier_weight=qualifier_weight,
    )
    _, _, _, grid = match_probabilities(lam_a, lam_b, max_goals=6)
    grid_df = pd.DataFrame(
        (grid * 100).round(2),
        index=[f"{i}" for i in range(grid.shape[0])],
        columns=[f"{j}" for j in range(grid.shape[1])],
    )
    grid_df.index.name = team_a
    grid_df.columns.name = team_b

    st.subheader("Score-line probability grid (%)")
    st.caption(
        f"Rows = {team_a} goals, columns = {team_b} goals. "
        "Each cell is the probability of that exact scoreline."
    )
    try:
        # Pretty heatmap when matplotlib is available.
        # `Blues` reads well on both light and dark Streamlit themes
        # (dark cells = high probability either way).
        st.dataframe(
            grid_df.style
                .background_gradient(cmap="Blues")
                .format("{:.2f}"),
            use_container_width=True,
        )
    except ImportError:
        # Fallback: plain dataframe — always works, no extra dependency
        st.dataframe(
            grid_df.round(2),
            use_container_width=True,
        )

    # --- Head-to-head detail ---
    h = head_to_head(hist, team_a, team_b)
    st.subheader("Head-to-head history (World Cup only)")
    if h["played"] == 0:
        st.info("These teams have never met at a FIFA World Cup.")
    else:
        st.write(
            f"**{h['played']}** previous meeting(s) — "
            f"{team_a} **{h['wins_a']}** wins, **{h['draws']}** draws, "
            f"{team_b} **{h['wins_b']}** wins."
        )
        st.dataframe(
            h["matches"][["year", "stage", "gf", "ga"]].rename(
                columns={"gf": f"{team_a} goals", "ga": f"{team_b} goals"}
            ),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Page: Team Profiles
# ---------------------------------------------------------------------------
def page_profiles(
    hist: pd.DataFrame,
    stats: pd.DataFrame,
    groups: pd.DataFrame,
    qualifiers: pd.DataFrame,
):
    st.title("👥 Team Profiles — 2026 Squads")

    teams_2026 = sorted(groups["team"].dropna().unique().tolist())
    team = st.selectbox("Choose a team", teams_2026)

    g_row = groups[groups["team"] == team].iloc[0]
    st.markdown(f"### {team}  ·  Group **{g_row['group']}**")

    # Always render the qualifying-form section (works for debutants too)
    qprof = qualifier_profile(qualifiers, team) if not qualifiers.empty else None

    if is_debutant(team, hist):
        if qprof:
            st.warning(
                f"🆕 **{team} are making their FIFA World Cup debut in 2026.** "
                f"No historical World Cup data, but their **{qprof['matches']} "
                "qualifying matches** (shown below) are used in projections."
            )
        else:
            st.warning(
                f"🆕 **{team} are making their FIFA World Cup debut in 2026.** "
                "There is no historical World Cup data and no qualifying "
                "data found, so all projections fall back to the tournament-wide "
                "baseline (see Methodology page)."
            )
        if qprof:
            _render_qualifier_section(team, qprof)
        return

    s = stats[stats["team"] == team]
    if s.empty:
        st.info("No historical data found.")
        return
    s = s.iloc[0]

    # If the team inherits records from predecessor states, surface that.
    names = hist_names(team)
    predecessors = [n for n in names if n != team]
    if predecessors:
        st.info(
            f"📜 Stats below include this team's record under previous "
            f"name(s): **{', '.join(predecessors)}**."
        )

    c = st.columns(5)
    c[0].metric("Matches played", int(s["matches"]))
    c[1].metric("Win %", f"{s['win_pct']:.1f}%")
    c[2].metric("Avg goals scored", f"{s['avg_gf']:.2f}")
    c[3].metric("Avg goals conceded", f"{s['avg_ga']:.2f}")
    c[4].metric("Goal difference", int(s["goal_diff"]))

    st.subheader("Recent World Cup form (last 10)")
    st.dataframe(recent_form(hist, team, 10), use_container_width=True, hide_index=True)

    st.subheader("Goals timeline")
    timeline = (
        team_history(hist, team)
        .groupby("year")
        .agg(scored=("gf", "sum"), conceded=("ga", "sum"))
        .reset_index()
    )
    st.line_chart(timeline.set_index("year"))

    if qprof:
        _render_qualifier_section(team, qprof)


def _render_qualifier_section(team: str, qprof: dict):
    """Render the 2026-qualifying form block on a team profile."""
    st.subheader("🗒️ 2026 qualifying form")
    n = qprof["matches"]
    c = st.columns(4)
    c[0].metric("Matches in dataset", n)
    c[1].metric("Record",
                f"{qprof['wins']}W / {qprof['draws']}D / {qprof['losses']}L")
    c[2].metric("Avg goals scored", f"{qprof['avg_gf']:.2f}")
    c[3].metric("Avg goals conceded", f"{qprof['avg_ga']:.2f}")
    games = qprof["games"][["team", "opponent", "gf", "ga", "result", "region"]]
    games = games.rename(columns={
        "team": "played as", "gf": "scored", "ga": "conceded",
    })
    st.dataframe(games, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Knockout Simulator — Monte Carlo through the R32 bracket
# ---------------------------------------------------------------------------

# Bracket order from official 2026 bracket image (top to bottom on left side,
# then top to bottom on right side). Each pair plays in the Round of 32.
# Winners of adjacent pairs meet in the R16, and so on.
KNOCKOUT_BRACKET_R32: list[tuple[str, str]] = [
    # Left side
    ("Germany", "Paraguay"),
    ("France", "Sweden"),
    ("South Africa", "Canada"),
    ("Netherlands", "Morocco"),
    ("Portugal", "Croatia"),
    ("Spain", "Austria"),
    ("USA", "Bosnia/Herzeg."),
    ("Belgium", "Senegal"),
    # Right side
    ("Brazil", "Japan"),
    ("Ivory Coast", "Norway"),
    ("Mexico", "Ecuador"),
    ("England", "DR Congo"),
    ("Argentina", "Cape Verde"),
    ("Australia", "Egypt"),
    ("Switzerland", "Algeria"),
    ("Colombia", "Ghana"),
]


def tournament_form_profile(results: pd.DataFrame) -> dict[str, dict]:
    """Build attack/defense per team using ONLY the 2026 tournament so far.

    Returns dict: team -> {games, gf_per, ga_per}
    """
    played = results.dropna(subset=["home_score", "away_score"]).copy()
    if played.empty:
        return {}
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)
    rows = []
    for _, m in played.iterrows():
        rows.append({"team": m["home_team"], "gf": m["home_score"], "ga": m["away_score"]})
        rows.append({"team": m["away_team"], "gf": m["away_score"], "ga": m["home_score"]})
    df = pd.DataFrame(rows)
    profile: dict[str, dict] = {}
    for team, sub in df.groupby("team"):
        profile[team] = {
            "games": int(len(sub)),
            "gf_per": float(sub["gf"].mean()),
            "ga_per": float(sub["ga"].mean()),
        }
    return profile


# ---------------------------------------------------------------------------
# Fox Sports per-process team stats (offensive & defensive)
# Source: foxsports.com/soccer/fifa-world-cup-men/team-stats — 2026 season
# ---------------------------------------------------------------------------
FOX_STATS_PATH = Path(__file__).resolve().parent / "fox_sports_stats.json"

# Fox Sports name → canonical app name
FOX_ALIAS = {
    "Czechia": "Czech Rep.",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Bosnia and Herzegovina": "Bosnia/Herzeg.",
    "United States": "USA",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "South Korea": "Korea Republic",
    "Cape Verde Islands": "Cape Verde",
    "Cabo Verde": "Cape Verde",
}


@st.cache_data(show_spinner=False)
def load_fox_stats() -> list[dict]:
    """Load the Fox Sports per-process team stats JSON.

    Returns a list of dicts: {team, offensive: {...}, defensive: {...}}.
    Returns [] if the file is missing.
    """
    if not FOX_STATS_PATH.exists():
        return []
    try:
        data = json.loads(FOX_STATS_PATH.read_text())
    except Exception:
        return []
    # Canonicalise team names
    for rec in data:
        rec["team"] = FOX_ALIAS.get(rec["team"], rec["team"])
    return data


@st.cache_data(show_spinner=False)
def build_team_strengths(_fox: list[dict] | None = None) -> dict[str, dict[str, float]]:
    """Build per-team attack and defence strength scores (z-scores over the 48 teams).

    Attack uses shots on goal/game, chances created/game, possession %, passing accuracy.
    Defence uses tackles/game, interceptions/game, MINUS fouls/game (less reckless).

    Higher = stronger. A team with no defensive stats gets defence=0 (neutral).
    Returns dict: canonical team -> {"attack": z, "defence": z}.
    """
    fox = _fox if _fox is not None else load_fox_stats()
    if not fox:
        return {}

    rows = []
    for s in fox:
        o = s.get("offensive") or {}
        d = s.get("defensive") or {}
        gp_o = max(int(o.get("GP") or 1), 1)
        gp_d = max(int(d.get("GP") or gp_o), 1) if d else gp_o
        rows.append({
            "team":   s["team"],
            "sog_pg": (o.get("SOG") or 0) / gp_o,
            "cc_pg":  (o.get("CC")  or 0) / gp_o,
            "poss":   float(o.get("POSS") or 50),
            "pa":     float(o.get("PA") or 0.7),
            "tkl_pg": ((d.get("TKL") or 0) / gp_d) if d else None,
            "int_pg": ((d.get("INT") or 0) / gp_d) if d else None,
            "f_pg":   ((d.get("F")   or 0) / gp_d) if d else None,
        })

    def _z(values):
        arr = np.array([v for v in values if v is not None], dtype=float)
        if len(arr) == 0:
            return 0.0, 1.0
        mu = float(arr.mean())
        sd = float(arr.std(ddof=0)) or 1.0
        return mu, sd

    sog_mu, sog_sd = _z([r["sog_pg"] for r in rows])
    cc_mu,  cc_sd  = _z([r["cc_pg"]  for r in rows])
    poss_mu, poss_sd = _z([r["poss"] for r in rows])
    pa_mu,  pa_sd  = _z([r["pa"]  for r in rows])
    tkl_mu, tkl_sd = _z([r["tkl_pg"] for r in rows])
    int_mu, int_sd = _z([r["int_pg"] for r in rows])
    f_mu,   f_sd   = _z([r["f_pg"]   for r in rows])

    def _zs(x, mu, sd):
        return 0.0 if x is None else (x - mu) / sd

    strengths: dict[str, dict[str, float]] = {}
    for r in rows:
        attack = float(np.mean([
            _zs(r["sog_pg"], sog_mu, sog_sd),
            _zs(r["cc_pg"],  cc_mu,  cc_sd),
            _zs(r["poss"],   poss_mu, poss_sd),
            _zs(r["pa"],     pa_mu, pa_sd),
        ]))
        if r["tkl_pg"] is None:
            defence = 0.0
        else:
            defence = float(np.mean([
                _zs(r["tkl_pg"], tkl_mu, tkl_sd),
                _zs(r["int_pg"], int_mu, int_sd),
                -_zs(r["f_pg"],   f_mu, f_sd),
            ]))
        strengths[r["team"]] = {"attack": attack, "defence": defence}
    return strengths


def knockout_expected_goals(
    team_a: str,
    team_b: str,
    form: dict[str, dict],
    tournament_avg: float,
    strengths: dict[str, dict[str, float]] | None = None,
    fox_weight: float = 0.0,
) -> tuple[float, float]:
    """Expected goals for a knockout match.

    Base: λ_A = (A's goals-scored per game) × (B's goals-conceded per game) / tournament_avg.
    Knockout suppression: shrink λ by 8% (knockouts are cagier than groups).

    Fox blend (when `strengths` is provided and `fox_weight > 0`): each team's
    attack-z and the opponent's defence-z are folded into λ via a multiplier
        m_A = (1 + w·atk_A) · (1 − w·def_B)
    where w = fox_weight. Defence-z is *positive when good* so we subtract it
    in the opponent's λ. Multiplier is clipped to [0.55, 1.65] so a single
    Fox stat can never more than ~double or halve a team's xG.
    """
    KNOCKOUT_SUPPRESSION = 0.92
    a = form.get(team_a, {"gf_per": tournament_avg, "ga_per": tournament_avg})
    b = form.get(team_b, {"gf_per": tournament_avg, "ga_per": tournament_avg})
    lam_a = max(0.15, a["gf_per"] * b["ga_per"] / max(tournament_avg, 0.1))
    lam_b = max(0.15, b["gf_per"] * a["ga_per"] / max(tournament_avg, 0.1))

    if strengths and fox_weight > 0:
        sa = strengths.get(team_a, {"attack": 0.0, "defence": 0.0})
        sb = strengths.get(team_b, {"attack": 0.0, "defence": 0.0})
        w = float(fox_weight)
        m_a = max(0.55, min(1.65, (1 + w * sa["attack"]) * (1 - w * sb["defence"])))
        m_b = max(0.55, min(1.65, (1 + w * sb["attack"]) * (1 - w * sa["defence"])))
        lam_a *= m_a
        lam_b *= m_b

    return lam_a * KNOCKOUT_SUPPRESSION, lam_b * KNOCKOUT_SUPPRESSION


def knockout_win_prob(
    team_a: str,
    team_b: str,
    form: dict[str, dict],
    tournament_avg: float,
    strengths: dict[str, dict[str, float]] | None = None,
    fox_weight: float = 0.0,
) -> tuple[float, float, float, float]:
    """P(A wins after ET/pens), P(B wins), expected goals λ_a, λ_b.

    Knockouts have no draws — we redistribute the draw probability
    proportionally to each team's regulation win probability (a
    standard approximation for ET + penalty shootouts).
    """
    lam_a, lam_b = knockout_expected_goals(
        team_a, team_b, form, tournament_avg,
        strengths=strengths, fox_weight=fox_weight,
    )
    p_a, p_d, p_b, _ = match_probabilities(lam_a, lam_b)
    if p_a + p_b < 1e-9:
        return 0.5, 0.5, lam_a, lam_b
    # Allocate the draw probability proportionally to each team's regulation win share
    share_a = p_a / (p_a + p_b)
    return p_a + p_d * share_a, p_b + p_d * (1 - share_a), lam_a, lam_b


def simulate_knockouts(
    bracket: list[tuple[str, str]],
    form: dict[str, dict],
    tournament_avg: float,
    n_sims: int = 20000,
    rng_seed: int = 42,
    strengths: dict[str, dict[str, float]] | None = None,
    fox_weight: float = 0.0,
) -> dict:
    """Monte Carlo through R32 → R16 → QF → SF → Final.

    Returns counts at each round + champion probabilities.
    """
    rng = np.random.default_rng(rng_seed)
    teams = [t for pair in bracket for t in pair]
    # Pre-compute pairwise win probabilities (cheap — 32 teams, sparse usage)
    win_cache: dict[tuple[str, str], float] = {}

    def p_win(a: str, b: str) -> float:
        key = (a, b)
        if key in win_cache:
            return win_cache[key]
        pa, _, _, _ = knockout_win_prob(
            a, b, form, tournament_avg,
            strengths=strengths, fox_weight=fox_weight,
        )
        win_cache[key] = pa
        win_cache[(b, a)] = 1 - pa
        return pa

    counts = {
        "R16":      {t: 0 for t in teams},
        "QF":       {t: 0 for t in teams},
        "SF":       {t: 0 for t in teams},
        "Final":    {t: 0 for t in teams},
        "Champion": {t: 0 for t in teams},
    }

    for _ in range(n_sims):
        # Round of 32
        r16_winners = []
        for a, b in bracket:
            w = a if rng.random() < p_win(a, b) else b
            counts["R16"][w] += 1
            r16_winners.append(w)
        # Round of 16
        qf_winners = []
        for i in range(0, 16, 2):
            a, b = r16_winners[i], r16_winners[i + 1]
            w = a if rng.random() < p_win(a, b) else b
            counts["QF"][w] += 1
            qf_winners.append(w)
        # Quarter-finals
        sf_winners = []
        for i in range(0, 8, 2):
            a, b = qf_winners[i], qf_winners[i + 1]
            w = a if rng.random() < p_win(a, b) else b
            counts["SF"][w] += 1
            sf_winners.append(w)
        # Semi-finals
        final_winners = []
        for i in range(0, 4, 2):
            a, b = sf_winners[i], sf_winners[i + 1]
            w = a if rng.random() < p_win(a, b) else b
            counts["Final"][w] += 1
            final_winners.append(w)
        # Final
        a, b = final_winners
        w = a if rng.random() < p_win(a, b) else b
        counts["Champion"][w] += 1

    # Convert to probabilities
    probs = {round_name: {t: c / n_sims for t, c in cs.items()}
             for round_name, cs in counts.items()}
    return probs


def _resolve_match(team_a, team_b, form, tournament_avg,
                   strengths=None, fox_weight=0.0):
    """Single-shot match resolution for the deterministic bracket walk.
    Returns (winner, score_a, score_b, et_flag).
    """
    pa, pb, la, lb = knockout_win_prob(
        team_a, team_b, form, tournament_avg,
        strengths=strengths, fox_weight=fox_weight,
    )
    # Modal scoreline
    a = np.array([poisson.pmf(i, la * 1.0) for i in range(8)])
    b = np.array([poisson.pmf(i, lb * 1.0) for i in range(8)])
    grid = np.outer(a, b)
    sa, sb = np.unravel_index(np.argmax(grid), grid.shape)
    sa, sb = int(sa), int(sb)
    # Tiebreaker on probability ties: better GD/game
    if abs(pa - pb) < 1e-6:
        ga = form.get(team_a, {"gf_per":0,"ga_per":0})
        gb = form.get(team_b, {"gf_per":0,"ga_per":0})
        a_fav = (ga["gf_per"] - ga["ga_per"]) >= (gb["gf_per"] - gb["ga_per"])
    else:
        a_fav = pa > pb
    et = False
    if sa == sb:
        et = True
        if a_fav: sa += 1
        else:     sb += 1
    elif (sa > sb) != a_fav:
        sa, sb = sb, sa
    winner = team_a if a_fav else team_b
    return winner, sa, sb, et


def walk_bracket(bracket, form, tournament_avg,
                 strengths=None, fox_weight=0.0):
    """Resolve the entire knockout tree deterministically (favourite advances)."""
    def resolve(a, b):
        return _resolve_match(a, b, form, tournament_avg,
                              strengths=strengths, fox_weight=fox_weight)
    r32 = [resolve(a, b) for a, b in bracket]
    r16p = [(r32[i][0], r32[i+1][0]) for i in range(0, 16, 2)]
    r16 = [resolve(a, b) for a, b in r16p]
    qfp  = [(r16[i][0], r16[i+1][0]) for i in range(0, 8, 2)]
    qf  = [resolve(a, b) for a, b in qfp]
    sfp  = [(qf[i][0], qf[i+1][0]) for i in range(0, 4, 2)]
    sf  = [resolve(a, b) for a, b in sfp]
    finalp = (sf[0][0], sf[1][0])
    final = resolve(*finalp)
    return {"R32":(bracket, r32), "R16":(r16p, r16), "QF":(qfp, qf),
            "SF":(sfp, sf), "FINAL":(finalp, final)}


def render_filled_bracket(bracket_walk):
    """Build the filled-in bracket image and return the matplotlib Figure."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(26, 12))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, 30); ax.set_ylim(0, 18); ax.axis("off")
    BOX_BG, BOX_BG_WIN = "#1f6b6e", "#2dd4bf"
    BOX_BG_CHAMP = "#fbbf24"
    TEXT_LIGHT, TEXT_DARK, LINE = "#0d1117", "#f0f9ff", "#2dd4bf"
    W, H = 2.6, 0.55

    def box(x, y, ta, tb, sa, sb, w, et=False):
        for k, (t, s, win) in enumerate([(ta, sa, w==ta), (tb, sb, w==tb)]):
            bg = BOX_BG_WIN if win else BOX_BG
            fg = TEXT_LIGHT if win else TEXT_DARK
            r = FancyBboxPatch((x, y - k*H - H/2), W, H*0.9,
                               boxstyle="round,pad=0.02,rounding_size=0.08",
                               linewidth=0, facecolor=bg, zorder=2)
            ax.add_patch(r)
            ax.text(x + 0.12, y - k*H - H/2 + H*0.45, t,
                    fontsize=8.5, color=fg, weight="bold", ha="left", va="center", zorder=3)
            ax.text(x + W - 0.15, y - k*H - H/2 + H*0.45, str(s),
                    fontsize=10, color=fg, weight="bold", ha="right", va="center", zorder=3)
        if et:
            ax.text(x + W/2, y - H - H/2 - 0.15, "AET",
                    fontsize=6, color="#fcd34d", ha="center", va="top", zorder=3)

    def conn(x1, y1, x2, y2):
        mid = (x1+x2)/2
        ax.plot([x1, mid, mid, x2], [y1, y1, y2, y2],
                color=LINE, linewidth=1.2, zorder=1, alpha=0.7)

    LEFT_X  = [0.3, 3.4, 6.5, 9.6]
    RIGHT_X = [27.1, 24.0, 20.9, 17.8]

    def ycen(n, total=15.0, top=17.0):
        if n == 1: return [top - total/2]
        step = total/(n-1)
        return [top - i*step for i in range(n)]
    left_r32 = ycen(8); right_r32 = ycen(8)
    def nxt(ys): return [(ys[i]+ys[i+1])/2 for i in range(0,len(ys),2)]
    left_r16, left_qf, left_sf = nxt(left_r32), None, None
    left_qf = nxt(left_r16); left_sf = nxt(left_qf)
    right_r16 = nxt(right_r32); right_qf = nxt(right_r16); right_sf = nxt(right_qf)
    final_y = (left_sf[0] + right_sf[0])/2
    final_x = (LEFT_X[3] + W + RIGHT_X[3])/2 - W/2

    r32_pairs, r32_res = bracket_walk["R32"]
    r16_pairs, r16_res = bracket_walk["R16"]
    qf_pairs,  qf_res  = bracket_walk["QF"]
    sf_pairs,  sf_res  = bracket_walk["SF"]
    fpair, fres = bracket_walk["FINAL"]

    for k,((a,b),(w,sa,sb,et)) in enumerate(zip(r32_pairs[:8], r32_res[:8])):
        box(LEFT_X[0], left_r32[k], a, b, sa, sb, w, et)
    for k,((a,b),(w,sa,sb,et)) in enumerate(zip(r32_pairs[8:], r32_res[8:])):
        box(RIGHT_X[0], right_r32[k], a, b, sa, sb, w, et)
    for k,((a,b),(w,sa,sb,et)) in enumerate(zip(r16_pairs[:4], r16_res[:4])):
        box(LEFT_X[1], left_r16[k], a, b, sa, sb, w, et)
    for k,((a,b),(w,sa,sb,et)) in enumerate(zip(qf_pairs[:2], qf_res[:2])):
        box(LEFT_X[2], left_qf[k], a, b, sa, sb, w, et)
    (a,b),(w,sa,sb,et) = sf_pairs[0], sf_res[0]
    box(LEFT_X[3], left_sf[0], a, b, sa, sb, w, et)
    for k,((a,b),(w,sa,sb,et)) in enumerate(zip(r16_pairs[4:], r16_res[4:])):
        box(RIGHT_X[1], right_r16[k], a, b, sa, sb, w, et)
    for k,((a,b),(w,sa,sb,et)) in enumerate(zip(qf_pairs[2:], qf_res[2:])):
        box(RIGHT_X[2], right_qf[k], a, b, sa, sb, w, et)
    (a,b),(w,sa,sb,et) = sf_pairs[1], sf_res[1]
    box(RIGHT_X[3], right_sf[0], a, b, sa, sb, w, et)
    (a,b),(w,sa,sb,et) = fpair, fres
    box(final_x, final_y, a, b, sa, sb, w, et)

    def cpair(xf, yf, xt, yt):
        for i in range(0,len(yf),2):
            x1 = xf + W; y1 = (yf[i] - H/2 + H*0.45 + yf[i] - H - H/2 + H*0.45)/2
            y2 = (yt[i//2] - H/2 + H*0.45 + yt[i//2] - H - H/2 + H*0.45)/2
            conn(x1, y1, xt, y2)
    def cpair_r(xf, yf, xt, yt):
        for i in range(0,len(yf),2):
            x1 = xf; y1 = (yf[i] - H/2 + H*0.45 + yf[i] - H - H/2 + H*0.45)/2
            y2 = (yt[i//2] - H/2 + H*0.45 + yt[i//2] - H - H/2 + H*0.45)/2
            conn(x1, y1, xt + W, y2)
    cpair(LEFT_X[0], left_r32, LEFT_X[1], left_r16)
    cpair(LEFT_X[1], left_r16, LEFT_X[2], left_qf)
    cpair(LEFT_X[2], left_qf, LEFT_X[3], left_sf)
    cpair_r(RIGHT_X[0], right_r32, RIGHT_X[1], right_r16)
    cpair_r(RIGHT_X[1], right_r16, RIGHT_X[2], right_qf)
    cpair_r(RIGHT_X[2], right_qf, RIGHT_X[3], right_sf)
    # SF -> Final
    conn(LEFT_X[3]+W, left_sf[0]-H*0.55, final_x, final_y-H*0.55)
    conn(RIGHT_X[3], right_sf[0]-H*0.55, final_x+W, final_y-H*0.55)

    labels = ["R32","R16","QF","SF","FINAL","SF","QF","R16","R32"]
    xs = [LEFT_X[0]+W/2, LEFT_X[1]+W/2, LEFT_X[2]+W/2, LEFT_X[3]+W/2,
          final_x+W/2,
          RIGHT_X[3]+W/2, RIGHT_X[2]+W/2, RIGHT_X[1]+W/2, RIGHT_X[0]+W/2]
    for lab, x in zip(labels, xs):
        ax.text(x, 17.7, lab, fontsize=11, color="#94a3b8", weight="bold", ha="center")

    champion = fres[0]
    MIDX = 15.0
    ax.text(MIDX, 17.45, "World Cup 2026 — Predicted Knockout Bracket",
            fontsize=15, color="#e2e8f0", weight="bold", ha="center")
    ax.text(MIDX, 17.05, "Based on 2026 tournament form only (group stage data)",
            fontsize=10, color="#94a3b8", ha="center", style="italic")
    ax.text(MIDX, 1.0, f"CHAMPION: {champion.upper()}",
            fontsize=26, color=BOX_BG_CHAMP, weight="bold", ha="center")
    ax.text(MIDX, 0.35, f"Final: {fpair[0]} {fres[1]}–{fres[2]} {fpair[1]}" +
            (" (AET)" if fres[3] else ""),
            fontsize=13, color="#fef3c7", ha="center")
    plt.tight_layout(pad=0)
    return fig


def page_knockout_simulator(results: pd.DataFrame):
    st.title("🏆 Knockout Simulator")
    st.markdown(
        "Monte-Carlo simulation of the Round of 32 bracket. Base signal is "
        "**current 2026 tournament form** (goals scored / conceded per game). "
        "Knockout suppression factor 0.92 applied; draws redistributed to "
        "extra-time / penalty winners proportional to regulation win share. "
        "Optionally blends in **Fox Sports per-process stats** (shots on goal, "
        "chances created, possession, passing accuracy, tackles, interceptions, "
        "fouls) to sharpen the favourite signal."
    )

    form = tournament_form_profile(results)
    bracket_teams = [t for pair in KNOCKOUT_BRACKET_R32 for t in pair]
    missing = [t for t in bracket_teams if t not in form]
    if missing:
        st.warning(
            f"No 2026 form data for: {', '.join(missing)}. They'll be modelled "
            "at the tournament average."
        )
    if not form:
        st.error("No played 2026 matches found — upload results first.")
        return

    # Tournament average (goals per team per game)
    games = sum(p["games"] for p in form.values())
    total_gf = sum(p["gf_per"] * p["games"] for p in form.values())
    tournament_avg = total_gf / max(games, 1)

    # Fox Sports strengths (cached). If unavailable the toggle is a no-op.
    strengths = build_team_strengths()
    has_fox = bool(strengths)

    c1, c2, c3 = st.columns(3)
    c1.metric("Tournament matches used", f"{games // 2}")
    c2.metric("Avg goals per team", f"{tournament_avg:.2f}")
    n_sims = c3.slider("Simulations", 1000, 50000, 20000, 1000)

    # Fox Sports blend control
    fcol1, fcol2 = st.columns([1, 2])
    use_fox = fcol1.toggle(
        "Blend in Fox Sports stats",
        value=has_fox,
        help="Fold per-process offensive & defensive z-scores into expected "
             "goals. Sharpens favourite vs underdog gaps when group-stage "
             "form is noisy (only 3 games per team).",
        disabled=not has_fox,
    )
    fox_weight = fcol2.slider(
        "Fox-stat weight",
        min_value=0.0,
        max_value=0.30,
        value=0.18 if has_fox else 0.0,
        step=0.02,
        help="How strongly per-process stats tilt λ. 0 = pure form; 0.18 = "
             "default; 0.30 = aggressive.",
        disabled=not (has_fox and use_fox),
    )
    fw = fox_weight if (use_fox and has_fox) else 0.0
    str_arg = strengths if (use_fox and has_fox) else None

    with st.spinner(f"Running {n_sims:,} simulations..."):
        probs = simulate_knockouts(
            KNOCKOUT_BRACKET_R32, form, tournament_avg, n_sims=n_sims,
            strengths=str_arg, fox_weight=fw,
        )

    # --- Filled-in bracket image (deterministic walk: favourite advances) ---
    st.markdown("### 📖 Predicted bracket path")
    cap = (
        "Below is the most-likely walk through the bracket: the favourite "
        "advances in every round, with each scoreline taken from the modal "
        "(most probable) score under the Poisson model. AET = decided in "
        "extra time / penalties."
    )
    if fw > 0:
        cap += f" Fox-stats blend ON (weight {fw:.2f})."
    st.caption(cap)
    walk = walk_bracket(KNOCKOUT_BRACKET_R32, form, tournament_avg,
                        strengths=str_arg, fox_weight=fw)
    fig = render_filled_bracket(walk)
    if fig is not None:
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning("matplotlib unavailable — install matplotlib>=3.7 to see the bracket.")

    # --- Champion leaderboard ---
    st.markdown("### 🥇 Championship probabilities")
    champ_df = pd.DataFrame([
        {
            "Team": t,
            "Win R32 %":  probs["R16"][t] * 100,
            "Reach QF %": probs["QF"][t] * 100,
            "Reach SF %": probs["SF"][t] * 100,
            "Reach Final %": probs["Final"][t] * 100,
            "Champion %": probs["Champion"][t] * 100,
        }
        for t in bracket_teams
    ]).sort_values("Champion %", ascending=False).reset_index(drop=True)
    st.dataframe(
        champ_df.style.format({
            "Win R32 %": "{:.1f}",
            "Reach QF %": "{:.1f}",
            "Reach SF %": "{:.1f}",
            "Reach Final %": "{:.1f}",
            "Champion %": "{:.1f}",
        }).background_gradient(subset=["Champion %"], cmap="Blues"),
        use_container_width=True,
        hide_index=True,
    )

    # --- Round of 32 individual match predictions ---
    st.markdown("### 🎯 Round of 32 — match-by-match")
    for a, b in KNOCKOUT_BRACKET_R32:
        p_a, p_b, lam_a, lam_b = knockout_win_prob(
            a, b, form, tournament_avg,
            strengths=str_arg, fox_weight=fw,
        )
        c1, c2 = st.columns([3, 1])
        with c1:
            prob_bar(p_a, 0.0, p_b, a, b)
            st.caption(f"Expected goals: {a} {lam_a:.2f} — {lam_b:.2f} {b}")
        with c2:
            fav = a if p_a > p_b else b
            edge = abs(p_a - p_b) * 100
            st.metric("Favourite", fav, f"+{edge:.1f}%")

    # --- Form table ---
    st.markdown("### 📊 Tournament form (the data driving the sim)")
    form_rows = []
    for t in bracket_teams:
        if t in form:
            p = form[t]
            form_rows.append({
                "Team": t,
                "Games": p["games"],
                "Goals For / game": round(p["gf_per"], 2),
                "Goals Against / game": round(p["ga_per"], 2),
                "GD / game": round(p["gf_per"] - p["ga_per"], 2),
            })
    form_df = pd.DataFrame(form_rows).sort_values("GD / game", ascending=False)
    st.dataframe(form_df, use_container_width=True, hide_index=True)

    # --- Fox Sports strengths table ---
    if has_fox:
        st.markdown("### 🧠 Fox Sports team strengths (z-scores)")
        st.caption(
            "Attack z combines shots on goal / chances created / possession / "
            "passing accuracy (per game). Defence z combines tackles + "
            "interceptions minus fouls (per game). Positive = better than "
            "the 48-team average."
        )
        srows = []
        for t in bracket_teams:
            s = strengths.get(t)
            if s:
                srows.append({
                    "Team": t,
                    "Attack z": round(s["attack"], 2),
                    "Defence z": round(s["defence"], 2),
                    "Composite": round(s["attack"] + s["defence"], 2),
                })
        sdf = pd.DataFrame(srows).sort_values("Composite", ascending=False)
        st.dataframe(
            sdf.style.background_gradient(subset=["Attack z"], cmap="Reds")
                     .background_gradient(subset=["Defence z"], cmap="Blues")
                     .background_gradient(subset=["Composite"], cmap="Greens"),
            use_container_width=True, hide_index=True,
        )


# ---------------------------------------------------------------------------
# Page: Team Stats — Fox Sports offensive & defensive
# ---------------------------------------------------------------------------
def page_team_stats():
    st.title("📈 Team Stats — Offensive & Defensive")
    st.caption(
        "Per-process stats from Fox Sports for every nation that played "
        "the 2026 World Cup group stage. Source: "
        "foxsports.com/soccer/fifa-world-cup-men/team-stats"
    )
    fox = load_fox_stats()
    if not fox:
        st.error("`fox_sports_stats.json` not found in the app folder.")
        return

    # --- Offensive table ---
    st.markdown("### ⚽ Offensive")
    st.caption(
        "GP = games played · S = shots · SOG = shots on goal · "
        "CC = chances created · POSS = possession % · "
        "PA = passing accuracy · CK = corner kicks · OFF = offsides."
    )
    off_rows = []
    for s in fox:
        o = s.get("offensive") or {}
        off_rows.append({
            "Team": s["team"],
            "GP": o.get("GP"),
            "S": o.get("S"),
            "SOG": o.get("SOG"),
            "CC": o.get("CC"),
            "POSS %": o.get("POSS"),
            "PA": o.get("PA"),
            "CK": o.get("CK"),
            "OFF": o.get("OFF"),
        })
    odf = pd.DataFrame(off_rows).sort_values("SOG", ascending=False)
    st.dataframe(
        odf.style.background_gradient(subset=["SOG", "CC"], cmap="Reds")
                 .background_gradient(subset=["POSS %", "PA"], cmap="Blues"),
        use_container_width=True, hide_index=True,
    )

    # --- Defensive table ---
    st.markdown("### 🛡️ Defensive")
    st.caption(
        "TI = throw-ins won · INT = interceptions · TKL = tackles · "
        "TA = tackles attempted · GK = goalkeeping actions · "
        "F = fouls · FK = free kicks conceded · OG = own goals."
    )
    def_rows = []
    for s in fox:
        d = s.get("defensive") or {}
        def_rows.append({
            "Team": s["team"],
            "GP": d.get("GP"),
            "INT": d.get("INT"),
            "TKL": d.get("TKL"),
            "TA": d.get("TA"),
            "GK": d.get("GK"),
            "F": d.get("F"),
            "FK": d.get("FK"),
            "OG": d.get("OG"),
        })
    ddf = pd.DataFrame(def_rows)
    # Show teams with no defensive data at the bottom
    ddf["_sort"] = ddf["TKL"].apply(lambda x: -1 if x is None else x)
    ddf = ddf.sort_values("_sort", ascending=False).drop(columns=["_sort"])
    st.dataframe(
        ddf.style.background_gradient(subset=["INT", "TKL"], cmap="Blues"),
        use_container_width=True, hide_index=True,
    )
    nulls = [r["Team"] for r in def_rows if r["TKL"] is None]
    if nulls:
        st.warning(
            f"No defensive stats available from Fox Sports for: "
            f"{', '.join(nulls)} — their defence z is set to 0 (neutral) in the model."
        )

    # --- Composite strengths ---
    strengths = build_team_strengths(fox)
    if strengths:
        st.markdown("### 🧠 Composite strength scores (z-scores)")
        st.caption(
            "How the simulator turns these tables into a single attack and "
            "defence number per team. Positive = better than the 48-team "
            "average; negative = worse."
        )
        rows = [{"Team": t, "Attack z": round(s["attack"], 2),
                 "Defence z": round(s["defence"], 2),
                 "Composite": round(s["attack"] + s["defence"], 2)}
                for t, s in strengths.items()]
        cdf = pd.DataFrame(rows).sort_values("Composite", ascending=False)
        st.dataframe(
            cdf.style.background_gradient(subset=["Attack z"], cmap="Reds")
                     .background_gradient(subset=["Defence z"], cmap="Blues")
                     .background_gradient(subset=["Composite"], cmap="Greens"),
            use_container_width=True, hide_index=True,
        )


# ---------------------------------------------------------------------------
# Page: Tournament Simulator — run the group stage
# ---------------------------------------------------------------------------
def page_simulator(
    hist: pd.DataFrame,
    fixtures: pd.DataFrame,
    groups: pd.DataFrame,
    results: pd.DataFrame,
    qual: pd.DataFrame | None,
    qualifier_weight: float,
):
    st.title("🥇 Group-Stage Simulator")
    st.caption("Plays out every group-stage match. Completed matches use the "
               "actual scoreline; future matches use the model's expected goals.")

    actual_by_no = {
        int(r["match_no"]): (int(r["home_score"]), int(r["away_score"]))
        for _, r in results.iterrows()
        if pd.notna(r["match_no"])
    }

    # Build a quick {slot -> team} mapping from groups
    slot_to_team = {}
    for _, r in groups.iterrows():
        slot_to_team[r["slot"]] = r["team"]

    # Group-stage matches only (slots like A1, B3 — single letter + digit)
    is_group_match = fixtures.apply(
        lambda r: isinstance(r["slot1"], str)
        and isinstance(r["slot2"], str)
        and len(r["slot1"]) == 2
        and r["slot1"][0].isalpha()
        and r["slot1"][1].isdigit(),
        axis=1,
    )
    group_fix = fixtures[is_group_match].copy()

    # Compute predicted (or actual) points per team
    table_rows = []
    for _, m in group_fix.iterrows():
        a, b = m["team1"], m["team2"]
        if not (a and b):
            continue
        actual = actual_by_no.get(int(m["match_no"]))
        if actual is not None:
            ga, gb = actual
            pts_a = 3 if ga > gb else (1 if ga == gb else 0)
            pts_b = 3 if gb > ga else (1 if gb == ga else 0)
            table_rows.append({"team": a, "xPts": pts_a, "xGF": ga, "xGA": gb})
            table_rows.append({"team": b, "xPts": pts_b, "xGF": gb, "xGA": ga})
        else:
            lam_a, lam_b, _ = expected_goals(
                hist, a, b, qual=qual, qualifier_weight=qualifier_weight,
            )
            p_a, p_d, p_b, _ = match_probabilities(lam_a, lam_b)
            # Expected points: P(W)*3 + P(D)*1
            table_rows.append({"team": a, "xPts": p_a * 3 + p_d, "xGF": lam_a, "xGA": lam_b})
            table_rows.append({"team": b, "xPts": p_b * 3 + p_d, "xGF": lam_b, "xGA": lam_a})

    df = pd.DataFrame(table_rows)
    agg = df.groupby("team").agg(
        xPts=("xPts", "sum"),
        xGF=("xGF", "sum"),
        xGA=("xGA", "sum"),
    )
    agg["xGD"] = agg["xGF"] - agg["xGA"]
    agg = agg.merge(
        groups.set_index("team")[["group"]], left_index=True, right_index=True
    )

    for grp in sorted(agg["group"].unique()):
        st.subheader(f"Group {grp}")
        sub = (
            agg[agg["group"] == grp]
            .sort_values(["xPts", "xGD", "xGF"], ascending=False)
            .drop(columns=["group"])
            .round(2)
        )
        sub.insert(0, "Predicted finish", range(1, len(sub) + 1))
        st.dataframe(sub, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Live Results — accuracy tracker + CSV template + download
# ---------------------------------------------------------------------------
def page_live_results(
    base_hist: pd.DataFrame,
    fixtures: pd.DataFrame,
    results: pd.DataFrame,
    qual: pd.DataFrame | None,
    qualifier_weight: float,
):
    st.title("📡 Live 2026 Results")
    st.caption("Track results as they come in and see how the model is doing.")

    # --- Template download ---
    with st.expander("📋 First time? Download the results CSV template", expanded=results.empty):
        template = pd.DataFrame(
            [
                {"match_no": m["match_no"],
                 "home_team": m["team1"],
                 "away_team": m["team2"],
                 "home_score": "",
                 "away_score": ""}
                for _, m in fixtures.iterrows()
                if m["team1"] and m["team2"]
            ]
        )
        st.markdown(
            "Pre-filled with every match where both teams are known. "
            "Fill in `home_score` / `away_score` as games are played, save as CSV, "
            "and upload from the sidebar."
        )
        st.download_button(
            "⬇️ Download template (results_2026_template.csv)",
            template.to_csv(index=False).encode("utf-8"),
            file_name="results_2026_template.csv",
            mime="text/csv",
        )

    if results.empty:
        st.info(
            "No results uploaded yet. Use the **Upload results CSV** control "
            "in the sidebar once games start."
        )
        return

    # --- Accuracy: predicted vs actual ---
    st.subheader("Predicted vs actual")
    rows = []
    for _, r in results.iterrows():
        mn = int(r["match_no"]) if pd.notna(r["match_no"]) else None
        h, a = r["home_team"], r["away_team"]
        hg, ag = int(r["home_score"]), int(r["away_score"])
        # Predict using the *base* history (no leakage — predict pre-tournament)
        lam_a, lam_b, _ = expected_goals(
            base_hist, h, a, qual=qual, qualifier_weight=qualifier_weight,
        )
        p_a, p_d, p_b, grid = match_probabilities(lam_a, lam_b)
        sa, sb, _ = likely_scoreline(grid)
        actual_outcome = "H" if hg > ag else ("A" if hg < ag else "D")
        pred_outcome = "H" if p_a > max(p_d, p_b) else (
            "A" if p_b > max(p_a, p_d) else "D"
        )
        # Probability the model assigned to what actually happened
        p_correct = {"H": p_a, "D": p_d, "A": p_b}[actual_outcome]
        rows.append({
            "Match": f"{h} vs {a}",
            "Predicted": f"{sa}–{sb}",
            "Actual": f"{hg}–{ag}",
            "Outcome correct": "✅" if pred_outcome == actual_outcome else "❌",
            "Exact score": "✅" if (sa, sb) == (hg, ag) else "❌",
            "Model P(actual)": round(p_correct, 3),
        })
    acc_df = pd.DataFrame(rows)
    st.dataframe(acc_df, use_container_width=True, hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    n = len(acc_df)
    correct_outcomes = (acc_df["Outcome correct"] == "✅").sum()
    correct_scores = (acc_df["Exact score"] == "✅").sum()
    avg_p = acc_df["Model P(actual)"].mean()
    c1.metric("Matches played", n)
    c2.metric("Outcome accuracy", f"{correct_outcomes/n*100:.1f}%" if n else "–")
    c3.metric("Exact-score hits", f"{correct_scores/n*100:.1f}%" if n else "–")
    c4.metric(
        "Avg model confidence",
        f"{avg_p*100:.1f}%" if n else "–",
        help="Average probability the model assigned to what actually happened. "
             "Higher is better.",
    )

    st.divider()
    st.subheader("Loaded results")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download current results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="results_2026.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Page: Methodology
# ---------------------------------------------------------------------------
def page_methodology(hist: pd.DataFrame, qualifiers: pd.DataFrame):
    st.title("📐 Methodology & Notes")

    base_gf, base_ga = baseline_goals(hist)
    n_qual = len(qualifiers) // 2 if not qualifiers.empty else 0
    st.markdown(
        f"""
### How predictions are made

The engine is a **bivariate Poisson model** — the standard statistical
approach for football scorelines.

**Step 1 — Team profile.** For each team, compute their average goals
scored and conceded across every World Cup match in the dataset
(`world-cup-Data-all-matches-2.csv`, {len(hist)//2:,} matches).

**Step 2 — Expected goals (xG).** For team A vs team B,
expected goals for A is the average of A's *attack rate* and B's
*defense rate*. This balances who scores often against who concedes often.

**Step 2b — 2026 qualifying form blend.** Each team's average goals
scored / conceded during 2026 qualifying ({n_qual} qualifying matches
in the dataset) are blended into their attack and defense ratings.
The weight scales with the number of qualifying matches a team has —
teams with more qualifying data get a stronger blend (capped at the
*Qualifier weight* slider value, default 75% — see calibration note
below). Use the sidebar toggle to turn this off entirely.

**Step 3 — Head-to-head adjustment.** If the two teams have met at a
previous World Cup, blend the historical head-to-head goal averages
into the xG (default weight: 20%).

**Step 4 — Match probabilities.** With expected goals λₐ and λ_b,
each team's goal count follows a Poisson distribution. We compute the
full score-line grid `P(A=i, B=j) = Pois(i; λₐ) · Pois(j; λ_b)` then
sum the appropriate cells for win / draw / loss probabilities.

The **most likely scoreline** is the highest-probability cell in the grid.

### Tournament baseline

Across all historical matches, the average team scores
**{base_gf:.2f}** and concedes **{base_ga:.2f}** goals per match.
These are the fall-back values for debutant teams.

### Historical name resolution

Several 2026 sides have played at past World Cups under a different name.
The app inherits these records via the `HISTORICAL_NAMES` mapping in
`app.py`:

| 2026 team   | Inherits World Cup record of                                  |
|-------------|---------------------------------------------------------------|
| Czech Rep.  | Czech Republic + **Czechoslovakia**                           |
| Germany     | Germany + **West Germany** (matches FIFA's own treatment)     |
| Russia      | Russia + **Soviet Union**                                     |
| Serbia      | Serbia + **Serbia and Montenegro** + **Yugoslavia**           |
| DR Congo    | **Zaire**                                                     |

Head-to-head lookups also use these aliases, so e.g. France-vs-Czech-Rep.
at 2026 will see France's previous meetings with Czechoslovakia.

### What to do about genuine debutant teams

Some 2026 sides have *no* prior World Cup data under any name
(**Cape Verde**, **Curaçao**, **Jordan**, **Uzbekistan**).
Possible strategies — current implementation uses **option 1**, but
you can swap in others by editing `expected_goals()` in `app.py`:

1. **2026 qualifying form (current default when data is present).**
   Use the team's average goals scored and conceded during 2026
   qualifying. Best signal we have for first-timers — it's recent and
   reflects the actual squad. Sample size is small (1–3 matches per
   debutant in the dataset), so it's blended toward the tournament
   baseline using `sample_factor = n / (n + 4)`.

0. **Tournament-baseline fallback.** Used only when no qualifying
   data exists for a team. Average goals-scored / goals-conceded
   across all World Cup matches, scaled by `debutant_strength`
   (default 0.70).

2. **FIFA-ranking-based prior.** Pull the team's current FIFA Coca-Cola
   ranking and map it to an attack/defense rating using a logistic
   curve. More accurate but requires an extra data source.

3. **Continental peer benchmark.** Use the average of confederation
   peers (e.g. give Cape Verde the average stats of African teams' first
   World Cup appearances). Reasonable middle-ground.

4. **Qualifying-campaign data.** Use goals scored/conceded during the
   2026 qualifying tournament. Best signal but means joining another
   dataset.

5. **Bayesian shrinkage.** Start every team — debutant or not — at the
   tournament baseline and shrink toward their own average proportional
   to `matches_played / (matches_played + k)`. Avoids the binary
   debutant/veteran cliff. (The current qualifier blend already does
   something similar with sample-size weighting.)

### Calibration against played 2026 matches

The model defaults have been calibrated twice against actual results.
We sweep the three main parameters and pick the combination that
best fits real outcomes (Win / Draw / Loss).

**Calibrated defaults (set after matchday 2, confirmed at matchday 3):**

| Parameter           | Original | **Calibrated** | Effect                                           |
|---------------------|----------|----------------|--------------------------------------------------|
| `qualifier_weight`  | 0.35     | **0.75**       | Trust 2026 form more — it's the freshest signal  |
| `h2h_weight`        | 0.30     | **0.20**       | Old WC meetings matter less than current form    |
| `debutant_strength` | 0.85     | **0.70**       | Treat debutants a touch more cautiously          |

**Performance vs. actual results:**

| Sample size           | Original defaults | Calibrated defaults | + live results feed |
|-----------------------|-------------------|---------------------|---------------------|
| 28 matches (MD1–2)    | 46.4% accuracy    | **57.1%**           | n/a (too sparse)    |
| 56 matches (MD1–3)    | 53.6% accuracy    | **62.5%**           | 62.5%               |

With only 28 matches, feeding played results back into the model
didn't help (each team had ≤2 datapoints). With 56 matches, the
"Use 2026 results in the model" toggle now lifts the *baseline*
config from 53.6% to 58.9% — turn it on once the group stage is
complete.

The biggest single lesson: **2026 qualifying form is a much stronger
predictor than historical World Cup form** for this tournament —
squads turn over, qualifying is recent, and several heavy WC names
(Spain, Brazil) have underperformed their long-run averages while
strong qualifiers (Sweden, Germany, Canada, England) have backed it up.
You can revert to the older settings via the sidebar sliders if you'd
like to A/B test.

### Knockout simulator — Fox Sports per-process blend

The Knockout Simulator uses **2026 tournament form only** for its base λ
(goals scored / conceded per game in the group stage), then optionally
blends in **per-process stats from Fox Sports** to sharpen the signal
when group form is noisy (only 3 games per team).

**Source:** [foxsports.com / fifa-world-cup-men / team-stats](https://www.foxsports.com/soccer/fifa-world-cup-men/team-stats?category=offensive&season=2026&groupId=12)
— see the new **📈 Team Stats** page for the raw tables.

**Strength scores (z-scores over all 48 nations):**

* **Attack z** = average z-score of shots-on-goal/game, chances-created/game,
  possession %, and passing accuracy.
* **Defence z** = average z-score of tackles/game, interceptions/game,
  *minus* fouls/game (less reckless defending counts as better).

**Blend into λ:**

```
m_A = (1 + w · attack_A) · (1 − w · defence_B)
λ_A ← λ_A · clip(m_A, 0.55, 1.65)
```

where `w` is the Fox-stat weight slider on the Knockout Simulator page
(default 0.18). The multiplier is clipped to [0.55, 1.65] so a single
per-process advantage can never more than ~halve or double a team's xG.
Teams with no defensive stats (Colombia, Japan, Cape Verde, Scotland,
Uzbekistan) get a neutral defence z = 0.

### Limitations

* Historical World Cup data only — domestic and continental form is ignored.
* Team strengths drift over decades; the model treats 1950 Brazil and
  2022 Brazil identically. A time-decay weight could be added.
* Home-advantage is not modelled (2026 has three hosts).
* Penalty shoot-outs and extra time are not modelled separately.
* Fox Sports per-process stats are season-level snapshots (no per-match
  granularity), so they cannot capture form trends within the tournament.
"""
    )


if __name__ == "__main__":
    main()
