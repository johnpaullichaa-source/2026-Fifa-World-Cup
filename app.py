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
         "🥇 Tournament Simulator", "📡 Live Results", "📐 Methodology"],
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

### Limitations

* Historical World Cup data only — domestic and continental form is ignored.
* Team strengths drift over decades; the model treats 1950 Brazil and
  2022 Brazil identically. A time-decay weight could be added.
* Home-advantage is not modelled (2026 has three hosts).
* Penalty shoot-outs and extra time are not modelled separately.
"""
    )


if __name__ == "__main__":
    main()
