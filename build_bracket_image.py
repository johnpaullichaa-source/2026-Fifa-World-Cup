"""Generate a filled-in bracket image showing the simulator's most likely path.

Each match resolves to: favourite wins by their modal Poisson scoreline.
"""
import sys, types, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import poisson

HERE = Path(__file__).resolve().parent

# stub streamlit so app.py can be imported
st_stub = types.ModuleType("streamlit")
class _D:
    def __call__(self,*a,**k): return self
    def __getattr__(self,n): return self
    def __enter__(self): return self
    def __exit__(self,*a): return False
st_stub.sidebar=_D(); st_stub.cache_data=lambda *a,**k:(lambda f:f)
for fn in ("set_page_config","markdown","title","header","subheader","write",
           "columns","metric","selectbox","slider","toggle","checkbox",
           "file_uploader","tabs","dataframe","table","info","warning","error",
           "success","caption","plotly_chart","pyplot","container","expander",
           "empty","rerun","stop","button","spinner","divider","radio"):
    setattr(st_stub, fn, lambda *a,**k: _D())
sys.modules["streamlit"]=st_stub
spec=importlib.util.spec_from_file_location("wc", HERE/"app.py")
wc=importlib.util.module_from_spec(spec); spec.loader.exec_module(wc)

results = pd.read_csv(HERE/"results_2026.csv")
form = wc.tournament_form_profile(results)
games = sum(p["games"] for p in form.values())
tavg = sum(p["gf_per"]*p["games"] for p in form.values())/games

# Fox Sports per-process strengths
FOX_WEIGHT = 0.18
strengths = wc.build_team_strengths()
print(f"Fox Sports strengths loaded for {len(strengths)} teams; weight={FOX_WEIGHT}")

def modal_score(team_a, team_b):
    """Return the most likely scoreline using the same xG model as the sim."""
    la, lb = wc.knockout_expected_goals(
        team_a, team_b, form, tavg,
        strengths=strengths, fox_weight=FOX_WEIGHT,
    )
    a = np.array([poisson.pmf(i, la) for i in range(8)])
    b = np.array([poisson.pmf(i, lb) for i in range(8)])
    grid = np.outer(a, b)
    i, j = np.unravel_index(np.argmax(grid), grid.shape)
    return int(i), int(j), la, lb

def resolve(team_a, team_b):
    """Favourite advances; returns (winner, score_a, score_b, et_flag)."""
    pa, pb, la, lb = wc.knockout_win_prob(
        team_a, team_b, form, tavg,
        strengths=strengths, fox_weight=FOX_WEIGHT,
    )
    sa, sb, _, _ = modal_score(team_a, team_b)
    # Determine favourite first (single source of truth)
    # Tiebreaker when win probs are equal: better goal differential per game
    if abs(pa - pb) < 1e-6:
        gd_a = form.get(team_a, {"gf_per":0,"ga_per":0})
        gd_b = form.get(team_b, {"gf_per":0,"ga_per":0})
        a_is_favourite = (gd_a["gf_per"] - gd_a["ga_per"]) >= (gd_b["gf_per"] - gd_b["ga_per"])
    else:
        a_is_favourite = pa > pb
    # If modal score is a draw, bump favourite by 1 to reflect ET/pens decisive
    et = False
    if sa == sb:
        et = True
        if a_is_favourite:
            sa += 1
        else:
            sb += 1
    else:
        # Ensure modal scoreline aligns with favourite (rare edge case)
        if (sa > sb) != a_is_favourite:
            sa, sb = sb, sa
    winner = team_a if a_is_favourite else team_b
    return winner, sa, sb, et

# Walk the bracket
r32 = wc.KNOCKOUT_BRACKET_R32
r32_results = [resolve(a,b) for a,b in r32]
r16_pairs   = [(r32_results[i][0], r32_results[i+1][0]) for i in range(0,16,2)]
r16_results = [resolve(a,b) for a,b in r16_pairs]
qf_pairs    = [(r16_results[i][0], r16_results[i+1][0]) for i in range(0,8,2)]
qf_results  = [resolve(a,b) for a,b in qf_pairs]
sf_pairs    = [(qf_results[i][0], qf_results[i+1][0]) for i in range(0,4,2)]
sf_results  = [resolve(a,b) for a,b in sf_pairs]
final_pair  = (sf_results[0][0], sf_results[1][0])
final_result= resolve(*final_pair)
champion    = final_result[0]

# Print for verification
print("=== R32 ===")
for (a,b),(w,sa,sb,et) in zip(r32, r32_results):
    print(f"  {a:18} {sa}-{sb} {b:18}  -> {w}{' (AET)' if et else ''}")
print("=== R16 ===")
for (a,b),(w,sa,sb,et) in zip(r16_pairs, r16_results):
    print(f"  {a:18} {sa}-{sb} {b:18}  -> {w}{' (AET)' if et else ''}")
print("=== QF ===")
for (a,b),(w,sa,sb,et) in zip(qf_pairs, qf_results):
    print(f"  {a:18} {sa}-{sb} {b:18}  -> {w}{' (AET)' if et else ''}")
print("=== SF ===")
for (a,b),(w,sa,sb,et) in zip(sf_pairs, sf_results):
    print(f"  {a:18} {sa}-{sb} {b:18}  -> {w}{' (AET)' if et else ''}")
print("=== FINAL ===")
a,b = final_pair; w,sa,sb,et = final_result
print(f"  {a:18} {sa}-{sb} {b:18}  -> {w}{' (AET)' if et else ''}")
print(f"\n*** CHAMPION: {champion} ***")

# --- Draw the bracket ---
fig, ax = plt.subplots(figsize=(26, 12), dpi=140)
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")
ax.set_xlim(0, 30)
ax.set_ylim(0, 18)
ax.axis("off")

# Colours
BOX_BG       = "#1f6b6e"
BOX_BG_WIN   = "#2dd4bf"
BOX_BG_CHAMP = "#fbbf24"
TEXT_LIGHT   = "#0d1117"
TEXT_DARK    = "#f0f9ff"
LINE         = "#2dd4bf"

def draw_match(x, y, team_a, team_b, score_a, score_b, winner, w=2.6, h=0.55, et=False):
    """Draw a two-team match box with scores; winner is highlighted."""
    # Team A box
    a_winner = (winner == team_a)
    b_winner = (winner == team_b)
    for k, (team, score, is_winner) in enumerate([
        (team_a, score_a, a_winner),
        (team_b, score_b, b_winner),
    ]):
        bg = BOX_BG_WIN if is_winner else BOX_BG
        fg = TEXT_LIGHT if is_winner else TEXT_DARK
        rect = FancyBboxPatch(
            (x, y - k*h - h/2), w, h*0.9,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=0, facecolor=bg, zorder=2,
        )
        ax.add_patch(rect)
        ax.text(x + 0.12, y - k*h - h/2 + h*0.45, team,
                fontsize=8.5, color=fg, weight="bold",
                ha="left", va="center", zorder=3)
        ax.text(x + w - 0.15, y - k*h - h/2 + h*0.45, str(score),
                fontsize=10, color=fg, weight="bold",
                ha="right", va="center", zorder=3)
    if et:
        ax.text(x + w/2, y - h - h/2 - 0.15, "AET",
                fontsize=6, color="#fcd34d", ha="center", va="top", zorder=3)

def draw_connector(x1, y1, x2, y2):
    """Connect winner's box right-edge to next box left-edge with a bent line."""
    mid = (x1 + x2)/2
    ax.plot([x1, mid, mid, x2], [y1, y1, y2, y2],
            color=LINE, linewidth=1.2, zorder=1, alpha=0.7)

# Layout: R32, R16, QF, SF on the left; mirror on the right; FINAL in the middle.
# Boxes are 2.6 wide; rounds spaced 3.1 apart on each side.
W = 2.6
LEFT_X  = [0.3, 3.4, 6.5, 9.6]      # R32, R16, QF, SF (left side)
RIGHT_X = [27.1, 24.0, 20.9, 17.8]   # R32, R16, QF, SF (right side) — mirror with bigger middle gap

# Vertical centres for R32 on left (8 matches spaced 1.9 apart, centred ~y=9.5)
def y_centers(n, total_height=15.5, top=16.5):
    if n == 1:
        return [(top - total_height/2)]
    step = total_height / (n - 1) if n > 1 else 0
    return [top - i*step for i in range(n)]

# Left R32 (matches 0..7) and Right R32 (matches 8..15)
left_y_r32  = y_centers(8, total_height=15.0, top=17.0)
right_y_r32 = y_centers(8, total_height=15.0, top=17.0)

# Subsequent rounds: midpoint of feeders
def next_round_ys(prev_ys):
    return [(prev_ys[i] + prev_ys[i+1])/2 for i in range(0, len(prev_ys), 2)]

left_y_r16 = next_round_ys(left_y_r32)
left_y_qf  = next_round_ys(left_y_r16)
left_y_sf  = next_round_ys(left_y_qf)

right_y_r16 = next_round_ys(right_y_r32)
right_y_qf  = next_round_ys(right_y_r16)
right_y_sf  = next_round_ys(right_y_qf)

# Final — placed between the two SF columns
final_y = (left_y_sf[0] + right_y_sf[0]) / 2
final_x = (LEFT_X[3] + W + RIGHT_X[3]) / 2 - W/2  # midpoint between left-SF right-edge and right-SF left-edge

# Draw left side
def draw_round(round_pairs, round_results, xs, ys, side="left"):
    for k, ((a,b),(w,sa,sb,et)) in enumerate(zip(round_pairs, round_results)):
        draw_match(xs, ys[k], a, b, sa, sb, w, et=et)

# Left R32
for k, ((a,b),(w,sa,sb,et)) in enumerate(zip(r32[:8], r32_results[:8])):
    draw_match(LEFT_X[0], left_y_r32[k], a, b, sa, sb, w, et=et)
# Right R32
for k, ((a,b),(w,sa,sb,et)) in enumerate(zip(r32[8:], r32_results[8:])):
    draw_match(RIGHT_X[0], right_y_r32[k], a, b, sa, sb, w, et=et)
# Left R16, QF, SF
for k, ((a,b),(w,sa,sb,et)) in enumerate(zip(r16_pairs[:4], r16_results[:4])):
    draw_match(LEFT_X[1], left_y_r16[k], a, b, sa, sb, w, et=et)
for k, ((a,b),(w,sa,sb,et)) in enumerate(zip(qf_pairs[:2], qf_results[:2])):
    draw_match(LEFT_X[2], left_y_qf[k], a, b, sa, sb, w, et=et)
(a,b),(w,sa,sb,et) = sf_pairs[0], sf_results[0]
draw_match(LEFT_X[3], left_y_sf[0], a, b, sa, sb, w, et=et)
# Right R16, QF, SF
for k, ((a,b),(w,sa,sb,et)) in enumerate(zip(r16_pairs[4:], r16_results[4:])):
    draw_match(RIGHT_X[1], right_y_r16[k], a, b, sa, sb, w, et=et)
for k, ((a,b),(w,sa,sb,et)) in enumerate(zip(qf_pairs[2:], qf_results[2:])):
    draw_match(RIGHT_X[2], right_y_qf[k], a, b, sa, sb, w, et=et)
(a,b),(w,sa,sb,et) = sf_pairs[1], sf_results[1]
draw_match(RIGHT_X[3], right_y_sf[0], a, b, sa, sb, w, et=et)

# Final box (centered)
(a,b),(w,sa,sb,et) = final_pair, final_result
draw_match(final_x, final_y, a, b, sa, sb, w, et=et)

# Connectors — left side
def connect_pair(x_from, ys_from, x_to, ys_to):
    """ys_from has 2N elements (feeders), ys_to has N (destinations)."""
    h = 0.55
    for i in range(0, len(ys_from), 2):
        # right edge of feeder box midpoint
        x1 = x_from + W
        # use winner row's y (top row = ys_from[i], bottom row = ys_from[i] - h)
        y_top = ys_from[i] - h/2 + h*0.45
        y_bot = ys_from[i] - h - h/2 + h*0.45
        # average both rows as feeder anchor
        y1 = (y_top + y_bot) / 2
        # next box left edge midpoint (between its two rows)
        target_y_top = ys_to[i//2] - h/2 + h*0.45
        target_y_bot = ys_to[i//2] - h - h/2 + h*0.45
        y2 = (target_y_top + target_y_bot) / 2
        draw_connector(x1, y1, x_to, y2)

def connect_pair_right(x_from, ys_from, x_to, ys_to):
    """Right side: connectors go from left edge of feeder to right edge of next box."""
    h = 0.55
    for i in range(0, len(ys_from), 2):
        x1 = x_from           # left edge of feeder
        y1 = ys_from[i] - h/2 + h*0.45
        y1b = ys_from[i] - h - h/2 + h*0.45
        y1 = (y1 + y1b)/2
        x2 = x_to + W         # right edge of next box
        y2_top = ys_to[i//2] - h/2 + h*0.45
        y2_bot = ys_to[i//2] - h - h/2 + h*0.45
        y2 = (y2_top + y2_bot)/2
        draw_connector(x1, y1, x2, y2)

connect_pair(LEFT_X[0], left_y_r32, LEFT_X[1], left_y_r16)
connect_pair(LEFT_X[1], left_y_r16, LEFT_X[2], left_y_qf)
connect_pair(LEFT_X[2], left_y_qf, LEFT_X[3], left_y_sf)
connect_pair_right(RIGHT_X[0], right_y_r32, RIGHT_X[1], right_y_r16)
connect_pair_right(RIGHT_X[1], right_y_r16, RIGHT_X[2], right_y_qf)
connect_pair_right(RIGHT_X[2], right_y_qf, RIGHT_X[3], right_y_sf)

# SF -> Final
h = 0.55
left_sf_anchor  = (LEFT_X[3] + W,  left_y_sf[0] - h*0.55)
right_sf_anchor = (RIGHT_X[3],     right_y_sf[0] - h*0.55)
final_left      = (final_x,        final_y - h*0.55)
final_right     = (final_x + W,    final_y - h*0.55)
draw_connector(left_sf_anchor[0], left_sf_anchor[1], final_left[0], final_left[1])
draw_connector(right_sf_anchor[0], right_sf_anchor[1], final_right[0], final_right[1])

# Round labels (top of each column)
labels = ["R32", "R16", "QF", "SF", "FINAL", "SF", "QF", "R16", "R32"]
xs     = [LEFT_X[0]+W/2, LEFT_X[1]+W/2, LEFT_X[2]+W/2, LEFT_X[3]+W/2,
          final_x+W/2,
          RIGHT_X[3]+W/2, RIGHT_X[2]+W/2, RIGHT_X[1]+W/2, RIGHT_X[0]+W/2]
for lab, x in zip(labels, xs):
    ax.text(x, 17.7, lab, fontsize=11, color="#94a3b8", weight="bold",
            ha="center", va="center")

# Title + Champion banner
MIDX = 15.0
ax.text(MIDX, 1.0, f"CHAMPION: {champion.upper()}",
        fontsize=26, color=BOX_BG_CHAMP, weight="bold",
        ha="center", va="center", zorder=5)
final_w, final_score_a, final_score_b, final_et = final_result
ax.text(MIDX, 0.35, f"Final: {final_pair[0]} {final_result[1]}–{final_result[2]} {final_pair[1]}{' (AET)' if final_result[3] else ''}",
        fontsize=13, color="#fef3c7", ha="center", va="center", zorder=5)
ax.text(MIDX, 17.45, "World Cup 2026 — Predicted Knockout Bracket",
        fontsize=15, color="#e2e8f0", weight="bold", ha="center", va="center")
ax.text(MIDX, 17.05, "2026 group-stage form + Fox Sports per-process stats blend",
        fontsize=10, color="#94a3b8", ha="center", va="center", style="italic")

plt.tight_layout(pad=0)
out = HERE/"bracket_filled.png"
plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#0d1117")
print(f"\nSaved: {out}")
