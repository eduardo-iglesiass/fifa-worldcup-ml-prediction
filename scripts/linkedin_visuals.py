import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

BASE        = Path(__file__).parent.parent
RESULTS_DIR = BASE / "outputs" / "results"
FIGURES_DIR = BASE / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BG       = "#0d1117"
CARD     = "#161b22"
BORDER   = "#21262d"
GOLD     = "#FFC300"
BLUE     = "#00A8E8"
WHITE    = "#FFFFFF"
TEXT     = "#E8E8E8"
DIM      = "#8B949E"
GREEN    = "#3fb950"
RED      = "#f85149"
GOLD_DIM = "#7a5e00"

win_probs = pd.read_csv(RESULTS_DIR / "wc2026_win_probabilities.csv")
top5_df   = pd.read_csv(RESULTS_DIR / "wc2026_top5_outcomes.csv")
elo_df    = pd.read_csv(BASE / "data" / "current_elo_ratings.csv")

elo_raw = dict(zip(elo_df["team"], elo_df["elo"]))

_ELO_MAP = {
    "United States": "United States",
    "Turkiye":       "Turkey",
    "Czechia":       "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Curacao":       "Curaçao",
}

def get_elo(team: str) -> float:
    return elo_raw.get(_ELO_MAP.get(team, team), elo_raw.get(team, 1500))

def ko_prob(team1: str, team2: str) -> tuple[float, float]:
    e1, e2 = get_elo(team1), get_elo(team2)
    p = 1.0 / (1.0 + 10 ** ((e2 - e1) / 400.0))
    return round(p * 100, 1), round((1 - p) * 100, 1)

# Short display names for tight bracket boxes
SHORT = {
    "United States":      "USA",
    "South Korea":        "S. Korea",
    "Bosnia-Herzegovina": "Bosnia",
    "Saudi Arabia":       "Saudi Ara.",
    "Cape Verde":         "C. Verde",
    "Ivory Coast":        "Ivory Cst",
    "New Zealand":        "N. Zealand",
    "Bosnia and Herzegovina": "Bosnia",
    "DR Congo":           "DR Congo",
    "Switzerland":        "Switzerland",
}

def sn(team: str) -> str:
    return SHORT.get(team, team)


def make_infographic():
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(BG)

    ax_bar  = fig.add_axes([0.04, 0.09, 0.55, 0.74])
    ax_stage = fig.add_axes([0.63, 0.09, 0.35, 0.74])
    for ax in (ax_bar, ax_stage):
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_visible(False)

    top_n = 15
    df = win_probs.head(top_n).copy()
    df = df.sort_values("win_pct")
    teams  = df["team"].tolist()
    probs  = df["win_pct"].tolist()
    colors = [GOLD if t == "Spain" else BLUE for t in teams]

    bars = ax_bar.barh(teams, probs, color=colors, height=0.65,
                       edgecolor="none", zorder=3)

    ax_bar.xaxis.set_tick_params(colors=DIM, labelsize=9)
    ax_bar.yaxis.set_tick_params(colors=TEXT, labelsize=10.5)
    ax_bar.tick_params(axis="x", colors=DIM)
    ax_bar.tick_params(axis="y", colors=TEXT)
    ax_bar.set_xlim(0, 26)
    ax_bar.set_xlabel("Win Probability (%)", color=DIM, fontsize=10, labelpad=8)

    for v in [5, 10, 15, 20, 25]:
        ax_bar.axvline(v, color=BORDER, linewidth=0.7, zorder=1, alpha=0.8)

    for bar, p, team in zip(bars, probs, teams):
        color = BG if team == "Spain" else WHITE
        weight = "bold" if team == "Spain" else "normal"
        ax_bar.text(p + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{p:.1f}%", va="center", ha="left",
                    color=GOLD if team == "Spain" else TEXT,
                    fontsize=9.5, fontweight=weight)

    spain_bar = bars[teams.index("Spain")]
    ax_bar.text(spain_bar.get_width() - 0.6,
                spain_bar.get_y() + spain_bar.get_height() / 2,
                "* Favourite", va="center", ha="right",
                color=BG, fontsize=8.5, fontweight="bold")

    ax_bar.set_title("Top 15 Teams — Tournament Win Probability",
                     color=TEXT, fontsize=13, pad=14, loc="left", fontweight="bold")

    qf_cols   = ["Quarterfinal", "Semifinal", "Runner-up", "Champion"]
    top5_teams = ["Spain", "Argentina", "France", "Brazil", "England"]

    rows = []
    for team in top5_teams:
        r = top5_df[top5_df["team"] == team].iloc[0]
        qf_prob    = sum(float(r[c]) for c in qf_cols)
        champ_prob = float(r["Champion"])
        rows.append((team, qf_prob, champ_prob))

    rows.sort(key=lambda x: x[1])
    t_names  = [r[0] for r in rows]
    qf_probs = [r[1] for r in rows]
    ch_probs = [r[2] for r in rows]
    bar_cols = [GOLD if t == "Spain" else BLUE for t in t_names]

    ax_stage.set_facecolor(BG)
    for sp in ax_stage.spines.values():
        sp.set_visible(False)

    bars_qf = ax_stage.barh(t_names, qf_probs, color=bar_cols,
                             height=0.52, edgecolor="none", zorder=3)

    ax_stage.set_xlim(0, max(qf_probs) * 1.28)
    ax_stage.tick_params(axis="x", colors=DIM, labelsize=9)
    ax_stage.tick_params(axis="y", colors=TEXT, labelsize=11)
    ax_stage.set_xlabel("Probability (%)", color=DIM, fontsize=10, labelpad=8)

    for v in [20, 40, 60]:
        ax_stage.axvline(v, color=BORDER, lw=0.7, zorder=1, alpha=0.8)

    for bar, qp, cp, team in zip(bars_qf, qf_probs, ch_probs, t_names):
        by = bar.get_y() + bar.get_height() / 2
        ax_stage.text(qp + 0.8, by, f"{qp:.1f}%",
                      va="center", ha="left",
                      color=GOLD if team == "Spain" else TEXT,
                      fontsize=10, fontweight="bold" if team == "Spain" else "normal")
        label_x = qp - 1.2
        ax_stage.text(label_x, by, f"Win: {cp:.1f}%",
                      va="center", ha="right",
                      color=BG if team == "Spain" else "#0d1117",
                      fontsize=8.5, fontweight="bold")

    ax_stage.set_title("Probability of Reaching Quarter-Finals or Better",
                       color=TEXT, fontsize=12.5, pad=14,
                       loc="left", fontweight="bold")

    fig.text(0.50, 0.955, "FIFA WORLD CUP 2026  ·  TOURNAMENT PREDICTIONS",
             ha="center", va="top", color=WHITE,
             fontsize=22, fontweight="bold")
    fig.text(0.50, 0.924, "Monte Carlo Simulation  ·  10,000 Tournaments  ·  Logistic Regression Model (59.9% accuracy on 2022 WC)",
             ha="center", va="top", color=DIM, fontsize=11)

    fig.add_artist(plt.Line2D([0.04, 0.96], [0.915, 0.915],
                              transform=fig.transFigure,
                              color=BORDER, linewidth=1.0))

    legend_y = 0.046
    patches = [
        mpatches.Patch(color=GOLD, label="Spain (Predicted Champion)"),
        mpatches.Patch(color=BLUE, label="Other teams"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2,
               facecolor=BG, edgecolor=BORDER,
               labelcolor=TEXT, fontsize=10,
               bbox_to_anchor=(0.30, legend_y))

    fig.text(0.96, legend_y + 0.01,
             "Model: Logistic Regression  |  Features: Elo, FIFA pts, form, H2H  |  github.com/worldcup2026",
             ha="right", va="center", color=DIM, fontsize=8.5)

    out = FIGURES_DIR / "linkedin_infographic.png"
    fig.savefig(out, dpi=100, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")


BRACKET_LEFT = [
    ("Germany",     "Sweden",       "1E vs 3ABCDF"),
    ("France",      "Egypt",        "1I vs 3CDFGH"),
    ("S. Korea",    "Canada",       "2A vs 2B"),
    ("Netherlands", "Brazil",       "1F vs 2C"),
    ("Colombia",    "Croatia",      "2K vs 2L"),
    ("Spain",       "Austria",      "1H vs 2J"),
    ("Turkiye",     "Norway",       "1D vs 3BEFIJ"),
    ("Belgium",     "Algeria",      "1G vs 3AEHIJ"),
]

BRACKET_RIGHT = [
    ("Morocco",      "Japan",         "1C vs 2F"),
    ("Ecuador",      "Senegal",       "2E vs 2I"),
    ("Mexico",       "Scotland",      "1A vs 3CEFHI"),
    ("England",      "Uzbekistan",    "1L vs 3EHIJK"),
    ("Argentina",    "Uruguay",       "1J vs 2H"),
    ("Australia",    "Iran",          "2D vs 2G"),
    ("Switzerland",  "Ivory Coast",   "1B vs 3EFGIJ"),
    ("Portugal",     "Panama",        "1K vs 3DEJL"),
]

LEFT_R32_WIN  = ["Germany","France","Canada","Netherlands","Colombia","Spain","Turkiye","Belgium"]
LEFT_R16      = [("France","Germany"),("Netherlands","Canada"),("Spain","Colombia"),("Belgium","Turkiye")]
LEFT_R16_WIN  = ["France","Netherlands","Spain","Belgium"]
LEFT_QF       = [("France","Netherlands"),("Spain","Belgium")]
LEFT_QF_WIN   = ["France","Spain"]
LEFT_SF       = [("Spain","France")]
LEFT_SF_WIN   = ["Spain"]

RIGHT_R32_WIN = ["Morocco","Ecuador","Mexico","England","Argentina","Australia","Switzerland","Portugal"]
RIGHT_R16     = [("Morocco","Ecuador"),("England","Mexico"),("Argentina","Australia"),("Portugal","Switzerland")]
RIGHT_R16_WIN = ["Morocco","England","Argentina","Portugal"]
RIGHT_QF      = [("Morocco","England"),("Argentina","Portugal")]
RIGHT_QF_WIN  = ["England","Argentina"]
RIGHT_SF      = [("Argentina","England")]
RIGHT_SF_WIN  = ["Argentina"]

FINAL         = ("Spain","Argentina")
CHAMPION      = "Spain"

SPAIN_PATH_TEAMS = {"Spain","Austria","Colombia","Belgium","France","Argentina"}

def _is_spain_match(t1, t2):
    return "Spain" in (t1, t2)

def _spain_in_set(t1, t2):
    return t1 in SPAIN_PATH_TEAMS or t2 in SPAIN_PATH_TEAMS

def make_bracket():
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(BG)

    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_facecolor(BG)

    BW = 0.095
    BH = 0.058
    ROW = BH / 2

    XL = {
        "r32": 0.010,
        "r16": 0.115,
        "qf":  0.220,
        "sf":  0.325,
    }
    X_FINAL = 0.4525
    XR = {
        "sf":  0.580,
        "qf":  0.685,
        "r16": 0.790,
        "r32": 0.895,
    }

    Y_TOP = 0.87
    Y_BOT = 0.10
    span  = Y_TOP - Y_BOT
    y8 = [Y_BOT + (7 - i) * span / 7 for i in range(8)]

    def y4(y8_list):
        return [(y8_list[i*2] + y8_list[i*2+1]) / 2 for i in range(4)]

    def y2(y4_list):
        return [(y4_list[i*2] + y4_list[i*2+1]) / 2 for i in range(2)]

    def y1(y2_list):
        return [(y2_list[0] + y2_list[1]) / 2]

    yl_r32 = y8
    yl_r16 = y4(yl_r32)
    yl_qf  = y2(yl_r16)
    yl_sf  = y1(yl_qf)
    y_fin  = yl_sf[0]

    yr_r32 = y8
    yr_r16 = y4(yr_r32)
    yr_qf  = y2(yr_r16)
    yr_sf  = y1(yr_qf)

    def box_color(t1, t2):
        if "Spain" in (t1, t2):
            return "#1c1800", GOLD, 1.8
        elif t1 in SPAIN_PATH_TEAMS or t2 in SPAIN_PATH_TEAMS:
            return "#0f1520", "#3a7fd5", 1.0
        return CARD, BORDER, 0.7

    def draw_match(x, y_c, t1, t2, winner=None, font_s=7.5):
        bg, border, lw = box_color(t1, t2)
        y_bot = y_c - BH / 2

        rect = FancyBboxPatch(
            (x, y_bot), BW, BH,
            boxstyle="round,pad=0.003",
            facecolor=bg, edgecolor=border,
            linewidth=lw, transform=ax.transAxes, zorder=4,
        )
        ax.add_patch(rect)

        ax.plot([x + 0.004, x + BW - 0.004], [y_c, y_c],
                color=border, lw=0.5, transform=ax.transAxes, zorder=5)

        p1, p2 = ko_prob(t1, t2)

        for row_i, (team, prob) in enumerate([(t1, p1), (t2, p2)]):
            sign = 1 if row_i == 0 else -1
            yrow = y_c + sign * ROW * 0.5

            is_spain = (team == "Spain")
            is_winner = (team == winner)
            t_color = GOLD if is_spain else (WHITE if is_winner else TEXT)
            fw = "bold" if (is_spain or is_winner) else "normal"

            ax.text(x + 0.005, yrow, sn(team),
                    color=t_color, fontsize=font_s, fontweight=fw,
                    va="center", ha="left",
                    transform=ax.transAxes, zorder=6, clip_on=True)

            p_color = GREEN if is_winner else (GOLD if is_spain else DIM)
            ax.text(x + BW - 0.004, yrow, f"{prob:.0f}%",
                    color=p_color, fontsize=font_s - 0.5, fontweight=fw,
                    va="center", ha="right",
                    transform=ax.transAxes, zorder=6)

    def draw_connector_left(x_from, x_to, y1_c, y2_c, spain_match=False):
        col = GOLD if spain_match else DIM
        lw  = 1.4  if spain_match else 0.6
        alpha = 1.0 if spain_match else 0.55

        x_mid = (x_from + x_to) / 2
        y_mid = (y1_c + y2_c) / 2

        for yy in (y1_c, y2_c):
            ax.plot([x_from, x_mid], [yy, yy],
                    color=col, lw=lw, alpha=alpha,
                    transform=ax.transAxes, zorder=2)
        ax.plot([x_mid, x_mid], [y1_c, y2_c],
                color=col, lw=lw, alpha=alpha,
                transform=ax.transAxes, zorder=2)
        ax.plot([x_mid, x_to], [y_mid, y_mid],
                color=col, lw=lw, alpha=alpha,
                transform=ax.transAxes, zorder=2)

    def draw_connector_right(x_from, x_to, y1_c, y2_c, spain_match=False):
        col = GOLD if spain_match else DIM
        lw  = 1.4  if spain_match else 0.6
        alpha = 1.0 if spain_match else 0.55

        x_mid = (x_from + x_to) / 2
        y_mid = (y1_c + y2_c) / 2

        for yy in (y1_c, y2_c):
            ax.plot([x_from, x_mid], [yy, yy],
                    color=col, lw=lw, alpha=alpha,
                    transform=ax.transAxes, zorder=2)
        ax.plot([x_mid, x_mid], [y1_c, y2_c],
                color=col, lw=lw, alpha=alpha,
                transform=ax.transAxes, zorder=2)
        ax.plot([x_mid, x_to], [y_mid, y_mid],
                color=col, lw=lw, alpha=alpha,
                transform=ax.transAxes, zorder=2)

    for i, (t1, t2, _) in enumerate(BRACKET_LEFT):
        draw_match(XL["r32"], yl_r32[i], t1, t2, winner=LEFT_R32_WIN[i])

    for pair in range(4):
        i1, i2 = pair * 2, pair * 2 + 1
        spain = "Spain" in (LEFT_R32_WIN[i1], LEFT_R32_WIN[i2])
        draw_connector_left(
            XL["r32"] + BW, XL["r16"],
            yl_r32[i1], yl_r32[i2], spain_match=spain
        )

    for i, (t1, t2) in enumerate(LEFT_R16):
        draw_match(XL["r16"], yl_r16[i], t1, t2, winner=LEFT_R16_WIN[i])

    for pair in range(2):
        i1, i2 = pair * 2, pair * 2 + 1
        spain = "Spain" in (LEFT_R16_WIN[i1], LEFT_R16_WIN[i2])
        draw_connector_left(
            XL["r16"] + BW, XL["qf"],
            yl_r16[i1], yl_r16[i2], spain_match=spain
        )

    for i, (t1, t2) in enumerate(LEFT_QF):
        draw_match(XL["qf"], yl_qf[i], t1, t2, winner=LEFT_QF_WIN[i])

    spain = "Spain" in (LEFT_QF_WIN[0], LEFT_QF_WIN[1])
    draw_connector_left(
        XL["qf"] + BW, XL["sf"],
        yl_qf[0], yl_qf[1], spain_match=spain
    )

    t1, t2 = LEFT_SF[0]
    draw_match(XL["sf"], yl_sf[0], t1, t2, winner=LEFT_SF_WIN[0])

    spain = "Spain" in LEFT_SF_WIN
    x_fin_center = X_FINAL + BW / 2
    ax.plot([XL["sf"] + BW, X_FINAL], [yl_sf[0], y_fin],
            color=GOLD if spain else DIM, lw=1.4 if spain else 0.6,
            alpha=1.0 if spain else 0.55, transform=ax.transAxes, zorder=2)

    for i, (t1, t2, _) in enumerate(BRACKET_RIGHT):
        draw_match(XR["r32"], yr_r32[i], t1, t2, winner=RIGHT_R32_WIN[i])

    x_r32_left  = XR["r32"]
    x_r16_right = XR["r16"] + BW
    for pair in range(4):
        i1, i2 = pair * 2, pair * 2 + 1
        spain = "Spain" in (RIGHT_R32_WIN[i1], RIGHT_R32_WIN[i2])
        draw_connector_right(
            x_r32_left, x_r16_right,
            yr_r32[i1], yr_r32[i2], spain_match=spain
        )

    for i, (t1, t2) in enumerate(RIGHT_R16):
        draw_match(XR["r16"], yr_r16[i], t1, t2, winner=RIGHT_R16_WIN[i])

    x_r16_left = XR["r16"]
    x_qf_right = XR["qf"] + BW
    for pair in range(2):
        i1, i2 = pair * 2, pair * 2 + 1
        spain = "Spain" in (RIGHT_R16_WIN[i1], RIGHT_R16_WIN[i2])
        draw_connector_right(
            x_r16_left, x_qf_right,
            yr_r16[i1], yr_r16[i2], spain_match=spain
        )

    for i, (t1, t2) in enumerate(RIGHT_QF):
        draw_match(XR["qf"], yr_qf[i], t1, t2, winner=RIGHT_QF_WIN[i])

    spain = "Spain" in (RIGHT_QF_WIN[0], RIGHT_QF_WIN[1])
    x_qf_left = XR["qf"]
    x_sf_right = XR["sf"] + BW
    draw_connector_right(
        x_qf_left, x_sf_right,
        yr_qf[0], yr_qf[1], spain_match=spain
    )

    t1, t2 = RIGHT_SF[0]
    draw_match(XR["sf"], yr_sf[0], t1, t2, winner=RIGHT_SF_WIN[0])

    spain_r = "Spain" in RIGHT_SF_WIN
    ax.plot([XR["sf"], X_FINAL + BW], [yr_sf[0], y_fin],
            color=GOLD if spain_r else DIM, lw=1.4 if spain_r else 0.6,
            alpha=1.0 if spain_r else 0.55, transform=ax.transAxes, zorder=2)

    ft1, ft2 = FINAL
    draw_match(X_FINAL, y_fin, ft1, ft2, winner=CHAMPION, font_s=8.5)

    champ_y = y_fin - BH / 2 - 0.065
    champ_box = FancyBboxPatch(
        (X_FINAL - 0.005, champ_y - 0.022), BW + 0.010, 0.042,
        boxstyle="round,pad=0.004",
        facecolor=GOLD, edgecolor=GOLD, linewidth=0,
        transform=ax.transAxes, zorder=4
    )
    ax.add_patch(champ_box)
    ax.text(X_FINAL + BW / 2, champ_y - 0.001,
            f"{CHAMPION.upper()}  CHAMPIONS",
            color=BG, fontsize=10.5, fontweight="bold",
            ha="center", va="center",
            transform=ax.transAxes, zorder=5)

    round_labels = [
        (XL["r32"]  + BW / 2, 0.946, "ROUND OF 32",    DIM,  False),
        (XL["r16"]  + BW / 2, 0.946, "ROUND OF 16",    DIM,  False),
        (XL["qf"]   + BW / 2, 0.946, "QUARTERS",       DIM,  False),
        (XL["sf"]   + BW / 2, 0.946, "SEMIS",          DIM,  False),
        (X_FINAL    + BW / 2, 0.946, "FINAL",          GOLD, True),
        (XR["sf"]   + BW / 2, 0.946, "SEMIS",          DIM,  False),
        (XR["qf"]   + BW / 2, 0.946, "QUARTERS",       DIM,  False),
        (XR["r16"]  + BW / 2, 0.946, "ROUND OF 16",    DIM,  False),
        (XR["r32"]  + BW / 2, 0.946, "ROUND OF 32",    DIM,  False),
    ]
    for lx, ly, lab, color, bold in round_labels:
        ax.text(lx, ly, lab, ha="center", va="center",
                color=color, fontsize=8,
                fontweight="bold" if bold else "normal",
                transform=ax.transAxes)

    ax.plot([0.01, 0.99], [0.930, 0.930],
            color=BORDER, lw=0.8, transform=ax.transAxes, zorder=1)

    ax.text(0.50, 0.985,
            "FIFA WORLD CUP 2026  ·  KNOCKOUT BRACKET PREDICTION",
            ha="center", va="top", color=WHITE,
            fontsize=19, fontweight="bold", transform=ax.transAxes)
    ax.text(0.50, 0.964,
            "Most likely bracket based on 10,000 Monte Carlo simulations  ·  Gold path = predicted Spain route to the title",
            ha="center", va="top", color=DIM,
            fontsize=9.5, transform=ax.transAxes)

    legend_items = [
        (GOLD,  "Spain / Spain's path"),
        (BLUE,  "Other teams"),
        (GREEN, "Predicted winner of match"),
        (DIM,   "Win probability (Elo model, neutral ground)"),
    ]
    lx = 0.01
    ly = 0.025
    for col, lab in legend_items:
        dot = mpatches.Circle((lx + 0.006, ly), 0.005,
                               color=col, transform=ax.transAxes, zorder=6)
        ax.add_patch(dot)
        ax.text(lx + 0.015, ly, lab, va="center", ha="left",
                color=TEXT, fontsize=7.5, transform=ax.transAxes)
        lx += 0.22

    out = FIGURES_DIR / "linkedin_bracket.png"
    fig.savefig(out, dpi=100, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating Image 1: Infographic ...")
    make_infographic()
    print("Generating Image 2: Bracket ...")
    make_bracket()
    print("Done.")
