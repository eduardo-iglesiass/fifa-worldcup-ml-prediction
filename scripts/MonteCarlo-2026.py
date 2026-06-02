import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

GROUPS = {
    "A": ["Mexico",       "South Korea",  "South Africa",  "Czechia"],
    "B": ["Canada",       "Switzerland",  "Qatar",         "Bosnia-Herzegovina"],
    "C": ["Brazil",       "Morocco",      "Scotland",      "Haiti"],
    "D": ["United States","Paraguay",     "Australia",     "Turkiye"],
    "E": ["Germany",      "Ecuador",      "Ivory Coast",   "Curacao"],
    "F": ["Netherlands",  "Japan",        "Tunisia",       "Sweden"],
    "G": ["Belgium",      "Iran",         "Egypt",         "New Zealand"],
    "H": ["Spain",        "Uruguay",      "Saudi Arabia",  "Cape Verde"],
    "I": ["France",       "Senegal",      "Norway",        "Iraq"],
    "J": ["Argentina",    "Austria",      "Algeria",       "Jordan"],
    "K": ["Portugal",     "Colombia",     "Uzbekistan",    "DR Congo"],
    "L": ["England",      "Croatia",      "Panama",        "Ghana"],
}

ALL_TEAMS = [team for group in GROUPS.values() for team in group]

print("=" * 60)
print("PHASE 3 — LOADING MODEL AND DATA")
print("=" * 60)

with open("best_model.pkl", "rb") as f:
    package = pickle.load(f)

model    = package["model"]
imputer  = package["imputer"]
scaler   = package["scaler"]
features = package["features"]
name     = package["name"]

print(f"Loaded model: {name}")

elo_df = pd.read_csv("current_elo_ratings.csv")
elo_map = dict(zip(elo_df["team"], elo_df["elo"]))

feat_df = pd.read_csv("features_phase1.csv", parse_dates=["date"])

print(f"Elo ratings loaded for {len(elo_map)} teams")

print("\n" + "=" * 60)
print("BUILDING TEAM FEATURE PROFILES")
print("=" * 60)


def get_team_profile(team, feat_df, elo_map):
    home_matches = feat_df[feat_df["home_team"] == team].sort_values("date")
    away_matches = feat_df[feat_df["away_team"] == team].sort_values("date")

    profile = {}
    profile["elo"] = elo_map.get(team, 1500)

    fifa_pts = None
    if len(home_matches) > 0 and not pd.isna(home_matches.iloc[-1]["fifa_points_home"]):
        fifa_pts = home_matches.iloc[-1]["fifa_points_home"]
    elif len(away_matches) > 0 and not pd.isna(away_matches.iloc[-1]["fifa_points_away"]):
        fifa_pts = away_matches.iloc[-1]["fifa_points_away"]
    profile["fifa_points"] = fifa_pts if fifa_pts is not None else feat_df["fifa_points_home"].mean()

    if len(home_matches) > 0:
        last = home_matches.iloc[-1]
        profile["form_winrate"]  = last["home_form_winrate"]
        profile["form_gf"]       = last["home_form_gf"]
        profile["form_ga"]       = last["home_form_ga"]
    elif len(away_matches) > 0:
        last = away_matches.iloc[-1]
        profile["form_winrate"]  = last["away_form_winrate"]
        profile["form_gf"]       = last["away_form_gf"]
        profile["form_ga"]       = last["away_form_ga"]
    else:
        profile["form_winrate"]  = feat_df["home_form_winrate"].mean()
        profile["form_gf"]       = feat_df["home_form_gf"].mean()
        profile["form_ga"]       = feat_df["home_form_ga"].mean()

    wc_matches = feat_df[feat_df["tournament"] == "FIFA World Cup"]
    home_wc = wc_matches[wc_matches["home_team"] == team]
    away_wc = wc_matches[wc_matches["away_team"] == team]
    wc_years = pd.concat([home_wc["date"], away_wc["date"]]).dt.year.nunique()
    profile["wc_experience"] = wc_years

    return profile


team_profiles = {}
for team in ALL_TEAMS:
    team_profiles[team] = get_team_profile(team, feat_df, elo_map)

print(f"Built profiles for all {len(ALL_TEAMS)} teams")
print("\n  Top 10 teams by Elo:")
sorted_teams = sorted(team_profiles.items(), key=lambda x: x[1]["elo"], reverse=True)
for team, p in sorted_teams[:10]:
    print(f"    {team}: {p['elo']:.0f}")


def predict_match(team_a, team_b, team_profiles, model, imputer, scaler,
                  features, feat_df, name):
    pa = team_profiles[team_a]
    pb = team_profiles[team_b]

    past = feat_df[
        ((feat_df["home_team"] == team_a) & (feat_df["away_team"] == team_b)) |
        ((feat_df["home_team"] == team_b) & (feat_df["away_team"] == team_a))
    ]
    if len(past) > 0:
        wins_a = len(past[
            ((past["home_team"] == team_a) & (past["result"] == "W")) |
            ((past["away_team"] == team_a) & (past["result"] == "L"))
        ])
        h2h = wins_a / len(past)
    else:
        h2h = 0.5

    row = {
        "elo_home":            pa["elo"],
        "elo_away":            pb["elo"],
        "elo_diff":            pa["elo"] - pb["elo"],
        "fifa_points_home":    pa["fifa_points"],
        "fifa_points_away":    pb["fifa_points"],
        "fifa_points_diff":    pa["fifa_points"] - pb["fifa_points"],
        "home_form_winrate":   pa["form_winrate"],
        "away_form_winrate":   pb["form_winrate"],
        "form_winrate_diff":   pa["form_winrate"] - pb["form_winrate"],
        "home_form_gf":        pa["form_gf"],
        "away_form_gf":        pb["form_gf"],
        "form_gf_diff":        pa["form_gf"] - pb["form_gf"],
        "home_form_ga":        pa["form_ga"],
        "away_form_ga":        pb["form_ga"],
        "form_ga_diff":        pa["form_ga"] - pb["form_ga"],
        "is_neutral":          1,
        "home_days_rest":      14,
        "away_days_rest":      14,
        "h2h_home_winrate":    h2h,
        "wc_experience_home":  pa["wc_experience"],
        "wc_experience_away":  pb["wc_experience"],
        "wc_experience_diff":  pa["wc_experience"] - pb["wc_experience"],
        "t_friendly":          0,
        "t_other_competitive": 0,
        "t_qualifier":         0,
        "t_world_cup":         1,
    }

    X = pd.DataFrame([row])[features]
    X_imp = imputer.transform(X)

    if name == "Logistic Regression":
        X_final = scaler.transform(X_imp)
    else:
        X_final = X_imp

    probs = model.predict_proba(X_final)[0]
    # 0=loss, 1=draw, 2=win for team_a
    return probs[2], probs[1], probs[0]


def build_match_probability_cache(ALL_TEAMS, team_profiles, model, imputer,
                                  scaler, features, feat_df, name):
    match_probs = {}
    total_matchups = len(ALL_TEAMS) * (len(ALL_TEAMS) - 1)

    print("\n" + "=" * 60)
    print("PRECOMPUTING MATCH PROBABILITIES")
    print("=" * 60)

    done = 0
    for team_a in ALL_TEAMS:
        for team_b in ALL_TEAMS:
            if team_a == team_b:
                continue
            match_probs[(team_a, team_b)] = predict_match(
                team_a, team_b, team_profiles, model, imputer, scaler,
                features, feat_df, name
            )
            done += 1

    print(f"Cached {done:,} ordered matchups out of {total_matchups:,}")
    return match_probs


MATCH_PROBS = build_match_probability_cache(
    ALL_TEAMS, team_profiles, model, imputer, scaler, features, feat_df, name
)


def print_group_stage_match_probabilities(GROUPS, match_probs):
    print("\n" + "=" * 60)
    print("GROUP STAGE — MATCH PROBABILITIES")
    print("=" * 60)

    for letter, teams in GROUPS.items():
        print(f"\nGroup {letter}:")
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                team_a = teams[i]
                team_b = teams[j]
                p_win, p_draw, p_loss = match_probs[(team_a, team_b)]
                print(
                    f"{team_a:<12} vs {team_b:<16} -> "
                    f"Win: {p_win * 100:>3.0f}%  "
                    f"Draw: {p_draw * 100:>3.0f}%  "
                    f"Loss: {p_loss * 100:>3.0f}%"
                )


print_group_stage_match_probabilities(GROUPS, MATCH_PROBS)


def simulate_match(team_a, team_b, match_probs, allow_draw=True):
    p_win, p_draw, p_loss = match_probs[(team_a, team_b)]

    if not allow_draw:
        # Redistribute draw probability 50/50 for knockout stage
        p_win_adj  = p_win  + p_draw * 0.5
        p_loss_adj = p_loss + p_draw * 0.5
        total = p_win_adj + p_loss_adj
        p_win_adj  /= total
        p_loss_adj /= total
        r = np.random.random()
        return team_a if r < p_win_adj else team_b

    r = np.random.random()
    if r < p_win:
        return team_a
    elif r < p_win + p_draw:
        return None
    else:
        return team_b


def simulate_group(group_teams, team_profiles, match_probs):
    standings = {t: {"points": 0, "gd": 0, "wins": 0} for t in group_teams}

    for i in range(len(group_teams)):
        for j in range(i+1, len(group_teams)):
            ta = group_teams[i]
            tb = group_teams[j]
            winner = simulate_match(ta, tb, match_probs, allow_draw=True)
            if winner == ta:
                standings[ta]["points"] += 3
                standings[ta]["wins"]   += 1
                standings[ta]["gd"]     += 1
                standings[tb]["gd"]     -= 1
            elif winner == tb:
                standings[tb]["points"] += 3
                standings[tb]["wins"]   += 1
                standings[tb]["gd"]     += 1
                standings[ta]["gd"]     -= 1
            else:
                standings[ta]["points"] += 1
                standings[tb]["points"] += 1

    # Sort: points → goal difference → elo tiebreaker
    sorted_teams = sorted(
        group_teams,
        key=lambda t: (
            standings[t]["points"],
            standings[t]["gd"],
            team_profiles[t]["elo"]
        ),
        reverse=True
    )

    return sorted_teams, standings


def simulate_tournament(GROUPS, team_profiles, match_probs):
    outcomes = {}

    group_results  = {}
    all_third      = []

    for letter, teams in GROUPS.items():
        sorted_teams, standings = simulate_group(teams, team_profiles, match_probs)
        group_results[letter] = sorted_teams

        third = sorted_teams[2]
        all_third.append((
            standings[third]["points"],
            standings[third]["gd"],
            team_profiles[third]["elo"],
            third
        ))

    # Best 8 third-place teams advance
    all_third.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    best_third = [t[3] for t in all_third[:8]]

    r32_teams = []
    for letter in "ABCDEFGHIJKL":
        r32_teams.append(group_results[letter][0])
    for letter in "ABCDEFGHIJKL":
        r32_teams.append(group_results[letter][1])
    r32_teams.extend(best_third)

    for team in ALL_TEAMS:
        if team not in r32_teams:
            outcomes[team] = "Group Stage"

    round_eliminated = []

    def play_knockout_round(teams):
        np.random.shuffle(teams)
        next_round = []
        eliminated = []
        for i in range(0, len(teams), 2):
            if i+1 < len(teams):
                winner = simulate_match(
                    teams[i], teams[i+1], match_probs, allow_draw=False
                )
                next_round.append(winner)
                loser = teams[i+1] if winner == teams[i] else teams[i]
                eliminated.append(loser)
            else:
                next_round.append(teams[i])
        round_eliminated.append(eliminated)
        return next_round

    r32     = play_knockout_round(r32_teams)
    r16     = play_knockout_round(r32)
    qf      = play_knockout_round(r16)
    sf      = play_knockout_round(qf)
    final   = play_knockout_round(sf)

    stage_names = [
        "Round of 32",
        "Round of 16",
        "Quarterfinal",
        "Semifinal",
        "Runner-up",
    ]
    for stage, eliminated_teams in zip(stage_names, round_eliminated):
        for team in eliminated_teams:
            outcomes[team] = stage

    outcomes[final[0]] = "Champion"

    return final[0], outcomes


print("\n" + "=" * 60)
print("RUNNING 10,000 MONTE CARLO SIMULATIONS")
print("=" * 60)

N_SIMS = 10000
champion_counts = {team: 0 for team in ALL_TEAMS}
simulation_champions = []
OUTCOME_ORDER = [
    "Group Stage",
    "Round of 32",
    "Round of 16",
    "Quarterfinal",
    "Semifinal",
    "Runner-up",
    "Champion",
]
outcome_counts = {
    team: {outcome: 0 for outcome in OUTCOME_ORDER}
    for team in ALL_TEAMS
}

for sim in range(N_SIMS):
    if sim % 1000 == 0:
        print(f"  Simulation {sim:,} / {N_SIMS:,}...")
    champion, outcomes = simulate_tournament(
        GROUPS, team_profiles, MATCH_PROBS
    )
    champion_counts[champion] += 1
    simulation_champions.append(champion)
    for team, outcome in outcomes.items():
        outcome_counts[team][outcome] += 1

print(f"\n{N_SIMS:,} simulations complete!")

print("\n" + "=" * 60)
print("WORLD CUP 2026 WIN PROBABILITIES")
print("=" * 60)

results_df = pd.DataFrame([
    {"team": team, "win_pct": count / N_SIMS * 100}
    for team, count in champion_counts.items()
]).sort_values("win_pct", ascending=False).reset_index(drop=True)

print(results_df[results_df["win_pct"] > 0].to_string(index=False))

top5_teams = results_df.head(5)["team"].tolist()
sim_numbers = np.arange(1, N_SIMS + 1)
convergence_df = pd.DataFrame({"simulation": sim_numbers})

fig, ax = plt.subplots(figsize=(11, 6))
PLOT_START_SIM = 100
plot_slice = sim_numbers >= PLOT_START_SIM
for team in top5_teams:
    cumulative_wins = np.cumsum([champion == team for champion in simulation_champions])
    cumulative_pct = cumulative_wins / sim_numbers * 100
    convergence_df[team] = cumulative_pct
    ax.plot(
        sim_numbers[plot_slice],
        cumulative_pct[plot_slice],
        linewidth=1.8,
        label=team
    )

ax.set_xlabel("Simulation number", fontsize=11)
ax.set_ylabel("Cumulative World Cup win probability (%)", fontsize=11)
ax.set_title(
    "FIFA World Cup 2026 — Monte Carlo Win Probability Convergence\n"
    f"Top 5 teams, shown from simulation {PLOT_START_SIM:,} of {N_SIMS:,}",
    fontsize=13,
)
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig("wc2026_simulation_convergence.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: wc2026_simulation_convergence.png")

top20 = results_df[results_df["win_pct"] > 0].head(20)

colors = []
for team in top20["team"]:
    elo = team_profiles[team]["elo"]
    if elo >= 1900:
        colors.append("#085041")
    elif elo >= 1700:
        colors.append("#0C447C")
    elif elo >= 1600:
        colors.append("#633806")
    else:
        colors.append("#aaaaaa")

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(
    top20["team"][::-1],
    top20["win_pct"][::-1],
    color=colors[::-1],
    edgecolor="white",
    linewidth=0.5
)

ax.set_xlabel("Probability of winning the World Cup (%)", fontsize=12)
ax.set_title("FIFA World Cup 2026 — Tournament Win Probabilities\n"
             f"Based on {N_SIMS:,} Monte Carlo simulations", fontsize=13)

for bar, val in zip(bars, top20["win_pct"][::-1]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=9)

legend_elements = [
    mpatches.Patch(color="#085041", label="Elo ≥ 1900 (Elite)"),
    mpatches.Patch(color="#0C447C", label="Elo 1700–1899 (Strong)"),
    mpatches.Patch(color="#633806", label="Elo 1600–1699 (Competitive)"),
    mpatches.Patch(color="#aaaaaa", label="Elo < 1600 (Outsider)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
ax.set_xlim(0, top20["win_pct"].max() * 1.15)
plt.tight_layout()
plt.savefig("wc2026_predictions.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: wc2026_predictions.png")

top5_teams = results_df.head(5)["team"].tolist()
outcome_rows = []
for team in top5_teams:
    row = {"team": team}
    for outcome in OUTCOME_ORDER:
        row[outcome] = outcome_counts[team][outcome] / N_SIMS * 100
    outcome_rows.append(row)

top5_outcomes_df = pd.DataFrame(outcome_rows)

print("\n" + "=" * 60)
print("TOP 5 TEAMS — ALL POSSIBLE OUTCOMES")
print("=" * 60)
print(top5_outcomes_df.to_string(index=False))

stage_colors = {
    "Group Stage": "#9CA3AF",
    "Round of 32": "#60A5FA",
    "Round of 16": "#34D399",
    "Quarterfinal": "#FBBF24",
    "Semifinal": "#F97316",
    "Runner-up": "#A78BFA",
    "Champion": "#DC2626",
}

fig, ax = plt.subplots(figsize=(11, 6))
left = np.zeros(len(top5_teams))
y_pos = np.arange(len(top5_teams))

for outcome in OUTCOME_ORDER:
    values = top5_outcomes_df[outcome].values
    bars = ax.barh(
        y_pos, values, left=left, color=stage_colors[outcome],
        edgecolor="white", linewidth=0.6, label=outcome,
    )
    for bar, value, left_edge in zip(bars, values, left):
        if value >= 4:
            ax.text(
                left_edge + value / 2, bar.get_y() + bar.get_height() / 2,
                f"{value:.0f}%", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold",
            )
    left += values

ax.set_yticks(y_pos)
ax.set_yticklabels(top5_teams)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Probability of final tournament outcome (%)", fontsize=11)
ax.set_title(
    "FIFA World Cup 2026 — Top 5 Team Outcome Distribution\n"
    f"Based on {N_SIMS:,} Monte Carlo simulations",
    fontsize=13,
)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=4, fontsize=9)
plt.tight_layout()
plt.savefig("wc2026_top5_outcomes.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: wc2026_top5_outcomes.png")

print("\n" + "=" * 60)
print("GROUP STAGE — TEAM RATINGS")
print("=" * 60)

for letter, teams in GROUPS.items():
    print(f"\n  Group {letter}:")
    for team in teams:
        elo = team_profiles[team]["elo"]
        wc_pct = champion_counts[team] / N_SIMS * 100
        print(f"    {team:<25} Elo: {elo:.0f}   Win prob: {wc_pct:.1f}%")

results_df.to_csv("wc2026_win_probabilities.csv", index=False)
convergence_df.to_csv("wc2026_simulation_convergence.csv", index=False)
top5_outcomes_df.to_csv("wc2026_top5_outcomes.csv", index=False)
print("\nSaved: wc2026_win_probabilities.csv")
print("Saved: wc2026_simulation_convergence.csv")
print("Saved: wc2026_top5_outcomes.csv")

print("\n" + "=" * 60)
print("PHASE 3 COMPLETE!")
print("=" * 60)
