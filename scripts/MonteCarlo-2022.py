import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TOURNAMENT_START = pd.Timestamp("2022-11-20")
N_SIMS = 10000

GROUPS = {
    "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
    "B": ["England", "Iran", "United States", "Wales"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Spain", "Costa Rica", "Germany", "Japan"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
}

ALL_TEAMS = [team for teams in GROUPS.values() for team in teams]

R16_BRACKET = [
    ("A", 0, "B", 1),
    ("C", 0, "D", 1),
    ("E", 0, "F", 1),
    ("G", 0, "H", 1),
    ("B", 0, "A", 1),
    ("D", 0, "C", 1),
    ("F", 0, "E", 1),
    ("H", 0, "G", 1),
]

OUTCOME_ORDER = [
    "Group Stage",
    "Round of 16",
    "Quarterfinal",
    "Semifinal",
    "Runner-up",
    "Champion",
]

print("=" * 60)
print("WORLD CUP 2022 MONTE CARLO PREDICTIONS")
print("=" * 60)

with open("best_model.pkl", "rb") as f:
    package = pickle.load(f)

model = package["model"]
imputer = package["imputer"]
scaler = package["scaler"]
features = package["features"]
model_name = package["name"]

feat_df = pd.read_csv("features_phase1.csv", parse_dates=["date"])
pre_wc_df = feat_df[feat_df["date"] < TOURNAMENT_START].copy()

print(f"Loaded model: {model_name}")
print(f"Using matches before {TOURNAMENT_START.date()}")


def get_latest_team_row(team, df):
    home = df[df["home_team"] == team].sort_values("date")
    away = df[df["away_team"] == team].sort_values("date")

    candidates = []
    if len(home) > 0:
        candidates.append(("home", home.iloc[-1]))
    if len(away) > 0:
        candidates.append(("away", away.iloc[-1]))

    if not candidates:
        return None, None

    side, row = max(candidates, key=lambda item: item[1]["date"])
    return side, row


def get_team_profile(team, df):
    side, row = get_latest_team_row(team, df)
    profile = {}

    if row is None:
        profile["elo"] = 1500
        profile["fifa_points"] = df["fifa_points_home"].mean()
        profile["form_winrate"] = df["home_form_winrate"].mean()
        profile["form_gf"] = df["home_form_gf"].mean()
        profile["form_ga"] = df["home_form_ga"].mean()
    elif side == "home":
        profile["elo"] = row["elo_home"]
        profile["fifa_points"] = row["fifa_points_home"]
        profile["form_winrate"] = row["home_form_winrate"]
        profile["form_gf"] = row["home_form_gf"]
        profile["form_ga"] = row["home_form_ga"]
    else:
        profile["elo"] = row["elo_away"]
        profile["fifa_points"] = row["fifa_points_away"]
        profile["form_winrate"] = row["away_form_winrate"]
        profile["form_gf"] = row["away_form_gf"]
        profile["form_ga"] = row["away_form_ga"]

    for key in ["fifa_points", "form_winrate", "form_gf", "form_ga"]:
        if pd.isna(profile[key]):
            fallback_col = {
                "fifa_points": "fifa_points_home",
                "form_winrate": "home_form_winrate",
                "form_gf": "home_form_gf",
                "form_ga": "home_form_ga",
            }[key]
            profile[key] = df[fallback_col].mean()

    wc_matches = df[df["tournament"] == "FIFA World Cup"]
    home_wc = wc_matches[wc_matches["home_team"] == team]
    away_wc = wc_matches[wc_matches["away_team"] == team]
    profile["wc_experience"] = pd.concat([home_wc["date"], away_wc["date"]]).dt.year.nunique()

    return profile


team_profiles = {team: get_team_profile(team, pre_wc_df) for team in ALL_TEAMS}


def predict_match(team_a, team_b):
    pa = team_profiles[team_a]
    pb = team_profiles[team_b]

    past = pre_wc_df[
        ((pre_wc_df["home_team"] == team_a) & (pre_wc_df["away_team"] == team_b))
        | ((pre_wc_df["home_team"] == team_b) & (pre_wc_df["away_team"] == team_a))
    ]
    if len(past) > 0:
        wins_a = len(
            past[
                ((past["home_team"] == team_a) & (past["result"] == "W"))
                | ((past["away_team"] == team_a) & (past["result"] == "L"))
            ]
        )
        h2h = wins_a / len(past)
    else:
        h2h = 0.5

    row = {
        "elo_home": pa["elo"],
        "elo_away": pb["elo"],
        "elo_diff": pa["elo"] - pb["elo"],
        "fifa_points_home": pa["fifa_points"],
        "fifa_points_away": pb["fifa_points"],
        "fifa_points_diff": pa["fifa_points"] - pb["fifa_points"],
        "home_form_winrate": pa["form_winrate"],
        "away_form_winrate": pb["form_winrate"],
        "form_winrate_diff": pa["form_winrate"] - pb["form_winrate"],
        "home_form_gf": pa["form_gf"],
        "away_form_gf": pb["form_gf"],
        "form_gf_diff": pa["form_gf"] - pb["form_gf"],
        "home_form_ga": pa["form_ga"],
        "away_form_ga": pb["form_ga"],
        "form_ga_diff": pa["form_ga"] - pb["form_ga"],
        "is_neutral": 1,
        "home_days_rest": 14,
        "away_days_rest": 14,
        "h2h_home_winrate": h2h,
        "wc_experience_home": pa["wc_experience"],
        "wc_experience_away": pb["wc_experience"],
        "wc_experience_diff": pa["wc_experience"] - pb["wc_experience"],
        "t_friendly": 0,
        "t_other_competitive": 0,
        "t_qualifier": 0,
        "t_world_cup": 1,
    }

    X = pd.DataFrame([row])[features]
    X_imp = imputer.transform(X)
    X_final = scaler.transform(X_imp) if model_name == "Logistic Regression" else X_imp
    probs = model.predict_proba(X_final)[0]

    return probs[2], probs[1], probs[0]


def build_match_probability_cache():
    match_probs = {}
    for team_a in ALL_TEAMS:
        for team_b in ALL_TEAMS:
            if team_a != team_b:
                match_probs[(team_a, team_b)] = predict_match(team_a, team_b)
    return match_probs


MATCH_PROBS = build_match_probability_cache()


def get_actual_group_stage_result(team_a, team_b):
    match = feat_df[
        (feat_df["tournament"] == "FIFA World Cup")
        & (feat_df["date"] >= TOURNAMENT_START)
        & (feat_df["date"] <= pd.Timestamp("2022-12-02"))
        & (
            (
                (feat_df["home_team"] == team_a)
                & (feat_df["away_team"] == team_b)
            )
            | (
                (feat_df["home_team"] == team_b)
                & (feat_df["away_team"] == team_a)
            )
        )
    ]

    if len(match) == 0:
        return None

    row = match.iloc[0]
    result = row["result"]
    if row["home_team"] == team_a:
        return result
    if result == "W":
        return "L"
    if result == "L":
        return "W"
    return "D"


def predicted_result_from_probs(p_win, p_draw, p_loss):
    result_probs = {"W": p_win, "D": p_draw, "L": p_loss}
    return max(result_probs, key=result_probs.get)


group_stage_prediction_rows = []

print("\n" + "=" * 60)
print("GROUP STAGE MATCH PROBABILITIES")
print("=" * 60)
for letter, teams in GROUPS.items():
    print(f"\nGroup {letter}:")
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            team_a = teams[i]
            team_b = teams[j]
            p_win, p_draw, p_loss = MATCH_PROBS[(team_a, team_b)]
            predicted_result = predicted_result_from_probs(p_win, p_draw, p_loss)
            actual_result = get_actual_group_stage_result(team_a, team_b)
            correct = actual_result is not None and predicted_result == actual_result

            group_stage_prediction_rows.append(
                {
                    "group": letter,
                    "team_a": team_a,
                    "team_b": team_b,
                    "win_pct_team_a": p_win * 100,
                    "draw_pct": p_draw * 100,
                    "loss_pct_team_a": p_loss * 100,
                    "predicted_result_for_team_a": predicted_result,
                    "actual_result_for_team_a": actual_result,
                    "correct": correct,
                }
            )

            print(
                f"{team_a:<14} vs {team_b:<14} -> "
                f"Win: {p_win * 100:>3.0f}%  "
                f"Draw: {p_draw * 100:>3.0f}%  "
                f"Loss: {p_loss * 100:>3.0f}%  "
                f"Pred: {predicted_result}  Actual: {actual_result}"
            )

group_stage_predictions_df = pd.DataFrame(group_stage_prediction_rows)
known_matches = group_stage_predictions_df["actual_result_for_team_a"].notna()
correct_group_predictions = group_stage_predictions_df.loc[known_matches, "correct"].sum()
total_group_matches = known_matches.sum()

print("\n" + "=" * 60)
print("2022 GROUP STAGE PREDICTION ACCURACY")
print("=" * 60)
print(
    f"Correct group-stage match outcomes: "
    f"{correct_group_predictions} / {total_group_matches} "
    f"({correct_group_predictions / total_group_matches * 100:.1f}%)"
)


def simulate_match(team_a, team_b, allow_draw=True):
    p_win, p_draw, p_loss = MATCH_PROBS[(team_a, team_b)]

    if not allow_draw:
        p_win = p_win + p_draw * 0.5
        p_loss = p_loss + p_draw * 0.5
        total = p_win + p_loss
        p_win = p_win / total
        p_loss = p_loss / total

    r = np.random.random()
    if r < p_win:
        return team_a
    if allow_draw and r < p_win + p_draw:
        return None
    return team_b


def simulate_group(group_teams):
    standings = {team: {"points": 0, "gd": 0, "wins": 0} for team in group_teams}

    for i in range(len(group_teams)):
        for j in range(i + 1, len(group_teams)):
            team_a = group_teams[i]
            team_b = group_teams[j]
            winner = simulate_match(team_a, team_b, allow_draw=True)

            if winner == team_a:
                standings[team_a]["points"] += 3
                standings[team_a]["wins"] += 1
                standings[team_a]["gd"] += 1
                standings[team_b]["gd"] -= 1
            elif winner == team_b:
                standings[team_b]["points"] += 3
                standings[team_b]["wins"] += 1
                standings[team_b]["gd"] += 1
                standings[team_a]["gd"] -= 1
            else:
                standings[team_a]["points"] += 1
                standings[team_b]["points"] += 1

    sorted_teams = sorted(
        group_teams,
        key=lambda team: (
            standings[team]["points"],
            standings[team]["gd"],
            standings[team]["wins"],
            team_profiles[team]["elo"],
        ),
        reverse=True,
    )
    return sorted_teams, standings


def play_knockout_round(teams):
    winners = []
    losers = []
    for i in range(0, len(teams), 2):
        team_a = teams[i]
        team_b = teams[i + 1]
        winner = simulate_match(team_a, team_b, allow_draw=False)
        loser = team_b if winner == team_a else team_a
        winners.append(winner)
        losers.append(loser)
    return winners, losers


def simulate_tournament():
    outcomes = {}
    group_results = {}

    for letter, teams in GROUPS.items():
        sorted_teams, _ = simulate_group(teams)
        group_results[letter] = sorted_teams
        for eliminated in sorted_teams[2:]:
            outcomes[eliminated] = "Group Stage"

    r16_teams = []
    for group_a, pos_a, group_b, pos_b in R16_BRACKET:
        r16_teams.extend([group_results[group_a][pos_a], group_results[group_b][pos_b]])

    qf, r16_out = play_knockout_round(r16_teams)
    sf, qf_out = play_knockout_round(qf)
    final, sf_out = play_knockout_round(sf)
    champion_list, final_out = play_knockout_round(final)
    champion = champion_list[0]

    for team in r16_out:
        outcomes[team] = "Round of 16"
    for team in qf_out:
        outcomes[team] = "Quarterfinal"
    for team in sf_out:
        outcomes[team] = "Semifinal"
    for team in final_out:
        outcomes[team] = "Runner-up"
    outcomes[champion] = "Champion"

    return champion, outcomes


print("\n" + "=" * 60)
print(f"RUNNING {N_SIMS:,} MONTE CARLO SIMULATIONS")
print("=" * 60)

champion_counts = {team: 0 for team in ALL_TEAMS}
outcome_counts = {
    team: {outcome: 0 for outcome in OUTCOME_ORDER}
    for team in ALL_TEAMS
}
simulation_champions = []

for sim in range(N_SIMS):
    if sim % 1000 == 0:
        print(f"  Simulation {sim:,} / {N_SIMS:,}...")
    champion, outcomes = simulate_tournament()
    champion_counts[champion] += 1
    simulation_champions.append(champion)
    for team, outcome in outcomes.items():
        outcome_counts[team][outcome] += 1

results_df = pd.DataFrame(
    [
        {"team": team, "win_pct": count / N_SIMS * 100}
        for team, count in champion_counts.items()
    ]
).sort_values("win_pct", ascending=False).reset_index(drop=True)

print("\n" + "=" * 60)
print("WORLD CUP 2022 WIN PROBABILITIES")
print("=" * 60)
print(results_df[results_df["win_pct"] > 0].to_string(index=False))

top20 = results_df[results_df["win_pct"] > 0].head(20)
fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(top20["team"][::-1], top20["win_pct"][::-1], color="#2563EB")
ax.set_xlabel("Probability of winning the World Cup (%)")
ax.set_title(f"World Cup 2022 Win Probabilities ({N_SIMS:,} simulations)")
for bar, val in zip(bars, top20["win_pct"][::-1]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%")
ax.set_xlim(0, top20["win_pct"].max() * 1.15)
plt.tight_layout()
plt.savefig("wc2022_predictions.png", dpi=150, bbox_inches="tight")
plt.show()

top5_teams = results_df.head(5)["team"].tolist()
outcome_rows = []
for team in top5_teams:
    row = {"team": team}
    for outcome in OUTCOME_ORDER:
        row[outcome] = outcome_counts[team][outcome] / N_SIMS * 100
    outcome_rows.append(row)
top5_outcomes_df = pd.DataFrame(outcome_rows)

stage_colors = {
    "Group Stage": "#9CA3AF",
    "Round of 16": "#60A5FA",
    "Quarterfinal": "#34D399",
    "Semifinal": "#F97316",
    "Runner-up": "#A78BFA",
    "Champion": "#DC2626",
}

fig, ax = plt.subplots(figsize=(11, 6))
left = np.zeros(len(top5_teams))
y_pos = np.arange(len(top5_teams))
for outcome in OUTCOME_ORDER:
    values = top5_outcomes_df[outcome].values
    ax.barh(y_pos, values, left=left, color=stage_colors[outcome],
            edgecolor="white", linewidth=0.6, label=outcome)
    left += values
ax.set_yticks(y_pos)
ax.set_yticklabels(top5_teams)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Probability of final tournament outcome (%)")
ax.set_title(f"World Cup 2022 Top 5 Outcome Distribution ({N_SIMS:,} simulations)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3)
plt.tight_layout()
plt.savefig("wc2022_top5_outcomes.png", dpi=150, bbox_inches="tight")
plt.show()

sim_numbers = np.arange(1, N_SIMS + 1)
convergence_df = pd.DataFrame({"simulation": sim_numbers})
fig, ax = plt.subplots(figsize=(11, 6))
plot_start = 100
plot_slice = sim_numbers >= plot_start
for team in top5_teams:
    cumulative_wins = np.cumsum([champion == team for champion in simulation_champions])
    cumulative_pct = cumulative_wins / sim_numbers * 100
    convergence_df[team] = cumulative_pct
    ax.plot(sim_numbers[plot_slice], cumulative_pct[plot_slice], linewidth=1.8, label=team)
ax.set_xlabel("Simulation number")
ax.set_ylabel("Cumulative World Cup win probability (%)")
ax.set_title(f"World Cup 2022 Monte Carlo Convergence from simulation {plot_start}")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("wc2022_simulation_convergence.png", dpi=150, bbox_inches="tight")
plt.show()

results_df.to_csv("wc2022_win_probabilities.csv", index=False)
group_stage_predictions_df.to_csv("wc2022_group_stage_prediction_accuracy.csv", index=False)
top5_outcomes_df.to_csv("wc2022_top5_outcomes.csv", index=False)
convergence_df.to_csv("wc2022_simulation_convergence.csv", index=False)

print("\nSaved: wc2022_win_probabilities.csv")
print("Saved: wc2022_group_stage_prediction_accuracy.csv")
print("Saved: wc2022_top5_outcomes.csv")
print("Saved: wc2022_simulation_convergence.csv")
print("Saved: wc2022_predictions.png")
print("Saved: wc2022_top5_outcomes.png")
print("Saved: wc2022_simulation_convergence.png")
print("\nWORLD CUP 2022 COMPLETE")
