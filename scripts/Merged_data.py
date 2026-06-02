import pandas as pd
import numpy as np
from tqdm import tqdm

RESULTS_PATH  = "InternationalFootball_Results(1872-2026)/results.csv"
FIFA_PATH     = "Fifa_worldranking(1992-2024)/fifa_ranking-2024-06-20.csv"
NAMES_PATH    = "InternationalFootball_Results(1872-2026)/former_names.csv"

print("=" * 60)
print("PHASE 1 — LOADING DATA")
print("=" * 60)

results = pd.read_csv(RESULTS_PATH, parse_dates=["date"])
fifa    = pd.read_csv(FIFA_PATH,    parse_dates=["rank_date"])
names   = pd.read_csv(NAMES_PATH,  parse_dates=["start_date", "end_date"])

print(f"Match results:   {len(results):,} rows")
print(f"FIFA rankings:   {len(fifa):,} rows")
print(f"Former names:    {len(names):,} rows")

print("\n" + "=" * 60)
print("FIXING FORMER COUNTRY NAMES")
print("=" * 60)


def build_name_map(names_df):
    name_map = {}
    for _, row in names_df.iterrows():
        name_map[row["former"]] = row["current"]
    return name_map


name_map = build_name_map(names)


def standardise_name(team):
    return name_map.get(team, team)


results["home_team"] = results["home_team"].apply(standardise_name)
results["away_team"] = results["away_team"].apply(standardise_name)
fifa["country_full"] = fifa["country_full"].apply(standardise_name)

print(f"Standardised {len(name_map)} former country names")

print("\n" + "=" * 60)
print("CLEANING MATCH RESULTS")
print("=" * 60)

results = results.dropna(subset=["home_score", "away_score"])
results = results[results["date"].dt.year >= 1990].copy()
results = results.sort_values("date").reset_index(drop=True)


def get_result(row):
    if row["home_score"] > row["away_score"]:
        return "W"
    elif row["home_score"] == row["away_score"]:
        return "D"
    else:
        return "L"


results["result"] = results.apply(get_result, axis=1)
results["result_num"] = results["result"].map({"W": 2, "D": 1, "L": 0})


def tournament_category(t):
    t = str(t).lower()
    if "world cup" in t and "qualification" not in t:
        return "world_cup"
    elif "friendly" in t:
        return "friendly"
    elif "qualification" in t or "qualifier" in t:
        return "qualifier"
    else:
        return "other_competitive"


results["tournament_type"] = results["tournament"].apply(tournament_category)
results["is_neutral"] = (results["neutral"] == True).astype(int)

print(f"Matches after 1990:    {len(results):,}")
print(f"Date range: {results['date'].min().date()} → {results['date'].max().date()}")
print(f"Result distribution:\n{results['result'].value_counts().to_string()}")

print("\n" + "=" * 60)
print("COMPUTING ELO RATINGS")
print("=" * 60)

ELO_K     = 40
ELO_START = 1500

elo_ratings = {}
elo_home_before = []
elo_away_before = []

for _, row in tqdm(results.iterrows(), total=len(results), desc="Computing Elo"):
    home = row["home_team"]
    away = row["away_team"]

    r_home = elo_ratings.get(home, ELO_START)
    r_away = elo_ratings.get(away, ELO_START)

    # Record Elo BEFORE the match (no data leakage)
    elo_home_before.append(r_home)
    elo_away_before.append(r_away)

    e_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
    e_away = 1 - e_home

    if row["result"] == "W":
        s_home, s_away = 1.0, 0.0
    elif row["result"] == "D":
        s_home, s_away = 0.5, 0.5
    else:
        s_home, s_away = 0.0, 1.0

    elo_ratings[home] = r_home + ELO_K * (s_home - e_home)
    elo_ratings[away] = r_away + ELO_K * (s_away - e_away)

results["elo_home"]  = elo_home_before
results["elo_away"]  = elo_away_before
results["elo_diff"]  = results["elo_home"] - results["elo_away"]

print(f"Elo computed for {len(elo_ratings)} teams")
print(f"  Top 5 teams by current Elo:")
top5 = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)[:5]
for team, elo in top5:
    print(f"    {team}: {elo:.0f}")

print("\n" + "=" * 60)
print("MERGING FIFA RANKING POINTS")
print("=" * 60)

fifa = fifa.sort_values("rank_date").reset_index(drop=True)


def get_fifa_points_before(team, match_date, fifa_df):
    subset = fifa_df[
        (fifa_df["country_full"] == team) &
        (fifa_df["rank_date"] <= match_date)
    ]
    if len(subset) == 0:
        return np.nan
    return subset.iloc[-1]["total_points"]


print("Merging FIFA points (this takes ~1-2 minutes)...")

fifa_home_points = []
fifa_away_points = []
confederation_home = []
confederation_away = []

for _, row in tqdm(results.iterrows(), total=len(results), desc="FIFA merge"):
    home_pts = get_fifa_points_before(row["home_team"], row["date"], fifa)
    away_pts = get_fifa_points_before(row["away_team"], row["date"], fifa)
    fifa_home_points.append(home_pts)
    fifa_away_points.append(away_pts)

    home_conf = fifa[fifa["country_full"] == row["home_team"]]["confederation"].mode()
    away_conf = fifa[fifa["country_full"] == row["away_team"]]["confederation"].mode()
    confederation_home.append(home_conf.iloc[0] if len(home_conf) > 0 else "Unknown")
    confederation_away.append(away_conf.iloc[0] if len(away_conf) > 0 else "Unknown")

results["fifa_points_home"] = fifa_home_points
results["fifa_points_away"] = fifa_away_points
results["fifa_points_diff"] = results["fifa_points_home"] - results["fifa_points_away"]
results["confederation_home"] = confederation_home
results["confederation_away"] = confederation_away

covered = results["fifa_points_home"].notna().sum()
print(f"FIFA points merged for {covered:,} / {len(results):,} matches")

print("\n" + "=" * 60)
print("COMPUTING ROLLING FORM FEATURES")
print("=" * 60)

FORM_WINDOW = 10

rows = []
for _, r in results.iterrows():
    rows.append({
        "date": r["date"], "team": r["home_team"],
        "goals_for": r["home_score"], "goals_against": r["away_score"],
        "win": 1 if r["result"] == "W" else 0,
        "draw": 1 if r["result"] == "D" else 0,
        "match_idx": r.name, "side": "home"
    })
    rows.append({
        "date": r["date"], "team": r["away_team"],
        "goals_for": r["away_score"], "goals_against": r["home_score"],
        "win": 1 if r["result"] == "L" else 0,
        "draw": 1 if r["result"] == "D" else 0,
        "match_idx": r.name, "side": "away"
    })

long_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

for col in ["win", "goals_for", "goals_against"]:
    long_df[f"roll_{col}"] = long_df.groupby("team")[col].transform(
        lambda x: x.shift(1).rolling(FORM_WINDOW, min_periods=3).mean()
    )

home_form = long_df[long_df["side"] == "home"][
    ["match_idx", "roll_win", "roll_goals_for", "roll_goals_against"]
].rename(columns={
    "roll_win": "home_form_winrate",
    "roll_goals_for": "home_form_gf",
    "roll_goals_against": "home_form_ga"
})

away_form = long_df[long_df["side"] == "away"][
    ["match_idx", "roll_win", "roll_goals_for", "roll_goals_against"]
].rename(columns={
    "roll_win": "away_form_winrate",
    "roll_goals_for": "away_form_gf",
    "roll_goals_against": "away_form_ga"
})

results = results.merge(home_form, left_index=True, right_on="match_idx", how="left")
results = results.merge(away_form, left_index=True, right_on="match_idx",
                        how="left", suffixes=("", "_dup"))
results = results.drop(columns=["match_idx", "match_idx_dup"], errors="ignore")

results["form_winrate_diff"] = results["home_form_winrate"] - results["away_form_winrate"]
results["form_gf_diff"]      = results["home_form_gf"]      - results["away_form_gf"]
results["form_ga_diff"]      = results["home_form_ga"]      - results["away_form_ga"]

print("Rolling form features computed")

print("\n" + "=" * 60)
print("COMPUTING DAYS OF REST")
print("=" * 60)

long_df_sorted = long_df.sort_values(["team", "date"])
long_df_sorted["prev_match_date"] = long_df_sorted.groupby("team")["date"].shift(1)
long_df_sorted["days_rest"] = (
    long_df_sorted["date"] - long_df_sorted["prev_match_date"]
).dt.days

home_rest = long_df_sorted[long_df_sorted["side"] == "home"][
    ["match_idx", "days_rest"]
].rename(columns={"days_rest": "home_days_rest"})

away_rest = long_df_sorted[long_df_sorted["side"] == "away"][
    ["match_idx", "days_rest"]
].rename(columns={"days_rest": "away_days_rest"})

results = results.merge(home_rest, left_index=True, right_on="match_idx", how="left")
results = results.merge(away_rest, left_index=True, right_on="match_idx",
                        how="left", suffixes=("", "_dup"))
results = results.drop(columns=["match_idx", "match_idx_dup"], errors="ignore")

print("Days of rest computed")

print("\n" + "=" * 60)
print("COMPUTING HEAD TO HEAD RECORDS")
print("=" * 60)

h2h_winrates = []

for idx, row in tqdm(results.iterrows(), total=len(results), desc="H2H"):
    home = row["home_team"]
    away = row["away_team"]
    match_date = row["date"]

    past = results[
        (results["date"] < match_date) &
        (
            ((results["home_team"] == home) & (results["away_team"] == away)) |
            ((results["home_team"] == away) & (results["away_team"] == home))
        )
    ]

    if len(past) == 0:
        h2h_winrates.append(np.nan)
        continue

    home_wins = len(past[
        ((past["home_team"] == home) & (past["result"] == "W")) |
        ((past["away_team"] == home) & (past["result"] == "L"))
    ])
    h2h_winrates.append(home_wins / len(past))

results["h2h_home_winrate"] = h2h_winrates
print("Head to head records computed")

print("\n" + "=" * 60)
print("COMPUTING WORLD CUP EXPERIENCE")
print("=" * 60)

wc_matches = results[results["tournament_type"] == "world_cup"].copy()

wc_exp_home = []
wc_exp_away = []

for _, row in results.iterrows():
    home = row["home_team"]
    away = row["away_team"]
    date = row["date"]

    past_wc = wc_matches[wc_matches["date"] < date]

    home_wcs = past_wc[
        (past_wc["home_team"] == home) | (past_wc["away_team"] == home)
    ]["date"].dt.year.nunique()

    away_wcs = past_wc[
        (past_wc["home_team"] == away) | (past_wc["away_team"] == away)
    ]["date"].dt.year.nunique()

    wc_exp_home.append(home_wcs)
    wc_exp_away.append(away_wcs)

results["wc_experience_home"] = wc_exp_home
results["wc_experience_away"] = wc_exp_away
results["wc_experience_diff"] = results["wc_experience_home"] - results["wc_experience_away"]

print("World Cup experience computed")

print("\n" + "=" * 60)
print("BUILDING FINAL FEATURE TABLE")
print("=" * 60)

FEATURES = [
    "elo_home", "elo_away", "elo_diff",
    "fifa_points_home", "fifa_points_away", "fifa_points_diff",
    "home_form_winrate", "away_form_winrate", "form_winrate_diff",
    "home_form_gf", "away_form_gf", "form_gf_diff",
    "home_form_ga", "away_form_ga", "form_ga_diff",
    "is_neutral", "home_days_rest", "away_days_rest",
    "h2h_home_winrate",
    "wc_experience_home", "wc_experience_away", "wc_experience_diff",
]

TARGET = "result_num"  # 2=Win, 1=Draw, 0=Loss

META = ["date", "home_team", "away_team", "result",
        "tournament", "confederation_home", "confederation_away"]

results = pd.get_dummies(results, columns=["tournament_type"], prefix="t")
t_cols = [c for c in results.columns if c.startswith("t_")]
FEATURES += t_cols

model_df = results[FEATURES + [TARGET] + META].copy()
model_df = model_df.dropna(subset=["elo_home", "elo_away"])

print(f"Final dataset: {len(model_df):,} rows × {len(FEATURES)} features")
print(f"  Date range: {model_df['date'].min().date()} → {model_df['date'].max().date()}")
print(f"\n  Result distribution:")
print(f"  {model_df['result'].value_counts().to_dict()}")
print(f"\n  Features: {FEATURES}")

model_df.to_csv("features_phase1.csv", index=False)
print("\nSaved: features_phase1.csv")

elo_df = pd.DataFrame(
    list(elo_ratings.items()), columns=["team", "elo"]
).sort_values("elo", ascending=False).reset_index(drop=True)
elo_df.to_csv("current_elo_ratings.csv", index=False)
print("Saved: current_elo_ratings.csv")

print("\n" + "=" * 60)
print("PHASE 1 COMPLETE — ready for Phase 2!")
print("=" * 60)
