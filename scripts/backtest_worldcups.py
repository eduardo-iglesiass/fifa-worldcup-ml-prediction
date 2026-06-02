import os
import pandas as pd
import numpy as np
import pickle
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42
N_SIMS = 10000

FEATURES = [
    "elo_home", "elo_away", "elo_diff",
    "fifa_points_home", "fifa_points_away", "fifa_points_diff",
    "home_form_winrate", "away_form_winrate", "form_winrate_diff",
    "home_form_gf", "away_form_gf", "form_gf_diff",
    "home_form_ga", "away_form_ga", "form_ga_diff",
    "is_neutral", "home_days_rest", "away_days_rest",
    "h2h_home_winrate",
    "wc_experience_home", "wc_experience_away", "wc_experience_diff",
    "t_friendly", "t_other_competitive", "t_qualifier", "t_world_cup",
]

WC2018 = {
    "cutoff": "2018-06-14",
    "groups": {
        "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
        "B": ["Portugal", "Spain", "Morocco", "Iran"],
        "C": ["France", "Australia", "Peru", "Denmark"],
        "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
        "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
        "F": ["Germany", "Mexico", "Sweden", "South Korea"],
        "G": ["Belgium", "Panama", "Tunisia", "England"],
        "H": ["Poland", "Senegal", "Colombia", "Japan"],
    },
    "winner": "France",
    "runner_up": "Croatia",
    "year": 2018,
}

WC2022 = {
    "cutoff": "2022-11-20",
    "groups": {
        "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
        "B": ["England", "Iran", "United States", "Wales"],
        "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
        "D": ["France", "Australia", "Denmark", "Tunisia"],
        "E": ["Spain", "Costa Rica", "Germany", "Japan"],
        "F": ["Belgium", "Canada", "Morocco", "Croatia"],
        "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
        "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
    },
    "winner": "Argentina",
    "runner_up": "France",
    "year": 2022,
}

feat_df = pd.read_csv(os.path.join(BASE_DIR, "data", "features_phase1.csv"), parse_dates=["date"])
for col in ["t_friendly", "t_other_competitive", "t_qualifier", "t_world_cup"]:
    if col in feat_df.columns:
        feat_df[col] = feat_df[col].astype(int)


def build_elo_ratings(feat_df, cutoff, k=40, default=1500):
    hist = feat_df[feat_df["date"] < cutoff].sort_values("date")
    elo = {}
    for _, row in hist.iterrows():
        h, a = row["home_team"], row["away_team"]
        eh = elo.get(h, default)
        ea = elo.get(a, default)
        exp_h = 1 / (1 + 10 ** ((ea - eh) / 400))
        result = row["result_num"]
        s_h = 1.0 if result == 2 else (0.5 if result == 1 else 0.0)
        elo[h] = eh + k * (s_h - exp_h)
        elo[a] = ea + k * ((1 - s_h) - (1 - exp_h))
    return elo


def train_model_before(feat_df, cutoff):
    train = feat_df[feat_df["date"] < "2018-01-01"].copy()
    if cutoff > "2022-01-01":
        train = feat_df[feat_df["date"] < "2022-01-01"].copy()
    X = train[FEATURES]
    y = train["result_num"]
    imp = SimpleImputer(strategy="mean")
    X_imp = imp.fit_transform(X)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_imp)
    model = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0,
                               random_state=RANDOM_STATE)
    model.fit(X_sc, y)
    return model, imp, scaler


def build_team_profile(team, feat_df, elo_map, cutoff):
    hist = feat_df[feat_df["date"] < cutoff]
    home_m = hist[hist["home_team"] == team].sort_values("date")
    away_m = hist[hist["away_team"] == team].sort_values("date")
    profile = {"elo": elo_map.get(team, 1500)}

    # FIFA points
    fifa = None
    if len(home_m) > 0 and not pd.isna(home_m.iloc[-1]["fifa_points_home"]):
        fifa = home_m.iloc[-1]["fifa_points_home"]
    elif len(away_m) > 0 and not pd.isna(away_m.iloc[-1]["fifa_points_away"]):
        fifa = away_m.iloc[-1]["fifa_points_away"]
    profile["fifa_points"] = fifa if fifa is not None else hist["fifa_points_home"].mean()

    # Form
    if len(home_m) > 0:
        last = home_m.iloc[-1]
        profile["form_winrate"] = last["home_form_winrate"]
        profile["form_gf"]      = last["home_form_gf"]
        profile["form_ga"]      = last["home_form_ga"]
    elif len(away_m) > 0:
        last = away_m.iloc[-1]
        profile["form_winrate"] = last["away_form_winrate"]
        profile["form_gf"]      = last["away_form_gf"]
        profile["form_ga"]      = last["away_form_ga"]
    else:
        profile["form_winrate"] = hist["home_form_winrate"].mean()
        profile["form_gf"]      = hist["home_form_gf"].mean()
        profile["form_ga"]      = hist["home_form_ga"].mean()

    # WC experience
    wc = hist[hist["tournament"] == "FIFA World Cup"]
    wc_yrs = pd.concat([
        wc[wc["home_team"] == team]["date"],
        wc[wc["away_team"] == team]["date"]
    ]).dt.year.nunique()
    profile["wc_experience"] = wc_yrs
    return profile


def predict_match(team_a, team_b, profiles, model, imp, scaler, feat_df, cutoff):
    pa, pb = profiles[team_a], profiles[team_b]
    hist = feat_df[feat_df["date"] < cutoff]
    past = hist[
        ((hist["home_team"] == team_a) & (hist["away_team"] == team_b)) |
        ((hist["home_team"] == team_b) & (hist["away_team"] == team_a))
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
        "elo_home": pa["elo"], "elo_away": pb["elo"],
        "elo_diff": pa["elo"] - pb["elo"],
        "fifa_points_home": pa["fifa_points"], "fifa_points_away": pb["fifa_points"],
        "fifa_points_diff": pa["fifa_points"] - pb["fifa_points"],
        "home_form_winrate": pa["form_winrate"], "away_form_winrate": pb["form_winrate"],
        "form_winrate_diff": pa["form_winrate"] - pb["form_winrate"],
        "home_form_gf": pa["form_gf"], "away_form_gf": pb["form_gf"],
        "form_gf_diff": pa["form_gf"] - pb["form_gf"],
        "home_form_ga": pa["form_ga"], "away_form_ga": pb["form_ga"],
        "form_ga_diff": pa["form_ga"] - pb["form_ga"],
        "is_neutral": 1, "home_days_rest": 14, "away_days_rest": 14,
        "h2h_home_winrate": h2h,
        "wc_experience_home": pa["wc_experience"],
        "wc_experience_away": pb["wc_experience"],
        "wc_experience_diff": pa["wc_experience"] - pb["wc_experience"],
        "t_friendly": 0, "t_other_competitive": 0, "t_qualifier": 0, "t_world_cup": 1,
    }
    X = pd.DataFrame([row])[FEATURES]
    X_imp = imp.transform(X)
    X_sc  = scaler.transform(X_imp)
    probs = model.predict_proba(X_sc)[0]
    return probs[2], probs[1], probs[0]  # win, draw, loss


def sim_match(ta, tb, probs_cache, allow_draw=True):
    pw, pd_, pl = probs_cache[(ta, tb)]
    if not allow_draw:
        pw2 = pw + pd_ * 0.5
        pl2 = pl + pd_ * 0.5
        t = pw2 + pl2
        return ta if np.random.random() < pw2 / t else tb
    r = np.random.random()
    if r < pw:
        return ta
    elif r < pw + pd_:
        return None
    return tb


def sim_group(teams, probs_cache, profiles):
    st = {t: {"points": 0, "gd": 0} for t in teams}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            ta, tb = teams[i], teams[j]
            w = sim_match(ta, tb, probs_cache)
            if w == ta:
                st[ta]["points"] += 3; st[ta]["gd"] += 1; st[tb]["gd"] -= 1
            elif w == tb:
                st[tb]["points"] += 3; st[tb]["gd"] += 1; st[ta]["gd"] -= 1
            else:
                st[ta]["points"] += 1; st[tb]["points"] += 1
    return sorted(teams, key=lambda t: (st[t]["points"], st[t]["gd"],
                                        profiles[t]["elo"]), reverse=True), st


def sim_tournament_wc32(groups, probs_cache, profiles, all_teams):
    group_order = {}
    for letter, teams in groups.items():
        ranked, _ = sim_group(teams, probs_cache, profiles)
        group_order[letter] = ranked

    # Traditional WC R16 bracket: winner vs runner-up across adjacent groups
    letters = list(groups.keys())
    r16 = []
    pairs = [(0,1),(2,3),(4,5),(6,7)]
    for (i, j) in pairs:
        r16.append(group_order[letters[i]][0])
        r16.append(group_order[letters[j]][1])
        r16.append(group_order[letters[j]][0])
        r16.append(group_order[letters[i]][1])

    eliminated = set(all_teams) - set(r16)
    outcomes = {t: "Group Stage" for t in eliminated}

    def knockout(bracket, stage):
        next_r = []
        for k in range(0, len(bracket), 2):
            w = sim_match(bracket[k], bracket[k+1], probs_cache, allow_draw=False)
            loser = bracket[k+1] if w == bracket[k] else bracket[k]
            outcomes[loser] = stage
            next_r.append(w)
        return next_r

    qf     = knockout(r16, "Round of 16")
    sf     = knockout(qf,  "Quarterfinal")
    final2 = knockout(sf,  "Semifinal")
    champ  = knockout(final2, "Runner-up")
    outcomes[champ[0]] = "Champion"
    return champ[0], outcomes


def run_backtest(wc_cfg, feat_df):
    cutoff = wc_cfg["cutoff"]
    groups = wc_cfg["groups"]
    all_teams = [t for grp in groups.values() for t in grp]

    print(f"\nBuilding Elo ratings before {cutoff}...")
    elo_map = build_elo_ratings(feat_df, cutoff)

    print("Training model on pre-cutoff data...")
    model, imp, scaler = train_model_before(feat_df, cutoff)

    print("Building team profiles...")
    profiles = {t: build_team_profile(t, feat_df, elo_map, cutoff) for t in all_teams}

    print("Precomputing match probabilities...")
    probs_cache = {}
    for ta in all_teams:
        for tb in all_teams:
            if ta != tb:
                probs_cache[(ta, tb)] = predict_match(
                    ta, tb, profiles, model, imp, scaler, feat_df, cutoff)

    print(f"Running {N_SIMS:,} simulations...")
    counts = {t: 0 for t in all_teams}
    for _ in range(N_SIMS):
        champ, _ = sim_tournament_wc32(groups, probs_cache, profiles, all_teams)
        counts[champ] += 1

    results = sorted([(t, counts[t] / N_SIMS * 100) for t in all_teams],
                     key=lambda x: x[1], reverse=True)
    return results


def print_latex_table(results, wc_cfg, top_n=10):
    year      = wc_cfg["year"]
    winner    = wc_cfg["winner"]
    runner_up = wc_cfg["runner_up"]
    res_dict  = dict(results)
    top10     = results[:top_n]

    print(f"\n% World Cup {year} Backtest")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Monte Carlo Backtest: " + f"{year}" + r" World Cup — Top 10 Predicted Champions}")
    print(r"\label{tab:backtest" + str(year) + r"}")
    print(r"\begin{tabular}{@{}clr@{}}")
    print(r"\toprule")
    print(r"Rank & Team & Win Probability (\%) \\")
    print(r"\midrule")
    for rank, (team, pct) in enumerate(top10, 1):
        marker = r" $\star$" if team == winner else (r" $\dagger$" if team == runner_up else "")
        print(f"{rank} & {team}{marker} & {pct:.1f} \\\\")
    print(r"\midrule")
    winner_rank = next((i+1 for i, (t, _) in enumerate(results) if t == winner), "N/A")
    runner_rank = next((i+1 for i, (t, _) in enumerate(results) if t == runner_up), "N/A")
    winner_pct  = res_dict.get(winner, 0)
    runner_pct  = res_dict.get(runner_up, 0)
    in_top3     = winner_rank <= 3
    print(r"\multicolumn{3}{@{}l}{\textit{Actual champion: " +
          f"{winner} (rank {winner_rank}, {winner_pct:.1f}\\%)" + r"}} \\")
    print(r"\multicolumn{3}{@{}l}{\textit{Actual runner-up: " +
          f"{runner_up} (rank {runner_rank}, {runner_pct:.1f}\\%)" + r"}} \\")
    top3_str = "Yes" if in_top3 else "No"
    print(r"\multicolumn{3}{@{}l}{\textit{Champion in model top-3: " + top3_str + r"}} \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print(r"{\small $\star$ = actual champion \quad $\dagger$ = actual runner-up}")


np.random.seed(RANDOM_STATE)

print("=" * 60)
print("BACKTEST: 2018 World Cup (Russia)")
print("=" * 60)
res2018 = run_backtest(WC2018, feat_df)
print_latex_table(res2018, WC2018)

print("\n" + "=" * 60)
print("BACKTEST: 2022 World Cup (Qatar)")
print("=" * 60)
res2022 = run_backtest(WC2022, feat_df)
print_latex_table(res2022, WC2022)
