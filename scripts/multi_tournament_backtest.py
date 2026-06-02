import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SIMS = 10_000

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

WC_CONFIGS = [
    {
        "year": 1998, "cutoff": "1998-06-10",
        "winner": "France", "runner_up": "Brazil",
        "groups": {
            "A": ["Brazil", "Norway", "Morocco", "Scotland"],
            "B": ["Italy", "Chile", "Cameroon", "Austria"],
            "C": ["France", "Saudi Arabia", "Denmark", "South Africa"],
            "D": ["Nigeria", "Paraguay", "Spain", "Bulgaria"],
            "E": ["Netherlands", "South Korea", "Mexico", "Belgium"],
            "F": ["Germany", "Yugoslavia", "Iran", "United States"],
            "G": ["Romania", "Colombia", "England", "Tunisia"],
            "H": ["Argentina", "Croatia", "Japan", "Jamaica"],
        },
    },
    {
        "year": 2002, "cutoff": "2002-05-31",
        "winner": "Brazil", "runner_up": "Germany",
        "groups": {
            "A": ["Denmark", "Senegal", "Uruguay", "France"],
            "B": ["Spain", "Paraguay", "South Africa", "Slovenia"],
            "C": ["Brazil", "Turkey", "Costa Rica", "China"],
            "D": ["South Korea", "United States", "Portugal", "Poland"],
            "E": ["Germany", "Republic of Ireland", "Cameroon", "Saudi Arabia"],
            "F": ["England", "Argentina", "Sweden", "Nigeria"],
            "G": ["Mexico", "Croatia", "Italy", "Ecuador"],
            "H": ["Japan", "Belgium", "Russia", "Tunisia"],
        },
    },
    {
        "year": 2006, "cutoff": "2006-06-09",
        "winner": "Italy", "runner_up": "France",
        "groups": {
            "A": ["Germany", "Costa Rica", "Poland", "Ecuador"],
            "B": ["England", "Paraguay", "Trinidad and Tobago", "Sweden"],
            "C": ["Argentina", "Serbia and Montenegro", "Netherlands", "Ivory Coast"],
            "D": ["Portugal", "Angola", "Iran", "Mexico"],
            "E": ["Italy", "Ghana", "United States", "Czech Republic"],
            "F": ["Brazil", "Croatia", "Australia", "Japan"],
            "G": ["Switzerland", "Togo", "France", "South Korea"],
            "H": ["Spain", "Ukraine", "Tunisia", "Saudi Arabia"],
        },
    },
    {
        "year": 2010, "cutoff": "2010-06-11",
        "winner": "Spain", "runner_up": "Netherlands",
        "groups": {
            "A": ["South Africa", "Mexico", "Uruguay", "France"],
            "B": ["Argentina", "Nigeria", "South Korea", "Greece"],
            "C": ["England", "United States", "Algeria", "Slovenia"],
            "D": ["Germany", "Australia", "Serbia", "Ghana"],
            "E": ["Netherlands", "Denmark", "Japan", "Cameroon"],
            "F": ["Italy", "Paraguay", "New Zealand", "Slovakia"],
            "G": ["Brazil", "North Korea", "Portugal", "Ivory Coast"],
            "H": ["Spain", "Switzerland", "Honduras", "Chile"],
        },
    },
    {
        "year": 2014, "cutoff": "2014-06-12",
        "winner": "Germany", "runner_up": "Argentina",
        "groups": {
            "A": ["Brazil", "Croatia", "Mexico", "Cameroon"],
            "B": ["Spain", "Netherlands", "Chile", "Australia"],
            "C": ["Colombia", "Greece", "Ivory Coast", "Japan"],
            "D": ["Uruguay", "Costa Rica", "England", "Italy"],
            "E": ["France", "Ecuador", "Switzerland", "Honduras"],
            "F": ["Argentina", "Bosnia-Herzegovina", "Iran", "Nigeria"],
            "G": ["Germany", "Portugal", "Ghana", "United States"],
            "H": ["Belgium", "South Korea", "Algeria", "Russia"],
        },
    },
    {
        "year": 2018, "cutoff": "2018-06-14",
        "winner": "France", "runner_up": "Croatia",
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
    },
    {
        "year": 2022, "cutoff": "2022-11-20",
        "winner": "Argentina", "runner_up": "France",
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
    },
]

feat_df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "features_phase1.csv"),
    parse_dates=["date"],
)
for col in ["t_friendly", "t_other_competitive", "t_qualifier", "t_world_cup"]:
    if col in feat_df.columns:
        feat_df[col] = feat_df[col].astype(int)


def build_elo(df, cutoff, k=40, default=1500):
    elo = {}
    for _, row in df[df["date"] < cutoff].sort_values("date").iterrows():
        h, a = row["home_team"], row["away_team"]
        eh, ea = elo.get(h, default), elo.get(a, default)
        exp_h = 1.0 / (1.0 + 10.0 ** ((ea - eh) / 400.0))
        s_h = 1.0 if row["result_num"] == 2 else (0.5 if row["result_num"] == 1 else 0.0)
        elo[h] = eh + k * (s_h - exp_h)
        elo[a] = ea + k * ((1.0 - s_h) - (1.0 - exp_h))
    return elo


def train_lr(df, cutoff):
    tr = df[df["date"] < cutoff].copy()
    imp = SimpleImputer(strategy="mean")
    sc = StandardScaler()
    X_imp = imp.fit_transform(tr[FEATURES])
    X_sc = sc.fit_transform(X_imp)
    mdl = LogisticRegression(
        solver="lbfgs", max_iter=1000, C=1.0, random_state=RANDOM_STATE
    )
    mdl.fit(X_sc, tr["result_num"])
    return mdl, imp, sc


def build_profile(team, df, elo_map, cutoff):
    hist = df[df["date"] < cutoff]
    hm = hist[hist["home_team"] == team].sort_values("date")
    am = hist[hist["away_team"] == team].sort_values("date")

    prof = {"elo": elo_map.get(team, 1500)}

    if len(hm) > 0 and not pd.isna(hm.iloc[-1]["fifa_points_home"]):
        prof["fifa"] = hm.iloc[-1]["fifa_points_home"]
    elif len(am) > 0 and not pd.isna(am.iloc[-1]["fifa_points_away"]):
        prof["fifa"] = am.iloc[-1]["fifa_points_away"]
    else:
        prof["fifa"] = hist["fifa_points_home"].mean()

    if len(hm) > 0:
        last = hm.iloc[-1]
        prof["wr"] = last["home_form_winrate"]
        prof["gf"] = last["home_form_gf"]
        prof["ga"] = last["home_form_ga"]
    elif len(am) > 0:
        last = am.iloc[-1]
        prof["wr"] = last["away_form_winrate"]
        prof["gf"] = last["away_form_gf"]
        prof["ga"] = last["away_form_ga"]
    else:
        prof["wr"] = hist["home_form_winrate"].mean()
        prof["gf"] = hist["home_form_gf"].mean()
        prof["ga"] = hist["home_form_ga"].mean()

    wc = hist[hist["tournament"] == "FIFA World Cup"]
    wc_yrs = pd.concat([
        wc[wc["home_team"] == team]["date"],
        wc[wc["away_team"] == team]["date"],
    ]).dt.year.nunique()
    prof["wc_exp"] = wc_yrs
    return prof


def get_probs(ta, tb, profs, mdl, imp, sc, df, cutoff):
    pa, pb = profs[ta], profs[tb]
    hist = df[df["date"] < cutoff]
    past = hist[
        ((hist["home_team"] == ta) & (hist["away_team"] == tb)) |
        ((hist["home_team"] == tb) & (hist["away_team"] == ta))
    ]
    if len(past) > 0:
        wins_a = (
            ((past["home_team"] == ta) & (past["result"] == "W")) |
            ((past["away_team"] == ta) & (past["result"] == "L"))
        ).sum()
        h2h = wins_a / len(past)
    else:
        h2h = 0.5

    row = {
        "elo_home": pa["elo"],        "elo_away": pb["elo"],
        "elo_diff": pa["elo"] - pb["elo"],
        "fifa_points_home": pa["fifa"], "fifa_points_away": pb["fifa"],
        "fifa_points_diff": pa["fifa"] - pb["fifa"],
        "home_form_winrate": pa["wr"], "away_form_winrate": pb["wr"],
        "form_winrate_diff": pa["wr"] - pb["wr"],
        "home_form_gf": pa["gf"],  "away_form_gf": pb["gf"],
        "form_gf_diff": pa["gf"] - pb["gf"],
        "home_form_ga": pa["ga"],  "away_form_ga": pb["ga"],
        "form_ga_diff": pa["ga"] - pb["ga"],
        "is_neutral": 1, "home_days_rest": 14, "away_days_rest": 14,
        "h2h_home_winrate": h2h,
        "wc_experience_home": pa["wc_exp"], "wc_experience_away": pb["wc_exp"],
        "wc_experience_diff": pa["wc_exp"] - pb["wc_exp"],
        "t_friendly": 0, "t_other_competitive": 0, "t_qualifier": 0, "t_world_cup": 1,
    }
    X = pd.DataFrame([row])[FEATURES]
    p = mdl.predict_proba(sc.transform(imp.transform(X)))[0]
    return p[2], p[1], p[0]


def _sim_match(ta, tb, cache, allow_draw=True):
    pw, pd_, pl = cache[(ta, tb)]
    if not allow_draw:
        pw2 = pw + pd_ * 0.5
        return ta if np.random.random() < pw2 / (pw2 + pl + pd_ * 0.5) else tb
    r = np.random.random()
    return ta if r < pw else (None if r < pw + pd_ else tb)


def _sim_group(teams, cache, profs):
    st = {t: [0, 0] for t in teams}  # [points, gd]
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            w = _sim_match(teams[i], teams[j], cache)
            if w == teams[i]:
                st[teams[i]][0] += 3; st[teams[i]][1] += 1; st[teams[j]][1] -= 1
            elif w == teams[j]:
                st[teams[j]][0] += 3; st[teams[j]][1] += 1; st[teams[i]][1] -= 1
            else:
                st[teams[i]][0] += 1; st[teams[j]][0] += 1
    return sorted(teams,
                  key=lambda t: (st[t][0], st[t][1], profs[t]["elo"]),
                  reverse=True)


def _knockout(bracket, cache):
    out = []
    for k in range(0, len(bracket), 2):
        out.append(_sim_match(bracket[k], bracket[k + 1], cache, allow_draw=False))
    return out


def sim_wc32(groups, cache, profs):
    letters = list(groups.keys())
    gr = {L: _sim_group(ts, cache, profs) for L, ts in groups.items()}
    r16 = []
    for i, j in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        r16 += [gr[letters[i]][0], gr[letters[j]][1],
                gr[letters[j]][0], gr[letters[i]][1]]
    return _knockout(_knockout(_knockout(_knockout(r16, cache), cache), cache), cache)[0]


def run_tournament(cfg):
    year    = cfg["year"]
    cutoff  = cfg["cutoff"]
    groups  = cfg["groups"]
    teams   = [t for g in groups.values() for t in g]

    print(f"  [{year}] elo...", end=" ", flush=True)
    elo_map = build_elo(feat_df, cutoff)

    print("model...", end=" ", flush=True)
    mdl, imp, sc = train_lr(feat_df, cutoff)

    print("profiles...", end=" ", flush=True)
    profs = {t: build_profile(t, feat_df, elo_map, cutoff) for t in teams}

    print("cache...", end=" ", flush=True)
    cache = {
        (ta, tb): get_probs(ta, tb, profs, mdl, imp, sc, feat_df, cutoff)
        for ta in teams for tb in teams if ta != tb
    }

    print(f"sims...", end=" ", flush=True)
    counts = {t: 0 for t in teams}
    for _ in range(N_SIMS):
        counts[sim_wc32(groups, cache, profs)] += 1

    ranked = sorted(teams, key=lambda t: counts[t], reverse=True)
    prob   = {t: counts[t] / N_SIMS * 100.0 for t in teams}
    rank   = {t: i + 1 for i, t in enumerate(ranked)}
    print("done.")

    return {
        "year":        year,
        "winner":      cfg["winner"],
        "runner_up":   cfg["runner_up"],
        "winner_rank": rank.get(cfg["winner"], 999),
        "winner_prob": prob.get(cfg["winner"], 0.0),
        "runner_prob": prob.get(cfg["runner_up"], 0.0),
        "in_top3":     rank.get(cfg["winner"], 999) <= 3,
        "all_probs":   prob,
    }


np.random.seed(RANDOM_STATE)
print("=" * 60)
print("MULTI-TOURNAMENT BACKTEST: 1998-2022")
print("=" * 60)

all_results = []
for cfg in WC_CONFIGS:
    all_results.append(run_tournament(cfg))

print()
print(r"\begin{table}[ht]")
print(r"\centering")
print(r"\caption{Multi-Tournament Backtest: 1998--2022 World Cups (pre-tournament data only, $N=10{,}000$ simulations each)}")
print(r"\label{tab:multi_backtest}")
print(r"\begin{tabular}{@{}lllrrrr@{}}")
print(r"\toprule")
print(r"Year & Actual Winner & Runner-up & "
      r"\shortstack{Winner\\Predicted\\Rank} & "
      r"\shortstack{Winner\\Prob (\%)} & "
      r"\shortstack{Runner-up\\Prob (\%)} & "
      r"\shortstack{Winner\\in Top 3?} \\")
print(r"\midrule")

for r in all_results:
    top3 = r"Yes $\checkmark$" if r["in_top3"] else r"No $\times$"
    print(f"{r['year']} & {r['winner']} & {r['runner_up']} & "
          f"{r['winner_rank']} & "
          f"{r['winner_prob']:.1f} & "
          f"{r['runner_prob']:.1f} & "
          f"{top3} \\\\")

n_correct = sum(1 for r in all_results if r["in_top3"])
print(r"\midrule")
print(r"\multicolumn{7}{@{}l}{\textit{Winner in top-3: " +
      f"{n_correct}/7 tournaments" + r"}} \\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

cal_probs, cal_won = [], []
for r in all_results:
    winner = r["winner"]
    for team, prob in r["all_probs"].items():
        cal_probs.append(prob)
        cal_won.append(1 if team == winner else 0)

cal_probs = np.array(cal_probs)
cal_won   = np.array(cal_won)

mask  = (cal_probs >= 10.0) & (cal_probs < 20.0)
n_in  = int(mask.sum())
n_won = int(cal_won[mask].sum())
frac  = n_won / n_in if n_in > 0 else 0.0

print()
print("Calibration — teams predicted at 10-20% win probability:")
print(f"  Bucket size: {n_in}  |  Actual wins: {n_won}  |  Observed rate: {frac:.1%}  (mid-point: 15%)")
