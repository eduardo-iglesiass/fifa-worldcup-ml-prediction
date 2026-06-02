import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss

RANDOM_STATE  = 42
TRAIN_CUTOFF  = "2022-01-01"
K_VALUES      = [10, 20, 30, 40, 50, 60]
TOP5_TEAMS    = ["Spain", "Argentina", "France", "England", "Brazil"]
DEFAULT_ELO   = 1500

BASE_FEATURES = [
    "fifa_points_home", "fifa_points_away", "fifa_points_diff",
    "home_form_winrate", "away_form_winrate", "form_winrate_diff",
    "home_form_gf", "away_form_gf", "form_gf_diff",
    "home_form_ga", "away_form_ga", "form_ga_diff",
    "is_neutral", "home_days_rest", "away_days_rest",
    "h2h_home_winrate",
    "wc_experience_home", "wc_experience_away", "wc_experience_diff",
    "t_friendly", "t_other_competitive", "t_qualifier", "t_world_cup",
]
ELO_FEATURES = ["elo_home", "elo_away", "elo_diff"]
ALL_FEATURES = ELO_FEATURES + BASE_FEATURES

results_raw = pd.read_csv(
    os.path.join(BASE_DIR, "InternationalFootball_Results(1872-2026)", "results.csv"),
    parse_dates=["date"]
)

feat_df = pd.read_csv(os.path.join(BASE_DIR, "data", "features_phase1.csv"), parse_dates=["date"])
for col in ["t_friendly", "t_other_competitive", "t_qualifier", "t_world_cup"]:
    feat_df[col] = feat_df[col].astype(int)


def build_elo_series(results_raw, feat_df, k):
    raw = results_raw.sort_values("date").copy()
    elo = {}
    elo_at_match = {}

    for _, row in raw.iterrows():
        h, a = row["home_team"], row["away_team"]
        hs = row["home_score"]
        as_ = row["away_score"]
        if pd.isna(hs) or pd.isna(as_):
            continue

        eh = elo.get(h, DEFAULT_ELO)
        ea = elo.get(a, DEFAULT_ELO)

        key = (str(row["date"])[:10], h, a)
        elo_at_match[key] = (eh, ea)

        if hs > as_:
            s_h = 1.0
        elif hs == as_:
            s_h = 0.5
        else:
            s_h = 0.0

        exp_h = 1 / (1 + 10 ** ((ea - eh) / 400))
        elo[h] = eh + k * (s_h - exp_h)
        elo[a] = ea + k * ((1 - s_h) - (1 - exp_h))

    elo_home_vals = []
    elo_away_vals = []

    for _, row in feat_df.iterrows():
        key = (str(row["date"])[:10], row["home_team"], row["away_team"])
        if key in elo_at_match:
            eh, ea = elo_at_match[key]
        else:
            eh = DEFAULT_ELO
            ea = DEFAULT_ELO
        elo_home_vals.append(eh)
        elo_away_vals.append(ea)

    out = feat_df.copy()
    out["elo_home"] = elo_home_vals
    out["elo_away"] = elo_away_vals
    out["elo_diff"] = out["elo_home"] - out["elo_away"]
    return out, elo


perf_rows  = []
rating_rows = []

print("Running Elo sensitivity analysis...")

for k in K_VALUES:
    print(f"  K = {k}...", end=" ", flush=True)
    enriched, final_elo = build_elo_series(results_raw, feat_df, k)

    train = enriched[enriched["date"] < TRAIN_CUTOFF]
    test  = enriched[enriched["date"] >= TRAIN_CUTOFF]

    X_train = train[ALL_FEATURES]
    y_train = train["result_num"]
    X_test  = test[ALL_FEATURES]
    y_test  = test["result_num"].values

    imp = SimpleImputer(strategy="mean")
    X_tr = imp.fit_transform(X_train)
    X_te = imp.transform(X_test)

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    model = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0,
                               random_state=RANDOM_STATE)
    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)

    acc = accuracy_score(y_test, y_pred)
    ll  = log_loss(y_test, y_prob)

    perf_rows.append({"K": k, "Accuracy": acc, "Log-Loss": ll})

    row = {"K": k}
    for team in TOP5_TEAMS:
        row[team] = final_elo.get(team, DEFAULT_ELO)
    rating_rows.append(row)

    print(f"Acc={acc:.3f}  LL={ll:.4f}")

print()
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Logistic Regression Performance vs Elo $K$ Parameter (test set: 2022 onwards)}")
print(r"\label{tab:elo_k_perf}")
print(r"\begin{tabular}{@{}rrr@{}}")
print(r"\toprule")
print(r"$K$ & Accuracy & Log-Loss \\")
print(r"\midrule")
for r in perf_rows:
    bold = r["K"] == 40
    bo, bc = (r"\textbf{", "}") if bold else ("", "")
    print(f"{bo}{r['K']}{bc} & "
          f"{bo}{r['Accuracy']:.3f}{bc} & "
          f"{bo}{r['Log-Loss']:.4f}{bc} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

print()
team_header = " & ".join(TOP5_TEAMS)
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Final Elo Ratings for Top-5 Teams Under Different $K$ Values}")
print(r"\label{tab:elo_k_ratings}")
col_spec = "r" + "r" * len(TOP5_TEAMS)
print(r"\begin{tabular}{@{}" + col_spec + r"@{}}")
print(r"\toprule")
print(f"$K$ & {team_header} \\\\")
print(r"\midrule")
for r in rating_rows:
    vals = " & ".join(f"{r[t]:.0f}" for t in TOP5_TEAMS)
    bold = r["K"] == 40
    bo, bc = (r"\textbf{", "}") if bold else ("", "")
    print(f"{bo}{r['K']}{bc} & {bo}{vals}{bc} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
