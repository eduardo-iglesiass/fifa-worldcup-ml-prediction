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
from sklearn.metrics import accuracy_score, log_loss

RANDOM_STATE = 42
TRAIN_CUTOFF = "2022-01-01"

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

ELO_COLS      = ["elo_home", "elo_away", "elo_diff"]
FIFA_COLS     = ["fifa_points_home", "fifa_points_away", "fifa_points_diff"]
FORM_COLS     = ["home_form_winrate", "away_form_winrate", "form_winrate_diff",
                 "home_form_gf", "away_form_gf", "form_gf_diff",
                 "home_form_ga", "away_form_ga", "form_ga_diff"]
H2H_COLS      = ["h2h_home_winrate"]
CONTEXT_COLS  = ["is_neutral", "home_days_rest", "away_days_rest",
                 "t_friendly", "t_other_competitive", "t_qualifier", "t_world_cup"]

ABLATIONS = {
    "Full model":      [],
    "No Elo":          ELO_COLS,
    "No FIFA points":  FIFA_COLS,
    "No form":         FORM_COLS,
    "No H2H":          H2H_COLS,
    "No context":      CONTEXT_COLS,
}

df = pd.read_csv(os.path.join(BASE_DIR, "data", "features_phase1.csv"), parse_dates=["date"])
for col in ["t_friendly", "t_other_competitive", "t_qualifier", "t_world_cup"]:
    df[col] = df[col].astype(int)

train = df[df["date"] < TRAIN_CUTOFF]
test  = df[df["date"] >= TRAIN_CUTOFF]
y_train = train["result_num"]
y_test  = test["result_num"].values


def run_ablation(name, drop_cols):
    feats = [f for f in FEATURES if f not in drop_cols]
    if len(feats) == 0:
        return None

    X_train_raw = train[feats]
    X_test_raw  = test[feats]

    imp = SimpleImputer(strategy="mean")
    X_tr = imp.fit_transform(X_train_raw)
    X_te = imp.transform(X_test_raw)

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    model = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0,
                               random_state=RANDOM_STATE)
    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_prob),
        "n_features": len(feats),
    }


results = {}
for name, drop_cols in ABLATIONS.items():
    results[name] = run_ablation(name, drop_cols)

full_acc = results["Full model"]["accuracy"]

print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Feature Ablation Study — Logistic Regression on Test Set (2022 onwards)}")
print(r"\label{tab:ablation}")
print(r"\begin{tabular}{@{}lrrr@{}}")
print(r"\toprule")
print(r"Feature Set & Accuracy & Log-Loss & $\Delta$ Accuracy \\")
print(r"\midrule")

for name in ABLATIONS.keys():
    r = results[name]
    delta = r["accuracy"] - full_acc
    delta_str = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}"
    bold = name == "Full model"
    bo = r"\textbf{" if bold else ""
    bc = r"}"        if bold else ""
    print(f"{bo}{name}{bc} & "
          f"{bo}{r['accuracy']:.3f}{bc} & "
          f"{bo}{r['log_loss']:.4f}{bc} & "
          f"{bo}{delta_str}{bc} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

worst = min(
    [(n, r) for n, r in results.items() if n != "Full model"],
    key=lambda x: x[1]["accuracy"]
)
drop = full_acc - worst[1]["accuracy"]
worst_label = worst[0].replace("No ", "")
print(f"\nLargest accuracy drop from removing {worst_label}: {drop:.3f}")
