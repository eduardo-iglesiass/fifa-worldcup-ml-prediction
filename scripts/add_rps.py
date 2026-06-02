import os
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

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


def rps_score(y_true, y_prob):
    # Ranked Probability Score for 3-outcome predictions; lower is better
    n = len(y_true)
    rps_vals = np.zeros(n)
    for i in range(n):
        o = int(y_true[i])
        o_loss = 1.0 if o == 0 else 0.0
        o_draw = 1.0 if o == 1 else 0.0
        O1 = o_loss
        O2 = o_loss + o_draw
        p_loss = y_prob[i, 0]
        p_draw = y_prob[i, 1]
        P1 = p_loss
        P2 = p_loss + p_draw
        rps_vals[i] = 0.5 * ((P1 - O1) ** 2 + (P2 - O2) ** 2)
    return rps_vals.mean()


df = pd.read_csv(os.path.join(BASE_DIR, "data", "features_phase1.csv"), parse_dates=["date"])
for col in ["t_friendly", "t_other_competitive", "t_qualifier", "t_world_cup"]:
    df[col] = df[col].astype(int)

train = df[df["date"] < TRAIN_CUTOFF]
test  = df[df["date"] >= TRAIN_CUTOFF]

X_train = train[FEATURES]
y_train = train["result_num"]
X_test  = test[FEATURES]
y_test  = test["result_num"].values

imp = SimpleImputer(strategy="mean")
X_train_imp = imp.fit_transform(X_train)
X_test_imp  = imp.transform(X_test)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train_imp)
X_test_sc  = sc.transform(X_test_imp)

with open(os.path.join(BASE_DIR, "models", "all_models.pkl"), "rb") as f:
    saved = pickle.load(f)

model_results = {}

for name, model in saved.items():
    if name == "Logistic Regression":
        y_prob = model.predict_proba(X_test_sc)
        y_pred = model.predict(X_test_sc)
    else:
        y_prob = model.predict_proba(X_test_imp)
        y_pred = model.predict(X_test_imp)

    model_results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_prob),
        "rps":      rps_score(y_test, y_prob),
    }

n_test = len(y_test)
naive_prob = np.zeros((n_test, 3))
naive_prob[:, 2] = 1.0
naive_pred = np.full(n_test, 2)

model_results["Naive (always Win)"] = {
    "accuracy": accuracy_score(y_test, naive_pred),
    "log_loss": log_loss(y_test, naive_prob),
    "rps":      rps_score(y_test, naive_prob),
}

row_order = ["Naive (always Win)", "Logistic Regression", "Random Forest", "XGBoost"]

print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Model Comparison on Test Set (2022 onwards): Accuracy, Log-Loss, and RPS}")
print(r"\label{tab:rps}")
print(r"\begin{tabular}{@{}lrrr@{}}")
print(r"\toprule")
print(r"Model & Accuracy & Log-Loss & RPS \\")
print(r"\midrule")
for name in row_order:
    r = model_results[name]
    bold_open  = r"\textbf{" if name != "Naive (always Win)" else ""
    bold_close = r"}"        if name != "Naive (always Win)" else ""
    print(f"{bold_open}{name}{bold_close} & "
          f"{r['accuracy']:.3f} & "
          f"{r['log_loss']:.4f} & "
          f"{r['rps']:.4f} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
