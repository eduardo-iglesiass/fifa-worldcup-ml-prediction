import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_YEARS   = list(range(2015, 2023))

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
    n = len(y_true)
    rps_vals = np.zeros(n)
    for i in range(n):
        o = int(y_true[i])
        O1 = 1.0 if o == 0 else 0.0
        O2 = (1.0 if o == 0 else 0.0) + (1.0 if o == 1 else 0.0)
        P1 = y_prob[i, 0]
        P2 = y_prob[i, 0] + y_prob[i, 1]
        rps_vals[i] = 0.5 * ((P1 - O1) ** 2 + (P2 - O2) ** 2)
    return rps_vals.mean()


df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "features_phase1.csv"),
    parse_dates=["date"],
)
for col in ["t_friendly", "t_other_competitive", "t_qualifier", "t_world_cup"]:
    if col in df.columns:
        df[col] = df[col].astype(int)


def make_models():
    return {
        "Logistic Regression": LogisticRegression(
            solver="lbfgs", max_iter=1000, C=1.0, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=20,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", random_state=RANDOM_STATE, verbosity=0
        ),
    }


print("=" * 60)
print("WALK-FORWARD CROSS-VALIDATION (2015-2022)")
print("=" * 60)

fold_metrics = {name: [] for name in ["Logistic Regression", "Random Forest", "XGBoost"]}

for test_year in TEST_YEARS:
    train_cutoff = f"{test_year}-01-01"
    test_start   = f"{test_year}-01-01"
    test_end     = f"{test_year + 1}-01-01"

    train = df[df["date"] < train_cutoff].copy()
    test  = df[(df["date"] >= test_start) & (df["date"] < test_end)].copy()

    if len(test) == 0:
        print(f"  [{test_year}] No test data — skipping.")
        continue

    print(f"  [{test_year}] train={len(train):,}  test={len(test):,}", end="  ", flush=True)

    X_train = train[FEATURES]
    y_train = train["result_num"].values
    X_test  = test[FEATURES]
    y_test  = test["result_num"].values

    imp = SimpleImputer(strategy="mean")
    X_train_imp = imp.fit_transform(X_train)
    X_test_imp  = imp.transform(X_test)

    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train_imp)
    X_test_sc  = sc.transform(X_test_imp)

    models = make_models()

    for name, mdl in models.items():
        if name == "Logistic Regression":
            mdl.fit(X_train_sc, y_train)
            y_prob = mdl.predict_proba(X_test_sc)
            y_pred = mdl.predict(X_test_sc)
        else:
            mdl.fit(X_train_imp, y_train)
            y_prob = mdl.predict_proba(X_test_imp)
            y_pred = mdl.predict(X_test_imp)

        fold_metrics[name].append({
            "accuracy": accuracy_score(y_test, y_pred),
            "log_loss": log_loss(y_test, y_prob),
            "rps":      rps_score(y_test, y_prob),
        })

    print("done.")

print()
summary = {}
for name, folds in fold_metrics.items():
    if not folds:
        continue
    accs = [f["accuracy"] for f in folds]
    lls  = [f["log_loss"] for f in folds]
    rpss = [f["rps"]      for f in folds]
    summary[name] = {
        "acc_mean": np.mean(accs), "acc_std": np.std(accs, ddof=1),
        "ll_mean":  np.mean(lls),  "ll_std":  np.std(lls,  ddof=1),
        "rps_mean": np.mean(rpss), "rps_std": np.std(rpss, ddof=1),
        "n_folds":  len(folds),
    }

n_folds = len(TEST_YEARS)
print(r"\begin{table}[ht]")
print(r"\centering")
print(r"\caption{Walk-Forward Cross-Validation Results (" +
      f"{TEST_YEARS[0]}--{TEST_YEARS[-1]}, $k={n_folds}$ annual folds" +
      r"). Values are mean $\pm$ std across folds.}")
print(r"\label{tab:wfcv}")
print(r"\begin{tabular}{@{}lllll@{}}")
print(r"\toprule")
print(r"Model & Folds & Accuracy & Log-Loss & RPS \\")
print(r"\midrule")

for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
    if name not in summary:
        continue
    s = summary[name]
    print(f"{name} & {s['n_folds']} & "
          f"${s['acc_mean']:.3f} \\pm {s['acc_std']:.3f}$ & "
          f"${s['ll_mean']:.4f} \\pm {s['ll_std']:.4f}$ & "
          f"${s['rps_mean']:.4f} \\pm {s['rps_std']:.4f}$ \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

print()
print("Per-fold log-loss (Logistic Regression):")
for i, (yr, fold) in enumerate(zip(TEST_YEARS, fold_metrics["Logistic Regression"])):
    print(f"  {yr}: accuracy={fold['accuracy']:.3f}  log-loss={fold['log_loss']:.4f}  RPS={fold['rps']:.4f}")

best = min(summary, key=lambda n: summary[n]["ll_mean"])
print(f"\nBest model by mean log-loss: {best} ({summary[best]['ll_mean']:.4f})")
