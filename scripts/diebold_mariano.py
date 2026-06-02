import os
import warnings
import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

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

df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "features_phase1.csv"),
    parse_dates=["date"],
)
for col in ["t_friendly", "t_other_competitive", "t_qualifier", "t_world_cup"]:
    if col in df.columns:
        df[col] = df[col].astype(int)

train = df[df["date"] < TRAIN_CUTOFF]
test  = df[df["date"] >= TRAIN_CUTOFF]

y_test = test["result_num"].values

imp = SimpleImputer(strategy="mean")
X_train_imp = imp.fit_transform(train[FEATURES])
X_test_imp  = imp.transform(test[FEATURES])

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train_imp)
X_test_sc  = sc.transform(X_test_imp)

with open(os.path.join(BASE_DIR, "models", "all_models.pkl"), "rb") as f:
    saved = pickle.load(f)


def per_obs_logloss(y_true, y_prob):
    n = len(y_true)
    losses = np.empty(n)
    for i in range(n):
        p = y_prob[i, int(y_true[i])]
        losses[i] = -np.log(np.clip(p, 1e-15, 1.0))
    return losses


model_losses = {}

for name, mdl in saved.items():
    if name == "Logistic Regression":
        y_prob = mdl.predict_proba(X_test_sc)
    else:
        y_prob = mdl.predict_proba(X_test_imp)
    model_losses[name] = per_obs_logloss(y_test, y_prob)


def diebold_mariano(l1, l2, h=1):
    # Positive DM means model 1 is worse than model 2
    # HLN correction (Harvey, Leybourne, Newbold 1997) for finite samples
    d = l1 - l2
    n = len(d)
    d_bar   = d.mean()
    gamma_0 = np.var(d, ddof=0)
    se      = np.sqrt(gamma_0 / n)

    dm_stat = d_bar / se
    p_val   = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))

    v       = (n + 1 - 2 * h + h * (h - 1) / n) / n
    dm_hln  = dm_stat * np.sqrt(v)
    p_hln   = 2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=n - 1))

    return dm_stat, p_val, dm_hln, p_hln, d_bar, n


l_lr  = model_losses["Logistic Regression"]
l_rf  = model_losses["Random Forest"]
l_xgb = model_losses["XGBoost"]

dm_lr_rf  = diebold_mariano(l_lr, l_rf)
dm_lr_xgb = diebold_mariano(l_lr, l_xgb)

print("=" * 60)
print("DIEBOLD-MARIANO TEST")
print("=" * 60)
print()

n_test = len(y_test)
print(f"Test set: {n_test:,} observations (matches from {TRAIN_CUTOFF} onwards)")
print()

for name, losses in model_losses.items():
    print(f"  {name}: mean log-loss = {losses.mean():.4f}")
print()

print(r"\begin{table}[ht]")
print(r"\centering")
print(r"\caption{Diebold--Mariano Test: Pairwise Log-Loss Comparison on Test Set "
      r"($n=" + f"{n_test:,}" + r"$ observations). "
      r"$H_0$: equal predictive accuracy. "
      r"Positive DM statistic indicates the first-listed model is less accurate.}")
print(r"\label{tab:dm_test}")
print(r"\begin{tabular}{@{}llrrrrl@{}}")
print(r"\toprule")
print(r"Model 1 & Model 2 & "
      r"$\bar{d}$ & "
      r"DM Statistic & "
      r"$p$-value & "
      r"HLN $p$-value & "
      r"Conclusion \\")
print(r"\midrule")


def sig_label(p):
    if p < 0.01:
        return "Significant ($p<0.01$)"
    elif p < 0.05:
        return "Significant ($p<0.05$)"
    elif p < 0.10:
        return "Marginal ($p<0.10$)"
    else:
        return "Not significant"


for label, (m1, m2), result in [
    ("LR vs RF",       ("Logistic Regression", "Random Forest"), dm_lr_rf),
    ("LR vs XGBoost",  ("Logistic Regression", "XGBoost"),       dm_lr_xgb),
]:
    dm, p, dm_hln, p_hln, d_bar, n = result
    conclusion = sig_label(p_hln)
    print(f"{m1} & {m2} & "
          f"${d_bar:+.5f}$ & "
          f"${dm:.3f}$ & "
          f"${p:.4f}$ & "
          f"${p_hln:.4f}$ & "
          f"{conclusion} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

print()
print("INTERPRETATION:")
print()

dm_lr_rf_stat, dm_lr_rf_p, _, dm_lr_rf_hln_p, d_bar_rf, _ = dm_lr_rf
dm_lr_xgb_stat, dm_lr_xgb_p, _, dm_lr_xgb_hln_p, d_bar_xgb, _ = dm_lr_xgb

direction_rf  = "worse than" if d_bar_rf  > 0 else "better than"
direction_xgb = "worse than" if d_bar_xgb > 0 else "better than"

print(f"LR vs Random Forest: LR is {direction_rf} RF  "
      f"(d={d_bar_rf:+.5f}, HLN p={dm_lr_rf_hln_p:.4f})")
print(f"LR vs XGBoost:       LR is {direction_xgb} XGBoost  "
      f"(d={d_bar_xgb:+.5f}, HLN p={dm_lr_xgb_hln_p:.4f})")
