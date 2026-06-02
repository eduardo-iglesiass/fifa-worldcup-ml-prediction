import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings("ignore")

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute        import SimpleImputer
from sklearn.metrics       import (accuracy_score, log_loss,
                                   confusion_matrix, classification_report)
from xgboost               import XGBClassifier

TRAIN_CUTOFF = "2022-01-01"
RANDOM_STATE = 42

FEATURES = [
    "elo_home", "elo_away", "elo_diff",
    "fifa_points_home", "fifa_points_away", "fifa_points_diff",
    "home_form_winrate", "away_form_winrate", "form_winrate_diff",
    "home_form_gf", "away_form_gf", "form_gf_diff",
    "home_form_ga", "away_form_ga", "form_ga_diff",
    "is_neutral",
    "home_days_rest", "away_days_rest",
    "h2h_home_winrate",
    "wc_experience_home", "wc_experience_away", "wc_experience_diff",
    "t_friendly", "t_other_competitive", "t_qualifier", "t_world_cup",
]

TARGET = "result_num"  # 2=Win, 1=Draw, 0=Loss

print("=" * 60)
print("PHASE 2 — LOADING DATA")
print("=" * 60)

df = pd.read_csv("data/features_phase1.csv", parse_dates=["date"])

for col in ["t_friendly", "t_other_competitive", "t_qualifier", "t_world_cup"]:
    if col in df.columns:
        df[col] = df[col].astype(int)

print(f"Loaded {len(df):,} matches")
print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")

print("\n" + "=" * 60)
print("SPLITTING TRAIN / TEST BY DATE")
print("=" * 60)

train = df[df["date"] < TRAIN_CUTOFF].copy()
test  = df[df["date"] >= TRAIN_CUTOFF].copy()

X_train = train[FEATURES]
y_train = train[TARGET]
X_test  = test[FEATURES]
y_test  = test[TARGET]

print(f"Training set: {len(train):,} matches (before {TRAIN_CUTOFF[:4]})")
print(f"Test set:     {len(test):,} matches ({TRAIN_CUTOFF[:4]} onwards)")

print("\n" + "=" * 60)
print("HANDLING MISSING VALUES (mean imputation)")
print("=" * 60)

# Fit imputer on train only — avoids data leakage
imputer = SimpleImputer(strategy="mean")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)

print(f"Missing values filled with column means")

print("\n" + "=" * 60)
print("DEFINING MODELS")
print("=" * 60)

models = {
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

# LR requires scaled features; tree models do not
scaler   = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled  = scaler.transform(X_test_imp)

print(f"{len(models)} models defined")

print("\n" + "=" * 60)
print("TRAINING AND EVALUATING MODELS")
print("=" * 60)

results = {}

for name, model in models.items():
    print(f"\n  Training {name}...")

    if name == "Logistic Regression":
        X_tr = X_train_scaled
        X_te = X_test_scaled
    else:
        X_tr = X_train_imp
        X_te = X_test_imp

    model.fit(X_tr, y_train)

    y_pred      = model.predict(X_te)
    y_pred_prob = model.predict_proba(X_te)

    acc      = accuracy_score(y_test, y_pred)
    logloss  = log_loss(y_test, y_pred_prob)

    results[name] = {
        "model":      model,
        "y_pred":     y_pred,
        "y_pred_prob":y_pred_prob,
        "accuracy":   acc,
        "log_loss":   logloss,
    }

    print(f"  Accuracy:  {acc:.3f}  ({acc*100:.1f}%)")
    print(f"    Log-loss:  {logloss:.4f}  (lower is better)")

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    name: {
        "Accuracy":   f"{r['accuracy']:.3f}",
        "Log-loss":   f"{r['log_loss']:.4f}",
    }
    for name, r in results.items()
}).T

print(comparison.to_string())

best_name = min(results, key=lambda x: results[x]["log_loss"])
print(f"\n  Best model: {best_name}")

print("\n" + "=" * 60)
print("PLOTTING ACCURACY BY OUTCOME")
print("=" * 60)

outcome_labels = {0: "Loss", 1: "Draw", 2: "Win"}
class_accuracy_rows = []

for name, r in results.items():
    for result_num, result_label in outcome_labels.items():
        actual_mask = y_test.values == result_num
        correct = r["y_pred"][actual_mask] == result_num
        class_accuracy_rows.append({
            "Model": name,
            "Outcome": result_label,
            "Correctly predicted": correct.mean(),
            "Matches": actual_mask.sum(),
        })

class_accuracy = pd.DataFrame(class_accuracy_rows)

print(class_accuracy.pivot(
    index="Model",
    columns="Outcome",
    values="Correctly predicted"
).map(lambda x: f"{x:.1%}").to_string())

plt.figure(figsize=(9, 5))
ax = sns.barplot(
    data=class_accuracy,
    x="Outcome",
    y="Correctly predicted",
    hue="Model",
    order=["Loss", "Draw", "Win"],
    palette=["#4C78A8", "#F58518", "#54A24B"],
)
ax.set_ylim(0, 1)
ax.set_xlabel("Actual match outcome")
ax.set_ylabel("Correct prediction rate")
ax.set_title("Prediction Accuracy by Actual Outcome")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

for container in ax.containers:
    labels = [f"{bar.get_height():.0%}" for bar in container]
    ax.bar_label(container, labels=labels, label_type="edge", padding=2)

plt.legend(title="")
plt.tight_layout()
plt.savefig("accuracy_by_outcome.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: accuracy_by_outcome.png")

print("\n" + "=" * 60)
print("PLOTTING CONFUSION MATRICES")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
labels = ["Loss (0)", "Draw (1)", "Win (2)"]

for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                xticklabels=labels, yticklabels=labels,
                cmap="Blues", cbar=False)
    ax.set_title(f"{name}\nAccuracy: {r['accuracy']:.3f}  Log-loss: {r['log_loss']:.4f}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

plt.suptitle("Confusion Matrices — Test Set (2022 onwards)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: confusion_matrices.png")

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (XGBoost)")
print("=" * 60)

xgb_model = results["XGBoost"]["model"]
importance = pd.Series(
    xgb_model.feature_importances_,
    index=FEATURES
).sort_values(ascending=True)

plt.figure(figsize=(8, 8))
importance.plot(kind="barh", color="steelblue")
plt.title("Feature Importance — XGBoost", fontsize=13)
plt.xlabel("Importance score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: feature_importance.png")

print("\n" + "=" * 60)
print("CALIBRATION CHECK — Best Model")
print("=" * 60)

best_probs = results[best_name]["y_pred_prob"]
win_probs  = best_probs[:, 2]

bins      = np.linspace(0, 1, 11)
bin_idx   = np.digitize(win_probs, bins) - 1
bin_idx   = np.clip(bin_idx, 0, 9)

actual_wins = (y_test.values == 2).astype(int)

bin_mean_pred   = []
bin_mean_actual = []

for i in range(10):
    mask = bin_idx == i
    if mask.sum() > 0:
        bin_mean_pred.append(win_probs[mask].mean())
        bin_mean_actual.append(actual_wins[mask].mean())

plt.figure(figsize=(9, 5))
plt.plot([0,1],[0,1], "k--", label="Perfect calibration")
plt.plot(bin_mean_pred, bin_mean_actual, "o-", color="steelblue", label=best_name)
plt.xlabel("Predicted win probability")
plt.ylabel("Actual win rate")
plt.title(f"Calibration Plot — {best_name}")
plt.legend()
plt.tight_layout()
plt.savefig("calibration_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: calibration_plot.png")

print("\n" + "=" * 60)
print("SAVING BEST MODEL")
print("=" * 60)

save_package = {
    "model":    results[best_name]["model"],
    "imputer":  imputer,
    "scaler":   scaler,
    "features": FEATURES,
    "name":     best_name,
}

with open("best_model.pkl", "wb") as f:
    pickle.dump(save_package, f)

print(f"Saved: best_model.pkl  ({best_name})")

with open("all_models.pkl", "wb") as f:
    pickle.dump({name: r["model"] for name, r in results.items()}, f)

print(f"Saved: all_models.pkl")

print("\n" + "=" * 60)
print(f"DETAILED CLASSIFICATION REPORT — {best_name}")
print("=" * 60)

print(classification_report(
    y_test,
    results[best_name]["y_pred"],
    target_names=["Loss", "Draw", "Win"]
))

print("\n" + "=" * 60)
print("PHASE 2 COMPLETE — ready for Phase 3 (2026 predictions)!")
print("=" * 60)
