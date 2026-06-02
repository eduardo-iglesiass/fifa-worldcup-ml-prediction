import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 10_000
Z = 1.96  # 95% CI

df = pd.read_csv(os.path.join(BASE_DIR, "outputs", "results", "wc2026_win_probabilities.csv"))
df = df[df["win_pct"] > 0].copy()
df = df.sort_values("win_pct", ascending=False).reset_index(drop=True)
df["p"] = df["win_pct"] / 100.0


def wilson_ci(p, n, z=1.96):
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lo = np.maximum(centre - margin, 0.0)
    hi = np.minimum(centre + margin, 1.0)
    return lo, hi


df["ci_lo"], df["ci_hi"] = wilson_ci(df["p"].values, N)

print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{2026 World Cup Win Probabilities with 95\% Wilson Score Confidence Intervals ($N = 10{,}000$ simulations)}")
print(r"\label{tab:ci2026}")
print(r"\begin{tabular}{@{}clrrr@{}}")
print(r"\toprule")
print(r"Rank & Team & Win Prob (\%) & CI Lower (\%) & CI Upper (\%) \\")
print(r"\midrule")
for i, row in df.iterrows():
    rank   = i + 1
    team   = row["team"]
    pct    = row["win_pct"]
    lo_pct = row["ci_lo"] * 100
    hi_pct = row["ci_hi"] * 100
    print(f"{rank} & {team} & {pct:.2f} & {lo_pct:.2f} & {hi_pct:.2f} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

top_team = df.iloc[0]
print(f"\n{top_team['team']}: {top_team['win_pct']:.1f}%  "
      f"95% CI [{top_team['ci_lo']*100:.1f}%, {top_team['ci_hi']*100:.1f}%]")
