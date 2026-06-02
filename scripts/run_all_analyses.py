import subprocess
import sys
import os
import time

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON      = sys.executable
OUTPUT_FILE = os.path.join(BASE_DIR, "outputs", "results", "all_analyses_output.txt")

SCRIPTS = [
    ("Script 2 — RPS Comparison",              "add_rps.py"),
    ("Script 3 — Confidence Intervals",        "confidence_intervals.py"),
    ("Script 4 — Feature Ablation",            "feature_ablation.py"),
    ("Script 5 — Elo Sensitivity",             "elo_sensitivity.py"),
    ("Script 1 — World Cup Backtests",         "backtest_worldcups.py"),
    ("Script 6 — Multi-Tournament Backtest",   "multi_tournament_backtest.py"),
    ("Script 7 — Walk-Forward CV",             "walk_forward_cv.py"),
    ("Script 8 — Diebold-Mariano Test",        "diebold_mariano.py"),
]

def run_script(label, script_path):
    full_path = os.path.join(BASE_DIR, "scripts", script_path)
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, full_path],
        capture_output=True,
        text=True,
        cwd=BASE_DIR,
    )
    elapsed = time.time() - t0
    output = result.stdout
    if result.returncode != 0 and result.stderr:
        output += f"\n[STDERR]\n{result.stderr}"
    print(output)
    print(f"  Done in {elapsed:.1f}s")
    return label, output, elapsed

all_output = []

total_start = time.time()
for label, path in SCRIPTS:
    label, out, elapsed = run_script(label, path)
    all_output.append((label, out, elapsed))

total_elapsed = time.time() - total_start

os.makedirs(os.path.join(BASE_DIR, "outputs", "results"), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("ALL ANALYSES OUTPUT\n")
    f.write("=" * 60 + "\n\n")
    for label, out, elapsed in all_output:
        f.write(f"\n{'='*60}\n")
        f.write(f"  {label}  ({elapsed:.1f}s)\n")
        f.write(f"{'='*60}\n")
        f.write(out)
        f.write("\n")

print(f"\n{'='*60}")
print(f"  All done in {total_elapsed/60:.1f} minutes")
print(f"  Full output saved to: {OUTPUT_FILE}")
print(f"{'='*60}")
