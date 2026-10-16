"""Export FairLens / HMDA aggregates for the portfolio site (revamp Gate 1).

Writes hmda.portfolio.json for the approval-disparity chart:
  - ground-truth approval rates by race, computed from the HMDA data with
    the analysis code's own loader (src/data_loader.py), and their
    disparate impact ratio (DIR = min group rate / max group rate),
  - the debiasing comparison (accuracy, DIR, DPD, EOD, four-fifths pass)
    from the analysis outputs, for the before/after model toggle.

    python scripts/portfolio/export_portfolio.py \
        --analysis-dir ~/Documents/hmda-mortgage --out <path/to/hmda.portfolio.json>

The analysis folder (data, src, results) is not in this repository, so the
export records the sha256 of every input next to this repo's commit.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FOUR_FIFTHS = 0.8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    adir = a.analysis_dir.expanduser()

    sys.path.insert(0, str(adir))
    from src.data_loader import add_protected_attributes, create_target, load_raw_data  # noqa: E402

    data_csv = adir / "data" / "state_mortgage_data.csv"
    debias_csv = adir / "results" / "debiasing_comparison.csv"
    loader_py = adir / "src" / "data_loader.py"

    with contextlib.redirect_stdout(io.StringIO()):   # the loader prints summaries
        df = add_protected_attributes(create_target(load_raw_data(str(data_csv))))

    known = df[df["race"].notna()]
    groups = (
        known.groupby("race")["approved"].agg(["size", "mean"])
        .rename(columns={"size": "n", "mean": "approval_rate"})
        .sort_values("approval_rate", ascending=False)
    )
    top = float(groups["approval_rate"].max())
    approval = [
        {
            "group": g,
            "n": int(row.n),
            "approval_rate": round(float(row.approval_rate), 4),
            "ratio_to_highest": round(float(row.approval_rate) / top, 4),
            "passes_four_fifths": float(row.approval_rate) / top >= FOUR_FIFTHS,
        }
        for g, row in groups.iterrows()
    ]
    dir_truth = float(groups["approval_rate"].min()) / top

    deb = pd.read_csv(debias_csv)
    models = [
        {
            "method": r["Method"],
            # The baseline has no mitigation stage; the CSV leaves it empty, which pandas
            # reads as NaN, and NaN is not JSON (JSON.parse rejects it on the site).
            "stage": None if pd.isna(r["Stage"]) else r["Stage"],
            "accuracy": round(float(r["Accuracy"]), 4),
            "dir": round(float(r["DIR"]), 4),
            "dpd": round(float(r["DPD"]), 4),
            "eod": round(float(r["EOD"]), 4),
            "passes_four_fifths": str(r["4/5 Rule"]).strip().upper() == "PASS",
        }
        for _, r in deb.iterrows()
    ]

    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "repo": "github.com/shauryadata/fairness-aware-mortgage-loan-approval-modeling",
            "commit": commit,
            "inputs": {
                "data/state_mortgage_data.csv": sha256(data_csv),
                "results/debiasing_comparison.csv": sha256(debias_csv),
                "src/data_loader.py": sha256(loader_py),
            },
            "live_app": "https://fairlensweb.vercel.app",
        },
        "dataset": {
            "records": int(len(df)),
            "records_with_known_race": int(len(known)),
            "approval_rate_overall": round(float(df["approved"].mean()), 4),
            "approval_definition": "HMDA action_taken 1, 2, 6 or 8 (originated, approved not accepted, purchased, preapproval approved)",
        },
        "four_fifths_threshold": FOUR_FIFTHS,
        "approval_by_race": approval,
        "dir_ground_truth": round(dir_truth, 4),
        "models": models,
        "models_note": "Model DIR, DPD and EOD are computed on each model's selection rates (analysis results/debiasing_comparison.csv).",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    print(f"wrote {a.out} ({a.out.stat().st_size:,} bytes)")
    print(f"  records: {len(df):,} (known race: {len(known):,}), overall approval {payload['dataset']['approval_rate_overall']}")
    for g in approval:
        print(f"  {g['group']:18} n={g['n']:>7,}  approval {g['approval_rate']:.4f}  ratio {g['ratio_to_highest']:.4f}")
    print(f"  ground-truth DIR: {payload['dir_ground_truth']}")
    for m in models:
        print(f"  {m['method']:20} acc {m['accuracy']:.4f}  DIR {m['dir']:.4f}  {'PASS' if m['passes_four_fifths'] else 'FAIL'}")
    print(f"  source commit: {commit}")


if __name__ == "__main__":
    main()
