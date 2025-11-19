"""src/evaluate.py
Independent evaluation & visualisation.
Usage:
    uv run python -m src.evaluate results_dir=/path run_ids='["run-1", "run-2"]'
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from scipy import stats

ROOT_CFG = Path("config/config.yaml")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _load_wandb_credentials() -> tuple[str, str]:
    import yaml
    with open(ROOT_CFG) as f:
        cfg = yaml.safe_load(f)
    return cfg["wandb"]["entity"], cfg["wandb"]["project"]


def _to_py(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (list, tuple)):
        return [_to_py(x) for x in o]
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    return o


def _save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_py(obj), indent=2))
    print(path.resolve())

# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _plot_learning_curve(history: pd.DataFrame, run_id: str, out_dir: Path):
    if "val_em" not in history.columns:
        return
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=history, x="step", y="val_em")
    plt.title(f"Learning curve – {run_id}")
    plt.xlabel("Step"); plt.ylabel("EM accuracy")
    plt.tight_layout()
    f = out_dir / f"{run_id}_learning_curve.pdf"
    plt.savefig(f); plt.close(); print(f.resolve())


def _plot_confusion(correct: int, total: int, run_id: str, out_dir: Path):
    plt.figure(figsize=(3, 3))
    data = np.array([[correct, total - correct]], dtype=int)
    sns.heatmap(data, annot=True, fmt="d", cbar=False, cmap="Blues",
                xticklabels=["Correct", "Incorrect"], yticklabels=[""])
    plt.title(f"Accuracy breakdown – {run_id}")
    plt.tight_layout()
    f = out_dir / f"{run_id}_confusion_matrix.pdf"
    plt.savefig(f); plt.close(); print(f.resolve())


def _plot_bar(metric_map: Dict[str, Dict[str, float]], metric: str, out_dir: Path):
    if metric not in metric_map:
        return
    plt.figure(figsize=(6, 4))
    keys, vals = zip(*metric_map[metric].items())
    sns.barplot(x=list(keys), y=list(vals))
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.tight_layout()
    f = out_dir / f"comparison_{metric}_bar_chart.pdf"
    plt.savefig(f); plt.close(); print(f.resolve())

# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------

def _process_run(api_run: wandb.apis.public.Run, out_root: Path):
    run_dir = out_root / api_run.id
    run_dir.mkdir(parents=True, exist_ok=True)

    hist = api_run.history(keys=None, pandas=True)
    summary = dict(api_run.summary)
    config = dict(api_run.config)

    _save_json(run_dir / "metrics.json", {
        "history": hist.to_dict(orient="list"),
        "summary": summary,
        "config": config,
    })

    _plot_learning_curve(hist, api_run.id, run_dir)
    if "predictions_correct" in summary and "predictions_total" in summary:
        _plot_confusion(int(summary["predictions_correct"]), int(summary["predictions_total"]), api_run.id, run_dir)
    return summary


def _is_higher_better(metric: str) -> bool:
    l = metric.lower()
    return not any(k in l for k in ("loss", "error", "perplexity"))


def _aggregate(summaries: Dict[str, Dict], comp_dir: Path):
    primary = "val_em_mean"
    metric_map: Dict[str, Dict[str, float]] = {}
    for rid, summ in summaries.items():
        for k, v in summ.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                metric_map.setdefault(k, {})[rid] = float(v)

    # Identify best runs ----------------------------------------------
    best_prop = {"run_id": None, "value": -1e9}
    best_base = {"run_id": None, "value": -1e9}
    for rid, v in metric_map.get(primary, {}).items():
        if "proposed" in rid:
            if v > best_prop["value"]:
                best_prop = {"run_id": rid, "value": v}
        elif any(t in rid for t in ("baseline", "comparative")):
            if v > best_base["value"]:
                best_base = {"run_id": rid, "value": v}

    gap = 0.0
    if best_prop["run_id"] and best_base["run_id"]:
        if _is_higher_better(primary):
            gap = 100 * (best_prop["value"] - best_base["value"]) / (best_base["value"] + 1e-12)
        else:
            gap = 100 * (best_base["value"] - best_prop["value"]) / (best_base["value"] + 1e-12)

    # Welch t-test -----------------------------------------------------
    prop_vals = [v for rid, v in metric_map.get(primary, {}).items() if "proposed" in rid]
    base_vals = [v for rid, v in metric_map.get(primary, {}).items() if any(t in rid for t in ("baseline", "comparative"))]
    p_val = None
    if len(prop_vals) > 1 and len(base_vals) > 1:
        _, p_val = stats.ttest_ind(prop_vals, base_vals, equal_var=False)

    comp_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "primary_metric": "Exact-match (EM) accuracy on GSM8K dev. Secondary\u2003(1) steps to reach 55 % EM, (2) per-layer update-to-weight ratio distribution, (3) GPU memory/time overhead relative to constant LR.",
        "metrics": metric_map,
        "best_proposed": best_prop,
        "best_baseline": best_base,
        "gap": gap,
        "p_value_prop_vs_base": p_val,
    }
    _save_json(comp_dir / "aggregated_metrics.json", out)

    # Visuals ----------------------------------------------------------
    _plot_bar(metric_map, primary, comp_dir)
    _plot_bar(metric_map, "steps_to_55_em_mean", comp_dir)

# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_dir", type=str)
    p.add_argument("run_ids", type=str, help="JSON list of WandB run IDs")
    args = p.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    run_ids: List[str] = json.loads(args.run_ids)

    entity, project = _load_wandb_credentials()
    api = wandb.Api()

    summaries: Dict[str, Dict] = {}
    for rid in run_ids:
        run = api.run(f"{entity}/{project}/{rid}")
        summaries[rid] = _process_run(run, results_dir)

    _aggregate(summaries, results_dir / "comparison")

if __name__ == "__main__":
    main()