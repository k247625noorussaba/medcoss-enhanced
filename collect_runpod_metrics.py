#!/usr/bin/env python3
"""
Scan snapshots/**/metrics.json and merge into a RunPod run JSON.

Updates only seed_results, averages, and raw_metrics (recomputed from disk).
All other top-level keys (dataset_info, "1", "2", etc.) are preserved as loaded.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

TASK_ORDER = ["PudMed20k", "Chest_XR", "NCT_CRC_HE", "GlaS"]


def seed_from_path(path: Path) -> Optional[int]:
    m = re.search(r"seed_(\d+)", str(path))
    return int(m.group(1)) if m else None


def normalize_task(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    t = str(raw).strip()
    if t.lower() in ("pudmed20k", "pubmed20k"):
        return "PudMed20k"
    if t == "Chest_XR":
        return "Chest_XR"
    if t == "NCT_CRC_HE":
        return "NCT_CRC_HE"
    if t.lower() == "glas":
        return "GlaS"
    return None


def load_run(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect metrics.json into seed_results / averages / raw_metrics in a run JSON."
    )
    parser.add_argument("--run-file", default="runpod_run_small.json", help="Run JSON to read and update.")
    parser.add_argument(
        "--snapshot-root",
        default="snapshots",
        help="Directory tree to search for metrics.json (relative to CWD if not absolute).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: same as --run-file).",
    )
    args = parser.parse_args()

    run_file = Path(args.run_file).expanduser()
    snapshot_root = Path(args.snapshot_root).expanduser()
    out_path = Path(args.out).expanduser() if args.out else run_file

    run_data = load_run(run_file)
    snap_resolved = snapshot_root.resolve()
    if not snap_resolved.is_dir():
        print(f"WARNING: snapshot root not a directory: {snap_resolved}", file=sys.stderr)

    metric_files = sorted(snap_resolved.rglob("metrics.json")) if snap_resolved.is_dir() else []

    by_task: Dict[str, List[Dict[str, Any]]] = {}
    raw_metrics: List[Dict[str, Any]] = []

    for p in metric_files:
        try:
            with p.open("r", encoding="utf-8") as f:
                m = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: skip {p}: {e}", file=sys.stderr)
            continue
        if not isinstance(m, dict):
            continue
        task = normalize_task(m.get("task"))
        if task is None:
            print(f"WARNING: skip {p}: unknown task {m.get('task')!r}", file=sys.stderr)
            continue

        seed = seed_from_path(p)
        row = dict(m)
        row["_metrics_file"] = str(p)
        row["_seed"] = seed

        raw_metrics.append(row)
        by_task.setdefault(task, []).append(row)

    compact: List[Dict[str, Any]] = []
    averages: Dict[str, Any] = {}

    for task in TASK_ORDER:
        items = sorted(by_task.get(task, []), key=lambda x: (x.get("_seed") is None, x.get("_seed")))

        if not items:
            continue

        if task == "GlaS":
            keys = ["dice", "hd"]
        else:
            keys = ["acc", "auc", "f1"]

        seed_results = []
        avg_payload: Dict[str, Any] = {}

        for m in items:
            seed_payload: Dict[str, Any] = {"seed": m.get("_seed")}
            for k in keys:
                seed_payload[k] = m.get(k)
            seed_results.append(seed_payload)

        for k in keys:
            vals = [m.get(k) for m in items if isinstance(m.get(k), (int, float))]
            if vals:
                avg_payload[k] = {
                    "mean": mean(vals),
                    "std": stdev(vals) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                }

        compact.append({task: {"seeds": seed_results, "average": avg_payload}})
        averages[task] = avg_payload

    run_data["seed_results"] = compact
    run_data["averages"] = averages
    run_data["raw_metrics"] = raw_metrics

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=4)
        f.write("\n")

    print(f"Found {len(metric_files)} metrics files")
    for p in metric_files:
        print(" -", p)
    print(f"Updated {out_path} (seed_results / averages / raw_metrics from snapshot scan)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
