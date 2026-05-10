import json
import re
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(".")
RUN_FILE = ROOT / "runpod_run_small.json"
SNAPSHOT_ROOT = ROOT / "snapshots"

TASK_ORDER = ["PudMed20k", "Chest_XR", "NCT_CRC_HE", "GlaS"]

def seed_from_path(path):
    m = re.search(r"seed_(\d+)", str(path))
    return int(m.group(1)) if m else None

with open(RUN_FILE, "r") as f:
    run_data = json.load(f)

metric_files = sorted(SNAPSHOT_ROOT.rglob("metrics.json"))

by_task = {}
raw_metrics = []

for p in metric_files:
    with open(p, "r") as f:
        m = json.load(f)

    task = m.get("task", p.parent.name)
    seed = seed_from_path(p)

    m["_metrics_file"] = str(p)
    m["_seed"] = seed

    raw_metrics.append(m)
    by_task.setdefault(task, []).append(m)

compact = []
averages = {}

for task in TASK_ORDER:
    items = sorted(by_task.get(task, []), key=lambda x: (x.get("_seed") is None, x.get("_seed")))

    if not items:
        continue

    if task == "GlaS":
        keys = ["dice", "hd"]
    else:
        keys = ["acc", "auc", "f1"]

    seed_results = []
    avg_payload = {}

    for m in items:
        seed_payload = {"seed": m.get("_seed")}
        for k in keys:
            seed_payload[k] = m.get(k)
        seed_results.append(seed_payload)

    for k in keys:
        vals = [m.get(k) for m in items if isinstance(m.get(k), (int, float))]
        if vals:
            avg_payload[k] = {
                "mean": mean(vals),
                "std": stdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals)
            }

    compact.append({
        task: {
            "seeds": seed_results,
            "average": avg_payload
        }
    })

    averages[task] = avg_payload

run_data["seed_results"] = compact
run_data["averages"] = averages
run_data["raw_metrics"] = raw_metrics

with open(RUN_FILE, "w") as f:
    json.dump(run_data, f, indent=4)

print(f"Found {len(metric_files)} metrics files")
for p in metric_files:
    print(" -", p)
print(f"Updated {RUN_FILE}")
