#!/usr/bin/env python3
import os
import re
import json
import argparse
from pathlib import Path
from statistics import mean, pstdev


BEST_RE = re.compile(r"best score:\s*([0-9.]+)")
TIME_RE = re.compile(r"time cost:\s*([0-9.]+)s")


def read_params(p: Path) -> dict:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def parse_log(p: Path) -> tuple[float | None, float | None]:
    best, tcost = None, None
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = BEST_RE.search(line)
                if m:
                    try:
                        best = float(m.group(1))
                    except ValueError:
                        pass
                m2 = TIME_RE.search(line)
                if m2:
                    try:
                        tcost = float(m2.group(1))
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return best, tcost


def iter_runs(base_dir: Path):
    # sweep dirs like MMDD-exp_name
    for sweep in sorted(base_dir.iterdir()):
        if not sweep.is_dir():
            continue
        for run in sorted(sweep.iterdir()):
            if not run.is_dir():
                continue
            params_p = run / "params.pkl"
            log_p = run / "train.log"
            if params_p.exists() and log_p.exists():
                yield sweep.name, run.name, params_p, log_p


def main():
    ap = argparse.ArgumentParser(description="Summarize GS* runs by best score/time.")
    ap.add_argument("--base", default="dump", help="Base dump directory")
    ap.add_argument("--dataset", default=None, help="Filter by dataset")
    ap.add_argument("--knowledge", default=None, help="Filter by use_knowledge_edges")
    ap.add_argument("--group", default=None, help="Group keys for mean/std, comma-separated (e.g., dataset,knowledge,episode)")
    ap.add_argument("--csv", action="store_true", help="Print CSV rows")
    args = ap.parse_args()

    rows = []
    base_dir = Path(args.base)
    for sweep, exp_id, params_p, log_p in iter_runs(base_dir):
        params = read_params(params_p)
        dataset = params.get("dataset")
        if args.dataset and dataset != args.dataset:
            continue
        knowledge = params.get("use_knowledge_edges", "none")
        if args.knowledge and knowledge != args.knowledge:
            continue
        seed = params.get("random_seed")
        ep = params.get("episode")
        exp_name = params.get("exp_name")
        best, tcost = parse_log(log_p)
        rows.append({
            "sweep": sweep,
            "exp_id": exp_id,
            "dataset": dataset,
            "knowledge": knowledge,
            "seed": seed,
            "episodes": ep,
            "best": best,
            "time_cost_s": tcost,
            "exp_name": exp_name,
            "path": str(log_p.parent),
        })

    if args.group:
        keys = [k.strip() for k in args.group.split(",") if k.strip()]
        # aggregate by keys
        groups = {}
        for r in rows:
            key = tuple(r.get(k) for k in keys)
            groups.setdefault(key, []).append(r)
        print("# groups:")
        print("group, n, mean_best, std_best, episodes, example_path")
        for key, items in sorted(groups.items()):
            vals = [x["best"] for x in items if isinstance(x.get("best"), (int, float))]
            if not vals:
                continue
            mu = mean(vals)
            sd = pstdev(vals) if len(vals) > 1 else 0.0
            ep = items[0].get("episodes")
            ex = items[0].get("path")
            print(f"{key}, {len(items)}, {mu:.5f}, {sd:.5f}, {ep}, {ex}")

    if args.csv:
        # print detailed csv
        print("sweep,exp_id,dataset,knowledge,seed,episodes,best,time_cost_s,exp_name,path")
        for r in rows:
            print(
                f"{r['sweep']},{r['exp_id']},{r['dataset']},{r['knowledge']},{r['seed']},{r['episodes']},{r['best']},{r['time_cost_s']},{r['exp_name']},{r['path']}"
            )


if __name__ == "__main__":
    main()

