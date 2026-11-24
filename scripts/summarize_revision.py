"""Repetitions -> assays -> three seed means and paired descriptive t intervals."""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("AUROC_pct", "AP_pct", "hits_at_100", "precision_at_100", "EF_at_100")
TOXCAST = {"toxcast-APR": 10, "toxcast-ATG": 40, "toxcast-BSK": 31,
           "toxcast-CEETOX": 4, "toxcast-CLD": 5, "toxcast-NVS": 39,
           "toxcast-OT": 4, "toxcast-TOX21": 20, "toxcast-Tanguay": 5}
BENCHMARK_ASSAYS = {"muv": 5, "pcba": 10, "sider": 6, "tox21": 3, **TOXCAST}


def describe_three(values):
    if len(values) != 3:
        raise ValueError("This protocol requires exactly three matched training seeds")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half_width = 4.302652729749464 * sd / math.sqrt(3)  # two-sided 95%, df=2
    return dict(mean=mean, sample_sd=sd, lower95=mean-half_width, upper95=mean+half_width)


def read_run(path):
    with (path / "summary.json").open(encoding="utf-8-sig") as stream:
        meta = json.load(stream)
    partition = meta["dataset"]
    rows = defaultdict(dict)
    with (path / "per_episode_metrics.csv").open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            assay, repeat = int(row["task_id"]), int(row["support_repeat"])
            if repeat in rows[assay]:
                raise ValueError(f"Duplicate assay/repetition in {path}: {assay}, {repeat}")
            values = {metric: float(row[metric]) for metric in METRICS}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite metric in {path}")
            rows[assay][repeat] = values
    if partition not in BENCHMARK_ASSAYS or len(rows) != BENCHMARK_ASSAYS[partition]:
        raise ValueError(f"Unexpected held-out assay count or partition in {path}: {partition}")
    if any(set(repeats) != set(range(20)) for repeats in rows.values()):
        raise ValueError(f"Expected repetitions 0..19 for every assay in {path}")
    if not meta.get("manifest_sha256"):
        raise ValueError(f"Missing manifest identity in {path}")
    return meta, rows


def collect_arm(paths):
    seed_assays, manifests = defaultdict(dict), {}
    settings = set()
    for path in paths:
        meta, rows = read_run(Path(path))
        part, shot, seed = meta["dataset"], int(meta["shot"]), int(meta["training_seed"])
        settings.add(("toxcast" if part in TOXCAST else part, shot, int(meta["endpoint"])))
        key = (part, seed)
        if key in manifests:
            raise ValueError(f"Repeated partition/seed: {key}")
        manifests[key] = meta["manifest_sha256"]
        for task, repeats in rows.items():
            seed_assays[seed][(part, task)] = repeats
    if len(settings) != 1 or set(seed_assays) != {0, 1, 2}:
        raise ValueError("Use one benchmark, shot and endpoint with all seeds 0, 1 and 2 per arm")
    benchmark, shot, endpoint = settings.pop()
    parts = TOXCAST if benchmark == "toxcast" else [benchmark]
    if set(manifests) != {(part, seed) for part in parts for seed in range(3)}:
        raise ValueError("Missing physical partition or training seed")
    if not (seed_assays[0].keys() == seed_assays[1].keys() == seed_assays[2].keys()):
        raise ValueError("Held-out assay IDs differ between seeds")
    return (benchmark, shot, endpoint), manifests, seed_assays


def report_arm(seed_assays):
    results = {}
    for metric in METRICS:
        seed_values, support_sds = [], []
        all_assay_means = defaultdict(list)
        for seed in range(3):
            assay_means = []
            for assay_key, repeats in seed_assays[seed].items():
                values = [row[metric] for row in repeats.values()]
                assay_means.append(statistics.mean(values))
                all_assay_means[assay_key].append(assay_means[-1])
                support_sds.append(statistics.stdev(values))
            seed_values.append(statistics.mean(assay_means))
        results[metric] = {**describe_three(seed_values), "seed_0_1_2": seed_values,
                           "mean_within_assay_support_sd": statistics.mean(support_sds),
                           "between_assay_sd": statistics.stdev(statistics.mean(v)
                                                               for v in all_assay_means.values())}
    return results


def from_index(path, benchmark, shot, family, config):
    """Reaggregate the already reported S29 rows, without substituting them for runs."""
    parts = TOXCAST if benchmark == "toxcast" else {benchmark}
    weights, by_seed, manifests = {}, defaultdict(dict), {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            part = row["dataset"]
            if (part not in parts or int(row["shot"]) != shot or row["config_name"] != config
                    or row["job_key"].split("__", 1)[0] != family):
                continue
            seed, count = int(row["training_seed"]), int(row["assay_count"])
            key = part, seed
            if key in manifests or count != BENCHMARK_ASSAYS[part] or int(row["support_repeats"]) != 20:
                raise ValueError("Duplicate row or unexpected coverage: " + row["job_key"])
            manifests[key] = row["eval_manifest"]
            weights[part] = count
            by_seed[seed][part] = {m: float(row[m]) for m in ("AUROC_pct", "AP_pct")}
    if set(manifests) != {(part, seed) for part in parts for seed in range(3)}:
        raise ValueError(f"Incomplete S29 group: {benchmark} {shot}-shot {family}/{config}")
    metrics = {}
    for m in ("AUROC_pct", "AP_pct"):
        values = [sum(records[p][m] * weights[p] for p in parts) / sum(weights.values())
                  for records in (by_seed[s] for s in range(3))]
        metrics[m] = {**describe_three(values), "seed_0_1_2": values}
    return (benchmark, shot, 2000), manifests, set(weights), metrics


def build_report(arms, index=None, benchmark=None, shot=None):
    tables, contrasts = {}, {}
    reference_name, reference_meta, reference_hashes, reference_assays = None, None, None, None
    for name, paths in arms:
        if name in tables:
            raise ValueError(f"Repeated arm label: {name}")
        if index is None:
            meta, hashes, assays = collect_arm(paths)
            assay_keys = set(assays[0])
            tables[name] = report_arm(assays)
        else:
            if len(paths) != 2:
                raise ValueError("S29 mode expects --arm LABEL FAMILY CONFIG_NAME")
            meta, hashes, assay_keys, tables[name] = from_index(index, benchmark, shot, *paths)
        if reference_name is None:
            reference_name, reference_meta = name, meta
            reference_hashes, reference_assays = hashes, assay_keys
        else:
            if (meta != reference_meta or hashes != reference_hashes
                    or assay_keys != reference_assays):
                raise ValueError(f"Unmatched benchmark, shot, endpoint, manifest or assay set: {name}")
            paired = {}
            for metric in tables[name]:
                a, b = tables[reference_name][metric], tables[name][metric]
                diffs = [x - y for x, y in zip(a["seed_0_1_2"], b["seed_0_1_2"])]
                paired[metric] = {**describe_three(diffs), "seed_0_1_2": diffs}
            contrasts[name] = paired
    benchmark, shot, endpoint = reference_meta
    return dict(source="S29 index values; manifest paths matched" if index else
                "per-episode outputs; manifest SHA-256 matched",
                benchmark=benchmark, shot=shot, endpoint=endpoint, arms=tables,
                reference=reference_name, paired_reference_minus_comparator=contrasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", nargs="+", required=True,
                        metavar="VALUE", help="LABEL RUN_DIR... or with --run-map, LABEL FAMILY CONFIG_NAME")
    parser.add_argument("--run-map", type=Path, help="Use native S29 index values for AUROC/AP only")
    parser.add_argument("--dataset", choices=["muv", "pcba", "tox21", "sider", "toxcast"])
    parser.add_argument("--shot", type=int, choices=[1, 10])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.run_map and (args.dataset is None or args.shot is None):
        parser.error("--run-map requires --dataset and --shot")
    report = build_report([(arm[0], arm[1:]) for arm in args.arm], args.run_map, args.dataset, args.shot)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
