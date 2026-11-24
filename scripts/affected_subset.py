"""S30/S31 from Full and Names-only PCBA 10-shot query predictions, seeds 0/1/2.

Use the corrected-fingerprint, train-only text, panel-visible, episode-2000
source exports identified in the supplement. Paths are ordered by training seed.
CSV and CSV.GZ files are streamed. Column options map the native export header.
"""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path


PCBA_TEST = {0, 1, 2, 3, 4, 123, 124, 125, 126, 127}


def ranking_metrics(records):
    """AUROC with half credit for ties and non-interpolated AP."""
    positive = sum(label for label, _ in records.values())
    negative = len(records) - positive
    if not positive or not negative:
        return None, None
    ordered = sorted(records.values(), key=lambda item: -item[1])
    tp = fp = concordance = ap = 0
    for _, group in itertools.groupby(ordered, key=lambda item: item[1]):
        labels = [label for label, _ in group]
        pos = sum(labels)
        neg = len(labels) - pos
        concordance += neg * (tp + 0.5 * pos)
        tp += pos
        fp += neg
        ap += (pos / positive) * (tp / (tp + fp))
    return 100 * concordance / (positive * negative), 100 * ap


def read_file(path, seed, columns, subset_only):
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    selected, molecules = {}, set()
    rows_read = 0
    with opener(path, "rt", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        idx = {field: header.index(column) for field, column in columns.items()
               if field != "flag" or not subset_only}
        seed_col = header.index("training_seed") if "training_seed" in header else None
        dataset_col = header.index("dataset") if "dataset" in header else None
        shot_col = header.index("shot") if "shot" in header else None
        for row in reader:
            rows_read += 1
            if ((seed_col is not None and int(row[seed_col]) != seed)
                    or (dataset_col is not None and row[dataset_col].lower() != "pcba")
                    or (shot_col is not None and int(row[shot_col]) != 10)):
                raise ValueError(f"Unexpected dataset, seed or shot in {path}, row {rows_read+1}")
            key = int(row[idx["task"]]), int(row[idx["repeat"]])
            if key not in selected:
                selected[key] = {}
            if not subset_only:
                flag = int(row[idx["flag"]])
                if flag == 0:
                    continue
                if flag != 1:
                    raise ValueError(f"Expected a binary empty-scaffold flag in {path}")
            mol = int(row[idx["mol"]])
            label = float(row[idx["label"]])
            score = float(row[idx["score"]])
            if label not in (0, 1) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid binary activity or predicted probability in {path}")
            if mol in selected[key]:
                raise ValueError(f"Duplicate affected molecule in {path}, episode {key}: {mol}")
            selected[key][mol] = int(label), score
            molecules.add(mol)
    expected = {(task, rep) for task in PCBA_TEST for rep in range(20)}
    if set(selected) != expected:
        raise ValueError(f"Expected all 10 held-out PCBA assays x 20 support repeats in {path}")

    rows, signatures = [], {}
    for (task, repeat), records in sorted(selected.items()):
        auroc, ap = ranking_metrics(records)
        members = sorted((mol, label) for mol, (label, _) in records.items())
        signatures[task, repeat] = hashlib.sha256(json.dumps(members).encode("ascii")).hexdigest()
        rows.append(dict(task_id=task, support_repeat=repeat, training_seed=seed,
                         query_count=len(records), query_positive=sum(y for _, y in members),
                         AUROC_pct=auroc, AP_pct=ap))
    return rows, signatures, molecules, rows_read


def summarize_affected(rows):
    metrics = ["AUROC_pct", "AP_pct"]
    seeds, assay_seeds, assay_summary = [], [], []
    for seed in range(3):
        within_seed = []
        for task in sorted(PCBA_TEST):
            records = [r for r in rows if r["task_id"] == task and r["training_seed"] == seed]
            valid = [r for r in records if r["AUROC_pct"] is not None]
            item = dict(task_id=task, training_seed=seed, total_evaluations=len(records),
                        eligible_evaluations=len(valid))
            for metric in metrics:
                item[metric] = statistics.mean(r[metric] for r in valid) if valid else None
            assay_seeds.append(item)
            if valid:
                within_seed.append(item)
        if not within_seed:
            raise ValueError(f"No eligible assays for seed {seed}; no seed-level mean can be estimated")
        seeds.append(dict(training_seed=seed, eligible_assays=len(within_seed),
                          **{m: statistics.mean(r[m] for r in within_seed) for m in metrics}))
    for task in sorted(PCBA_TEST):
        episodes = [r for r in rows if r["task_id"] == task]
        eligible = [r for r in assay_seeds if r["task_id"] == task and r["AUROC_pct"] is not None]
        item = dict(task_id=task, eligible_seeds=len(eligible),
                    eligible_evaluations=sum(r["eligible_evaluations"] for r in eligible))
        for col in ("query_count", "query_positive"):
            item[col + "_range"] = [min(r[col] for r in episodes), max(r[col] for r in episodes)]
        for m in metrics:
            values = [r[m] for r in eligible]
            item[m] = dict(mean=statistics.mean(values) if values else None,
                           sample_sd=statistics.stdev(values) if len(values) > 1 else None)
        assay_summary.append(item)
    return dict(episodes=rows, assay_seeds=assay_seeds, by_assay=assay_summary, by_seed=seeds,
                eligible_evaluations=sum(r["AUROC_pct"] is not None for r in rows),
                total_evaluations=len(rows), overall={
                    m: dict(mean=statistics.mean(r[m] for r in seeds),
                            sample_sd=statistics.stdev(r[m] for r in seeds)) for m in metrics})


def build_report(full, names, columns, subset_only=False):
    results = {}
    for arm, files in [("Full", full), ("Names", names)]:
        combined, memberships, molecules, sources = [], {}, set(), []
        for seed, path in enumerate(files):
            episodes, signatures, ids, count = read_file(Path(path), seed, columns, subset_only)
            combined.extend(episodes)
            memberships[seed] = signatures
            molecules.update(ids)
            sources.append(dict(path=str(path), training_seed=seed, input_rows=count))
        if arm == "Full":
            full_memberships = memberships
        elif full_memberships != memberships:
            raise ValueError("Full and Names have different affected molecules or activity labels")
        results[arm] = dict(sources=sources, distinct_affected_molecules=len(molecules),
                            **summarize_affected(combined))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("full", "names"):
        parser.add_argument("--" + arg, nargs=3, required=True, metavar=("SEED0", "SEED1", "SEED2"))
    parser.add_argument("--subset-only", action="store_true",
                        help="The six files have already been filtered to empty_scaffold=1")
    defaults = dict(task="task_id", repeat="support_repeat", mol="molecule_id", label="y_true",
                    score="y_prob", flag="empty_scaffold")
    for field, default in defaults.items():
        parser.add_argument("--" + field + "-column", default=default)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    columns = {field: getattr(args, field + "_column") for field in defaults}
    report = build_report(args.full, args.names, columns, args.subset_only)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
