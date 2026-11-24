"""Summarize three S15 attribution.jsonl files and bootstrap 32 molecule clusters."""
import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def ranks(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.all(np.isfinite(scores)):
        raise ValueError("Invalid atom importance scores")
    order = np.argsort(scores, kind="stable")
    values = scores[order]
    boundaries = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    result = np.empty_like(scores)
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        result[order[begin:end]] = (begin + end - 1) / 2.
    return result


def agreement(a, b, overlap):
    x, y = ranks(a["importance"]), ranks(b["importance"])
    if len(x) != len(y):
        raise ValueError("Different atom counts in records for the same molecule")
    x -= x.mean()
    y -= y.mean()
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    corr = float(x.dot(y) / norm) if norm else None
    first, second = set(a["top_atoms"]), set(b["top_atoms"])
    k = max(1, math.ceil(.2 * len(x)))
    if len(first) != k or len(second) != k or not (first | second) <= set(range(len(x))):
        raise ValueError("Invalid top-20% atom set")
    denom = len(first | second) if overlap == "jaccard" else k
    return corr, len(first & second) / denom


def correlation_mean(pairs):
    values = [v[0] for v in pairs if v[0] is not None]
    return float(np.mean(values)) if values else np.nan


def molecule_summary(contexts, overlap):
    tasks, seeds = (1, 4, 10), (0, 1, 2)
    if set(contexts) != set(itertools.product(tasks, seeds)):
        raise ValueError("Expected three assays x three seeds for each molecule")
    same, different = [], []
    for task in tasks:
        rows = [contexts[task, seed] for seed in seeds]
        if len({(r["context_sha256"], r["label"]) for r in rows}) != 1:
            raise ValueError("Mismatched assay context or label across seeds")
        same.extend(agreement(a, b, overlap) for a, b in itertools.combinations(rows, 2))
    for seed in seeds:
        rows = [contexts[task, seed] for task in tasks]
        different.extend(agreement(a, b, overlap) for a, b in itertools.combinations(rows, 2))
    same_corr, diff_corr = correlation_mean(same), correlation_mean(different)
    same_top, diff_top = np.mean([v[1] for v in same]), np.mean([v[1] for v in different])
    if any(len(r["random_drops"]) != 10 for r in contexts.values()):
        raise ValueError("Expected ten matched-size random masks per context")
    top = np.mean([r["top_drop"] for r in contexts.values()])
    random = np.mean([np.mean(r["random_drops"]) for r in contexts.values()])
    return dict(same_assay_rank_correlation=same_corr, different_assay_rank_correlation=diff_corr,
                same_minus_different_rank_correlation=same_corr-diff_corr,
                same_assay_overlap=same_top, different_assay_overlap=diff_top,
                same_minus_different_overlap=same_top-diff_top, top_margin_decrease=top,
                random_margin_decrease=random, top_minus_random_margin_decrease=top-random)


def build_report(paths, overlap="fraction", bootstrap_seed=0):
    grouped = defaultdict(dict)
    for path in paths:
        with Path(path).open(encoding="utf-8-sig") as stream:
            for line in stream:
                row = json.loads(line)
                entries = grouped[row["molecule_id"]]
                key = row["task_id"], row["training_seed"]
                if key in entries:
                    raise ValueError("Duplicate molecule/assay/seed")
                entries[key] = row
    if len(grouped) != 32:
        raise ValueError("Expected the 32 selected molecule clusters")
    clusters = [molecule_summary(grouped[m], overlap) for m in sorted(grouped)]
    patterns = Counter(tuple(entries[task, 0]["label"] for task in (1, 4, 10))
                       for entries in grouped.values())
    if len(patterns) != 8 or set(patterns.values()) != {4}:
        raise ValueError("Expected four unique molecules per activity pattern")
    draws = np.random.RandomState(bootstrap_seed).randint(0, len(clusters), (20000, len(clusters)))
    output = {}
    for metric in clusters[0]:
        values = np.array([r[metric] for r in clusters])
        valid = np.isfinite(values)
        samples = values[draws]
        counts = np.isfinite(samples).sum(axis=1)
        nonempty = counts > 0
        means = np.nansum(samples[nonempty], axis=1) / counts[nonempty]
        ci = np.percentile(means, [2.5, 97.5]).tolist() if len(means) else [None, None]
        output[metric] = dict(estimate=float(values[valid].mean()) if valid.any() else None,
                              lower95=ci[0], upper95=ci[1], eligible_molecules=int(valid.sum()))
    return dict(metrics=output, bootstrap_unit="molecule", bootstrap_resamples=20000,
                bootstrap_seed=bootstrap_seed, interval="percentile",
                correlation="Spearman, average tied ranks, excluding undefined correlations",
                overlap=overlap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs=3, type=Path)
    parser.add_argument("--overlap", choices=["fraction", "jaccard"], default="fraction",
                        help="Size of intersection divided by k (fraction) or union (jaccard)")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args.files, args.overlap, args.bootstrap_seed)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
