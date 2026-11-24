"""Validation and aggregation for the revision JSONL evaluation manifests."""
import hashlib
import json
from collections import defaultdict

import numpy as np


def read_episodes(path, dataset, shot, seed):
    expected = {(task, rep) for task in dataset.test_task_range for rep in range(20)}
    classes = {task: [set(ids) for ids in dataset.index_list[task]]
               for task in dataset.test_task_range}
    with open(path, encoding="utf-8-sig") as stream:
        for line in stream:
            row = json.loads(line)
            task = row["task_id"]
            key = task, row["support_repeat"]
            if (row["dataset"], row["shot"], row["training_seed"]) != (dataset.name, shot, seed):
                raise ValueError("Manifest dataset, shot or training seed mismatch")
            if key not in expected:
                raise ValueError(f"Unexpected or repeated assay/repetition: {key}")
            expected.remove(key)
            support = row["support_indices"]
            query = row["query_indices"]
            s, q = set(support), set(query)
            negatives, positives = classes[task]
            if (len(support) != 2 * shot or len(s) != len(support)
                    or len(s & negatives) != shot or len(s & positives) != shot):
                raise ValueError(f"Support labels do not match the local dataset: {key}")
            if len(q) != len(query) or s & q or (s | q) != (negatives | positives):
                raise ValueError(f"Query IDs do not match remaining labeled molecules: {key}")
            if not (q & positives and q & negatives):
                raise ValueError(f"AUROC requires both query classes: {key}")
            yield row
    if expected:
        raise ValueError(f"Missing {len(expected)} assay/repetition pairs")


def episode_metrics(labels, scores, query_ids):
    from sklearn.metrics import average_precision_score, roc_auc_score
    budget = min(100, len(scores))
    ranked = np.lexsort((query_ids, -scores))
    hits = int(labels[ranked[:budget]].sum())
    precision = hits / budget
    return {"AUROC_pct": 100 * roc_auc_score(labels, scores),
            "AP_pct": 100 * average_precision_score(labels, scores),
            "query_count": len(labels), "hits_at_100": hits,
            "precision_at_100": precision, "EF_at_100": precision / labels.mean()}


def summarize(rows):
    fields = ["AUROC_pct", "AP_pct", "hits_at_100", "precision_at_100", "EF_at_100"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    assays = [{"task_id": task, **{key: float(np.mean([r[key] for r in records]))
                                   for key in fields}} for task, records in sorted(grouped.items())]
    return assays, {key: float(np.mean([r[key] for r in assays])) for key in fields}


def manifest_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
