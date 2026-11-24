"""Protocol, screening and optional molecular-prediction exports for a fixed-endpoint run."""
import csv
import gzip
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkstemp

from evaluation import manifest_hash


@contextmanager
def _atomic_path(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = mkstemp(dir=str(destination.parent), prefix="." + destination.name + ".")
    os.close(fd)
    try:
        yield temporary
        os.replace(temporary, str(destination))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _text_resource(args, conditioning):
    if conditioning == "support_adapted_id" or not (args.use_ltafm or args.use_semantic_sampler):
        return {"semantic_embedding_path": None, "text_variant": None, "text_fit_scope": None}
    path = Path(args.semantic_embedding_path)
    metadata = {}
    if path.with_suffix(".json").is_file():
        with path.with_suffix(".json").open(encoding="utf-8-sig") as stream:
            header = json.load(stream)
        metadata.update(header)
        metadata.update(header.get("metadata", {}))
    return {"semantic_embedding_path": str(path),
            "text_variant": metadata.get("text_variant", metadata.get("variant", path.stem)),
            "text_fit_scope": metadata.get("fit_scope")}


def write_run_metadata_and_screening(output_dir, args, rows, summary):
    """Write S29-compatible artifacts and update summary before the caller serializes it."""
    output_dir = Path(output_dir)
    conditioning = summary.get("conditioning", getattr(args, "conditioning", "text"))
    used_id = conditioning == "support_adapted_id"
    active_conditioning = conditioning if used_id or args.use_ltafm or args.use_semantic_sampler else "none"
    resource = _text_resource(args, active_conditioning)
    digest = summary.get("manifest_sha256") or manifest_hash(args.eval_manifest)
    protocol = dict(dataset=args.dataset, shot=args.n_support, training_seed=args.random_seed,
                    checkpoint_policy="fixed_final", endpoint=args.episode,
                    meta_test_metrics_used_for_selection=False,
                    query_target_label_supplied_to_model=False,
                    query_auxiliary_label_mode="mask" if args.mask_query_aux else "retain",
                    query_auxiliary_labels_supplied_to_model=not args.mask_query_aux,
                    fingerprint_mode=args.fingerprint_mode,
                    empty_scaffold_zero_fingerprint_rule=args.fingerprint_mode == "legacy",
                    conditioning=active_conditioning, support_adapted_id=used_id,
                    use_ltafm=bool(args.use_ltafm), use_semantic_sampler=bool(args.use_semantic_sampler),
                    use_knowledge_edges=args.use_knowledge_edges, relation_backend=args.relation_backend,
                    rel_top_k=args.rel_top_k, inner_update_step=args.inner_update_step,
                    eval_manifest=str(args.eval_manifest), eval_manifest_sha256=digest,
                    support_repeats=20, test_batch_size=args.test_batch_size,
                    aggregation="Mean within assay across repetitions, then equal mean across assays",
                    **resource)
    measures = ("task_id", "support_repeat", "query_count", "hits_at_100", "precision_at_100", "EF_at_100")
    with _atomic_path(output_dir / "per_episode_screening_top100.csv") as temporary:
        with open(temporary, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("dataset", "shot", "training_seed", "budget_k") + measures)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(dataset=args.dataset, shot=args.n_support, training_seed=args.random_seed,
                                     budget_k=min(100, row["query_count"]),
                                     **{field: row[field] for field in measures}))
    with _atomic_path(output_dir / "evaluation_protocol.json") as temporary:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(protocol, stream, indent=2, ensure_ascii=False, allow_nan=False)
    summary.update(macro_auroc=summary["AUROC_pct"] / 100,
                   macro_auprc=summary["AP_pct"] / 100,
                   num_assays=len({row["task_id"] for row in rows}),
                   fingerprint_mode=args.fingerprint_mode, support_adapted_id=used_id,
                   conditioning=active_conditioning, **resource)


class _PredictionStream:
    def __init__(self, stream):
        self.csv = csv.writer(stream)
        self.csv.writerow(("task_id", "support_repeat", "molecule_id", "y_true", "y_prob", "empty_scaffold"))
        self.dataset = None
        self.flags = {}
        self.smiles = None

    def empty_scaffold(self, dataset, index):
        if self.dataset is None:
            self.dataset = dataset
            # Older PyG stores the collated graph as data; newer releases use _data.
            # Read the SMILES column directly when local IDs index the complete dataset.
            state = getattr(dataset, "__dict__", {})
            storage = state.get("_data")
            if storage is None:
                storage = state.get("data")
            smiles = getattr(storage, "smiles", None)
            if isinstance(smiles, (tuple, list)) and getattr(dataset, "_indices", None) is None:
                self.smiles = smiles
        elif self.dataset is not dataset:
            raise ValueError("One prediction file must use a single dataset's local molecule IDs")
        if index not in self.flags:
            from rdkit.Chem.Scaffolds import MurckoScaffold
            smiles = self.smiles[index] if self.smiles is not None else dataset[index].smiles
            self.flags[index] = int(MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles) == "")
        return self.flags[index]


@contextmanager
def prediction_writer(output_dir, enabled):
    """Publish a complete compressed CSV on clean exit. Preserve any earlier export on failure."""
    if not enabled:
        yield None
        return
    with _atomic_path(Path(output_dir) / "predictions.csv.gz") as temporary:
        with gzip.open(temporary, "wt", compresslevel=3, encoding="utf-8", newline="") as stream:
            yield _PredictionStream(stream)


def write_predictions(writer, task_id, repeat, indices, labels, scores, dataset):
    if writer is None:
        return
    if len(indices) != len(labels) or len(indices) != len(scores):
        raise ValueError("A query molecule ID, activity label and probability are required for each row")
    for index, label, score in zip(indices, labels, scores):
        index, label, score = int(index), float(label), float(score)
        if index < 0 or label not in (0., 1.) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Prediction export requires binary labels and finite probabilities")
        writer.csv.writerow((task_id, repeat, index, int(label), score, writer.empty_scaffold(dataset, index)))
