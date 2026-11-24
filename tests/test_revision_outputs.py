import csv
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from revision_outputs import prediction_writer, write_predictions, write_run_metadata_and_screening


class ExportTests(unittest.TestCase):
    def test_protocol_and_summary_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "full.json").write_text(json.dumps({"fit_scope": "meta-train-only", "variant": "full"}))
            args = SimpleNamespace(dataset="pcba", n_support=10, random_seed=2, episode=2000,
                                   mask_query_aux=True, fingerprint_mode="corrected", use_ltafm=True,
                                   use_semantic_sampler=True, relation_backend="krgts", use_knowledge_edges="both",
                                   rel_top_k=1, inner_update_step=3, eval_manifest="seed_2.jsonl", test_batch_size=128,
                                   semantic_embedding_path=str(root / "full.pt"))
            summary = {"AUROC_pct": 75, "AP_pct": 3.3, "manifest_sha256": "test-hash"}
            row = dict(task_id=0, support_repeat=0, query_count=75, hits_at_100=3, precision_at_100=.04, EF_at_100=1)
            write_run_metadata_and_screening(root, args, [row], summary)
            self.assertEqual(summary["macro_auroc"], .75)
            self.assertAlmostEqual(summary["macro_auprc"], .033)
            self.assertEqual(summary["text_fit_scope"], "meta-train-only")
            protocol = json.loads((root / "evaluation_protocol.json").read_text())
            self.assertEqual(protocol["eval_manifest_sha256"], "test-hash")
            self.assertFalse(protocol["query_auxiliary_labels_supplied_to_model"])
            self.assertFalse(protocol["empty_scaffold_zero_fingerprint_rule"])
            with (root / "per_episode_screening_top100.csv").open(newline="") as stream:
                exported = next(csv.DictReader(stream))
            self.assertEqual((exported["budget_k"], exported["hits_at_100"]), ("75", "3"))

    def test_prediction_stream_caches_molecules_and_commits_on_success(self):
        class Dataset:
            def __init__(self):
                self.seen = []

            def __getitem__(self, idx):
                self.seen.append(idx)
                return SimpleNamespace(smiles=["CCO", "c1ccccc1"][idx])

        dataset = Dataset()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with prediction_writer(root, True) as writer:
                for rep in (0, 1):
                    write_predictions(writer, 0, rep, [1, 0], [0, 1], [.1, .7], dataset)
            self.assertEqual(dataset.seen, [1, 0])
            path = root / "predictions.csv.gz"
            with gzip.open(path, "rt", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual([r["empty_scaffold"] for r in rows], ["0", "1", "0", "1"])
            self.assertEqual([r["molecule_id"] for r in rows], ["1", "0", "1", "0"])
            self.assertEqual(rows[2]["support_repeat"], "1")
            committed = path.read_bytes()
            with self.assertRaises(ValueError):
                with prediction_writer(root, True) as writer:
                    write_predictions(writer, 0, 0, [0], [0], [2.0], dataset)
            self.assertEqual(path.read_bytes(), committed)
            self.assertEqual(list(root.iterdir()), [path])
            with prediction_writer(root, False) as writer:
                write_predictions(writer, 0, 0, [], [], [], dataset)
            self.assertEqual(path.read_bytes(), committed)
