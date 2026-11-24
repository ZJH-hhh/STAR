import json
import tempfile
import unittest
from pathlib import Path

import torch

from attribution import intervene, rank_top
from scripts.summarize_attribution import build_report


class AttributionTests(unittest.TestCase):
    def test_query_only_node_mask_and_true_class_margin(self):
        layer = torch.nn.Identity()
        query = torch.tensor([[3.], [5.], [1.], [0.]], requires_grad=True)
        support = torch.tensor([[100.]])
        calls = []

        def score():
            support_value = layer(support).sum()
            query_value = layer(query).sum()
            calls.append(float(support_value))
            return support_value + query_value

        row = intervene(layer, query, score, 0, 12)
        self.assertEqual(row["importance"], [3., 5., 1., 0.])
        self.assertEqual(row["top_atoms"], [1])
        self.assertEqual(row["top_drop"], -5.)
        self.assertEqual(len(row["random_drops"]), 10)
        self.assertEqual(calls, [100.] * 12)
        self.assertEqual(rank_top([1., 1., 0.]), [0])

    def test_bootstrap_uses_complete_matched_molecules(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for seed in range(3):
                path = Path(tmp)/str(seed)
                paths.append(path)
                with path.open("w") as stream:
                    for mol in range(32):
                        for bit, task in enumerate((1, 4, 10)):
                            row = dict(task_id=task, molecule_id=mol, training_seed=seed,
                                       context_sha256=str(task), label=(mol >> bit) % 2,
                                       importance=[1., 2., 3.], top_atoms=[2],
                                       top_drop=3., random_drops=[1.] * 10)
                            stream.write(json.dumps(row) + "\n")
            metrics = build_report(paths)["metrics"]
            self.assertEqual(metrics["same_assay_rank_correlation"]["eligible_molecules"], 32)
            self.assertAlmostEqual(metrics["different_assay_rank_correlation"]["estimate"], 1.)
            contrast = metrics["top_minus_random_margin_decrease"]
            self.assertEqual([contrast[k] for k in ("estimate", "lower95", "upper95")], [2.]*3)
