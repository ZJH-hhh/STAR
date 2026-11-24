import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.precompute_property_embeddings import build_revision_resources, ordered_rows
from semantic_utils import load_task_embeddings


def fixture():
    return [
        dict(dataset="test-panel", task_id=str(i), task_name=f"Example {i}",
             split="train" if i < 4 else "test", processed_text=text,
             names_only_text=f"Example {i}")
        for i, text in enumerate(["first kinase", "enzyme beta", "alpha transport", "receptor binding",
                                  "unseenalpha transport", "unseenbeta ligand"])
    ]


class TextResourceTests(unittest.TestCase):
    def test_numpy_resources_and_controls(self):
        rows = fixture()[::-1]
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            full = build_revision_resources(rows, dest, "all-corpus", seeds=(1,))
            root = dest / "all_corpus"
            loaded = load_task_embeddings(root / "full.npz")
            self.assertEqual(loaded["property_names"], [r["task_name"] for r in rows[::-1]])
            np.testing.assert_allclose(loaded["embedding_matrix"].numpy(), full)
            np.testing.assert_allclose(np.linalg.norm(full, axis=1), 1, atol=1e-6)
            with np.load(root / "name.npz") as names:
                self.assertEqual(names["embedding_matrix"].shape, full.shape)
                self.assertTrue(np.all(names["embedding_matrix"][:, 1:] == 0))
            perm = json.loads((root / "shuffled_seed1.json").read_text())["source_task_ids"]
            self.assertTrue(np.all(np.array(perm) != np.arange(len(rows))))
            self.assertEqual(sorted(perm), list(range(len(rows))))
            with np.load(root / "shuffled_seed1.npz") as shuffled:
                np.testing.assert_array_equal(shuffled["embedding_matrix"], full[perm])
            with np.load(root / "random_seed1.npz") as random:
                self.assertEqual(random["embedding_matrix"].shape, full.shape)
                self.assertFalse(np.array_equal(random["embedding_matrix"], full))

    def test_train_only_excludes_held_out_text_from_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            rows = fixture()
            original = build_revision_resources(rows, dest, "meta-train-only", seeds=(0,))
            loaded = load_task_embeddings(dest / "full.pt")
            self.assertTrue(torch.equal(loaded["embedding_matrix"], torch.from_numpy(original)))
            self.assertEqual(loaded["metadata"]["fit_task_ids"], [0, 1, 2, 3])
            rows[-1]["processed_text"] = "completelyunseen"
            changed = build_revision_resources(rows, dest / "changed", "meta-train-only", seeds=(0,))
            np.testing.assert_array_equal(changed[:4], original[:4])
            np.testing.assert_array_equal(changed[-1], np.zeros(changed.shape[1]))
            perm = json.loads((dest / "shuffled_seed0.json").read_text())["source_task_ids"]
            self.assertEqual(sorted(perm[:4]), [0, 1, 2, 3])

    def test_registry_validation(self):
        rows = fixture()
        rows[0]["processed_text_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            ordered_rows(rows)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            ordered_rows(rows[1:])
