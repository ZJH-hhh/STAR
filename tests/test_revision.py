import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem

from dataset.dataset import FewshotMolDataset
from evaluation import episode_metrics, read_episodes, summarize


class RevisionTests(unittest.TestCase):
    def test_fingerprints_override_stale_processed_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp, "tox21")
            raw.mkdir()
            smiles = ["CCO", "c1ccccc1"]
            lines = [",".join(["1"] * 12 + [s]) for s in smiles]
            (raw / "tox21.csv").write_text(",".join(["a"] * 12 + ["smiles"]) + "\n" + "\n".join(lines))
            base = FewshotMolDataset(tmp, "tox21")
            # Simulate an old on-disk cache with valid-looking keys.
            records = [base[i] for i in range(2)]
            for record in records:
                record.fingerprint = torch.zeros(2214)
                record.groupprint = torch.zeros(49)
            torch.save(base.collate(records), base.processed_paths[0])
            fixed = FewshotMolDataset(tmp, "tox21", compute_knowledge=True)
            old = FewshotMolDataset(tmp, "tox21", compute_knowledge=True, fingerprint_mode="legacy")
            for i, smi in enumerate(smiles):
                mol = Chem.MolFromSmiles(smi)
                reference = list(MACCSkeys.GenMACCSKeys(mol))[1:]
                reference += list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
                self.assertTrue(torch.equal(fixed[i].fingerprint, torch.tensor(reference)))
                self.assertTrue(torch.equal(fixed[i].groupprint, old[i].groupprint))
            self.assertGreater(int(fixed[0].fingerprint.sum()), 0)
            self.assertEqual(int(old[0].fingerprint.sum()), 0)
            self.assertTrue(torch.equal(old[1].fingerprint, fixed[1].fingerprint))

    def test_manifest_and_metrics(self):
        dataset = SimpleNamespace(name="toy", test_task_range=[0], index_list=[[[0, 1], [2, 3]]])
        rows = [dict(dataset="toy", shot=1, training_seed=0, task_id=0, support_repeat=i,
                     support_indices=[0, 2], query_indices=[1, 3]) for i in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "rows.jsonl")
            def write_rows():
                path.write_text("\n".join(json.dumps(row) for row in rows))
            write_rows()
            self.assertEqual(len(list(read_episodes(path, dataset, 1, 0))), 20)
            rows[-1]["query_indices"] = [0, 3]
            write_rows()
            with self.assertRaises(ValueError):
                list(read_episodes(path, dataset, 1, 0))
            rows.pop()
            write_rows()
            with self.assertRaises(ValueError):
                list(read_episodes(path, dataset, 1, 0))
        result = episode_metrics(np.array([0, 1]), np.array([0.2, 0.8]), [1, 3])
        self.assertEqual(result["AP_pct"], 100)
        self.assertEqual(result["AUROC_pct"], 100)
        self.assertEqual(result["hits_at_100"], 1)
        records = [dict(result, task_id=0)] * 2 + [dict(result, task_id=1, AUROC_pct=0)]
        self.assertEqual(summarize(records)[1]["AUROC_pct"], 50)


if __name__ == "__main__":
    unittest.main()
