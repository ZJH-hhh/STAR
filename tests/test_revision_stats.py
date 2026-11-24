import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts import affected_subset as affected
from scripts import summarize_revision as stats


class RevisionStatsTests(unittest.TestCase):
    def test_raw_run_pairing_and_variability(self):
        with tempfile.TemporaryDirectory() as tmp:
            arms = []
            for name, offset in [("Full", 1), ("Names", 0)]:
                paths = []
                for seed in range(3):
                    path = Path(tmp, name + str(seed))
                    path.mkdir()
                    paths.append(path)
                    meta = dict(dataset="tox21", shot=10, training_seed=seed, endpoint=2000,
                                manifest_sha256=str(seed))
                    (path/"summary.json").write_text(json.dumps(meta))
                    with (path/"per_episode_metrics.csv").open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=["task_id", "support_repeat", *stats.METRICS])
                        writer.writeheader()
                        for task in range(3):
                            for rep in range(20):
                                values = {m: seed + task + offset for m in stats.METRICS}
                                writer.writerow(dict(task_id=task, support_repeat=rep, **values))
                arms.append((name, paths))
            report = stats.build_report(arms)
            self.assertEqual(report["arms"]["Full"]["AUROC_pct"]["between_assay_sd"], 1.)
            paired = report["paired_reference_minus_comparator"]["Names"]["AUROC_pct"]
            self.assertEqual(paired["seed_0_1_2"], [1., 1., 1.])
            self.assertEqual(paired["sample_sd"], 0.)
            meta["manifest_sha256"] = "mismatched"
            (path/"summary.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "Unmatched"):
                stats.build_report(arms)

    def test_affected_ineligible_and_tied_predictions(self):
        self.assertEqual(affected.ranking_metrics({0: (1, .5), 1: (0, .5)}), (50., 50.))
        self.assertEqual(affected.ranking_metrics({0: (0, .5)}), (None, None))
        with tempfile.TemporaryDirectory() as tmp:
            full, names = [], []
            columns = dict(task="task", repeat="rep", mol="mol", label="y", score="p", flag="flag")
            for name, paths in [("full", full), ("names", names)]:
                for seed in range(3):
                    path = Path(tmp, name+str(seed)+".csv")
                    paths.append(path)
                    with path.open("w", newline="") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(["task", "rep", "mol", "y", "p", "flag"])
                        for task in affected.PCBA_TEST:
                            for rep in range(20):
                                writer.writerow([task, rep, 10, 0, .5, 1])
                                writer.writerow([task, rep, 11, 1, .5, int(task != 0)])
            report = affected.build_report(full, names, columns)
            self.assertEqual(report["Full"]["eligible_evaluations"], 540)
            self.assertEqual(report["Full"]["overall"]["AUROC_pct"]["mean"], 50.)
            self.assertIsNone(report["Full"]["by_assay"][0]["AP_pct"]["mean"])
            with names[-1].open("a", newline="") as stream:
                csv.writer(stream).writerow([1, 1, 20, 1, .5, 1])
            with self.assertRaisesRegex(ValueError, "different affected molecules"):
                affected.build_report(full, names, columns)


if __name__ == "__main__":
    unittest.main()
