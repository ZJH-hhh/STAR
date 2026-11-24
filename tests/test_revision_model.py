import importlib.util
import unittest
from copy import deepcopy
from pathlib import Path

import torch


def load_layer(name):
    # These layers have no dependency on PyG or the optional training runtime.
    path = Path(__file__).resolve().parents[1] / "models" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


film = load_layer("ltafm")
connector = load_layer("graph_connector").InductiveGraphConnector


class ModelRevisionTests(unittest.TestCase):
    def test_assay_variance_is_deterministic_and_differentiable(self):
        torch.manual_seed(3)
        modulator = film.TaskAdaptiveModulator(6, 3, hidden_dim=9, dropout=0.8)
        with torch.no_grad():
            modulator.film.mlp[-1].weight.normal_()
        modulator.train()
        assays = torch.eye(3)
        gamma, _ = modulator.film.modulation(assays, deterministic=True)
        expected = -modulator.diversity_weight * (
            (gamma - gamma.mean(dim=0)).square().mean())
        self.assertTrue(torch.allclose(modulator.assay_diversity(assays), expected))
        self.assertLess(expected.item(), 0)
        self.assertEqual(modulator.assay_diversity(assays[:1]).item(), 0)
        repeated = assays[0:1].expand(3, -1)
        self.assertEqual(modulator.assay_diversity(repeated).item(), 0)
        modulator.assay_diversity(assays).backward()
        self.assertGreater(modulator.film.mlp[-1].weight.grad.abs().sum().item(), 0)

    def test_support_gradient_and_independent_id_initialization(self):
        ids = film.AssayIDEmbedding(5, 4)
        checkpoint = deepcopy(ids)
        ids.reset_held_out(3, 1)
        first = ids.weight[3].detach().clone()
        ids.reset_held_out(4, 1)
        self.assertFalse(torch.equal(first, ids.weight[4]))
        logits = ids(torch.tensor([3])).sum(dim=-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        (gradient,) = torch.autograd.grad(loss, ids.weight)
        self.assertEqual(gradient[:3].abs().sum().item(), 0)
        self.assertGreater(gradient[3].abs().sum().item(), 0)
        with torch.no_grad():
            ids.weight.add_(gradient, alpha=-0.05)
        self.assertFalse(torch.equal(first, ids.weight[3]))
        checkpoint.reset_held_out(3, 1)
        self.assertTrue(torch.equal(first, checkpoint.weight[3]))

    def test_observed_only_graph_ignores_missing_and_query_targets(self):
        support = torch.tensor([[0., 1., float("nan")], [1., 0., 0.]])
        query = torch.tensor([[1., float("nan"), 0.]])
        def edges(row):
            return connector.connect_task_and_sample(support, row, nan_w=0.)[4][0]
        edge = edges(query)
        self.assertEqual(set(edge[1, edge[0] == 5].tolist()), {2})
        query[0, 0] = 0.
        self.assertTrue(torch.equal(edge, edges(query)))
        masked = edges(torch.full_like(query, -1))
        self.assertFalse((masked == 5).any().item())
