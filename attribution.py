"""Atom attribution on the specified 32-molecule Tox21 panel (Supplement S15)."""
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch


ASSAYS = (1, 4, 10)  # Tox21 S26: NR-AR-LBD, NR-ER, SR-MMP


def rank_top(scores):
    k = max(1, math.ceil(.2 * len(scores)))
    return sorted(range(len(scores)), key=lambda atom: (-scores[atom], atom))[:k]


def intervene(layer, query_x, logit_fn, label, random_seed):
    """Capture or zero query node representations; support calls pass through."""
    captured = []
    selected = None

    def query_hook(module, inputs, output):
        if inputs[0] is not query_x:
            return
        if selected is None:
            captured.append(output)
            return
        perturbed = output.clone()
        perturbed[selected] = 0
        return perturbed

    hook = layer.register_forward_hook(query_hook)
    try:
        sign = 2 * int(label) - 1
        with torch.enable_grad():
            margin = sign * logit_fn().reshape(())
            if len(captured) != 1:
                raise RuntimeError("Expected exactly one final query node representation")
            nodes = captured.pop()
            grad = torch.autograd.grad(margin, nodes)[0]
            scores = torch.abs((grad * nodes).sum(dim=1)).detach().cpu().tolist()
        baseline = float(margin.detach())
        if not scores or not all(math.isfinite(x) for x in scores):
            raise ValueError("Invalid atom importance")
        with torch.no_grad():
            selected = rank_top(scores)
            top_atoms = list(selected)
            top_drop = baseline - float(sign * logit_fn())
            rng = random.Random(random_seed)
            random_drops = []
            for _ in range(10):
                selected = rng.sample(range(len(scores)), len(top_atoms))
                random_drops.append(baseline - float(sign * logit_fn()))
        return dict(importance=scores, top_atoms=top_atoms, margin=baseline, top_drop=top_drop,
                    random_drops=random_drops, random_mask_seed=random_seed)
    finally:
        hook.remove()


def run_attribution(learner, path, output_dir):
    from torch_geometric.data import Batch
    args = learner.args
    if (args.dataset != "tox21" or args.episode != 2000 or args.inner_update_step != 3
            or args.random_seed not in (0, 1, 2)):
        raise ValueError("S15 requires Tox21, a 2,000-episode checkpoint, three adaptation steps and seed 0/1/2")
    with open(path, encoding="utf-8-sig") as stream:
        contexts = [json.loads(line) for line in stream if line.strip()]
    if len(contexts) != 3 or {c["task_id"] for c in contexts} != set(ASSAYS):
        raise ValueError("Supply one context for each Tox21 task ID 1, 4 and 10")
    panel = set(contexts[0]["attribution_indices"])
    if len(panel) != 32:
        raise ValueError("Supply the 32 original selected molecule indices")
    rows = []
    assay_labels = {mol: {} for mol in panel}
    for context in contexts:
        task = context["task_id"]
        support = context["support_indices"]
        query = context["query_indices"]
        if (set(context["attribution_indices"]) != panel or len(context["attribution_indices"]) != 32
                or len(set(query)) != len(query) or len(set(support)) != 2 * args.n_support
                or len(support) != 2 * args.n_support or set(support).intersection(query)
                or not panel.issubset(query)):
            raise ValueError("Mismatched attribution panel or invalid support/query sets")
        support_labels = learner.dataset.y_matrix[support, task].tolist()
        if Counter(support_labels) != {0: args.n_support, 1: args.n_support}:
            raise ValueError("Attribution context requires balanced labeled supports")
        model, s_list, s_data, s_y, sampled_task, _ = learner.prepare_eval_episode(context)
        signature = hashlib.sha256(json.dumps([support, query, context["support_seed"],
                                               sampled_task.tolist()]).encode()).hexdigest()
        layer = model.module.mol_encoder.gnn
        for mol in sorted(panel):
            graph = learner.dataset[mol]
            batch = Batch.from_data_list([graph]).to(learner.device)
            activity = batch.y[0, task].item()
            if activity not in (0, 1):
                raise ValueError("Selected molecule lacks a binary activity label")
            label = int(activity)
            assay_labels[mol][task] = label
            learner._attach_knowledge(batch, learner._build_knowledge(s_list, [graph]))
            q_y = learner._eval_labels(batch, sampled_task)

            def logit():
                return model(s_data, batch, s_y, q_y, sampled_task,
                             know_w=float(args.know_weight), know_temp=args.know_temp)[1]

            # Reuse identical random-control atom sets across training seeds.
            mask_seed = int.from_bytes(hashlib.sha256(f"{mol}:{task}".encode()).digest()[:8], "big")
            record = intervene(layer, batch.x, logit, label, mask_seed)
            rows.append(dict(task_id=task, training_seed=args.random_seed, molecule_id=mol,
                             label=label, context_sha256=signature, **record))
    patterns = Counter(tuple(labels[t] for t in ASSAYS) for labels in assay_labels.values())
    if len(patterns) != 8 or set(patterns.values()) != {4}:
        raise ValueError("The 32 source molecules must include four per three-assay activity pattern")
    with (Path(output_dir)/"attribution.jsonl").open("w", encoding="utf-8") as stream:
        for record in rows:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
