import csv
import os

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch, DataLoader
import numpy as np

from models import (
    MAML, GSMeta, TaskSelector, NCESoftmaxLoss,
)
from models.semantic_sampler import SemanticGuidedSampler, create_semantic_sampler
from dataset import FewshotMolDataset, dataset_sampler
from models.knowledge_edges import compute_knowledge_adj

from sklearn.metrics import roc_auc_score

import logging
import random
from tqdm import tqdm
from copy import deepcopy

logger = logging.getLogger()


class MovingAVG:
    def __init__(self):
        self.count = 0
        self.avg = 0

    def get_avg(self):
        return self.avg

    def update(self, x):
        self.count += 1
        self.avg = self.avg + (x - self.avg) / self.count


class MetaLearner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if int(args.gpu) >= 0 else 'cpu')
        need_knowledge = (args.use_knowledge_edges != 'none') or (args.relation_backend == 'krgts')
        self.dataset = FewshotMolDataset(root=args.data_root, name=args.dataset,
                                         compute_knowledge=need_knowledge)
        self.train_task_range, self.test_task_range = self.dataset.train_task_range, self.dataset.test_task_range
        self.task_sampling = getattr(args, 'task_sampling', 'random')
        model = GSMeta(task_num=self.dataset.total_tasks,
                        train_task_num=self.dataset.n_task_train,
                        args=args).to(self.device)
        self.maml = MAML(model, lr=args.inner_lr, first_order=not args.second_order, anil=False, allow_unused=True)
        # 预训练
        # self.maml = MAML(model, lr=args.inner_lr, first_order=not args.second_order, anil=False, allow_unused=True, allow_nograd=True)
        # 预训练
        opt_params = list(self.maml.parameters())
        # 处理类别不平衡
        # 如果数据不平衡，可以设置pos_weight
        self.cls_criterion = nn.BCEWithLogitsLoss()  # pos_weight可根据数据分布调整

        self.n_support, self.n_query = args.n_support, args.n_query
        self.inner_update_step = args.inner_update_step

        if self.args.train_auxi_task_num is None:
            self.train_auxi_task_num = len(self.train_task_range) - 1
        else:
            self.train_auxi_task_num = min(args.train_auxi_task_num, len(self.train_task_range) - 1)
        if self.args.test_auxi_task_num is None:
            self.test_auxi_task_num = len(self.train_task_range)
        else:
            self.test_auxi_task_num = min(args.test_auxi_task_num, len(self.train_task_range))

        selector_in_dim = args.emb_dim if getattr(args, 'relation_backend', 'baseline') == 'baseline' \
            else args.rel_hidden_dim
        self.task_selector = TaskSelector(input_size=selector_in_dim,
                                          hidden_size=args.task_hid_dim,
                                          t=args.task_t).to(self.device)

        self.task_opt = optim.AdamW(self.task_selector.parameters(), lr=args.task_lr, weight_decay=args.weight_decay)
        self.task_reward_avg = MovingAVG()
        self.nce_loss = NCESoftmaxLoss(t=args.nce_t)
        self.args.pool_num = min(self.args.pool_num, len(self.train_task_range))
        
        # Semantic-Guided Task Sampling
        self.use_semantic_sampler = bool(getattr(args, 'use_semantic_sampler', False))
        self.semantic_warmup = int(getattr(args, 'semantic_warmup', 0))
        self.semantic_reward_bonus = float(getattr(args, 'semantic_reward_bonus', 0.1))
        
        if self.use_semantic_sampler:
            self.semantic_sampler = create_semantic_sampler(args)
            if self.semantic_sampler is not None:
                # Build property name to index mapping from dataset
                self.property_names = self.dataset.task_names if hasattr(self.dataset, 'task_names') else None
                logger.info(f"SemanticGuidedSampler enabled with weight={args.semantic_similarity_weight}")
            else:
                self.use_semantic_sampler = False
                logger.warning("SemanticGuidedSampler disabled: embedding file not found")
        else:
            self.semantic_sampler = None
        
        self.opt = optim.AdamW(opt_params, lr=args.meta_lr, weight_decay=args.weight_decay)

    def save_checkpoint(self, save_dir, epoch, score, is_best=False):
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            "epoch": int(epoch),
            "score": float(score),
            "is_best": bool(is_best),
            "model_state_dict": self.maml.module.state_dict(),
            "task_selector_state_dict": self.task_selector.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "task_optimizer_state_dict": self.task_opt.state_dict(),
            "args": vars(self.args),
            "train_task_range": list(self.train_task_range),
            "test_task_range": list(self.test_task_range),
        }
        path = os.path.join(save_dir, "best_model.pt" if is_best else "checkpoint.pt")
        torch.save(checkpoint, path)
        return path

    def _get_task_names(self):
        path = getattr(self.args, "semantic_embedding_path", "")
        if path and os.path.exists(path):
            try:
                emb_data = torch.load(path, map_location="cpu")
                names = emb_data.get("sorted_names", emb_data.get("property_names"))
                if names and len(names) == self.dataset.total_tasks:
                    return list(names)
            except Exception as exc:
                logger.warning(f"Could not read task names from semantic embeddings: {exc}")

        try:
            with open(self.dataset.raw_paths[0], newline="") as fin:
                header = next(csv.reader(fin))
            y_idx = self.dataset.names[self.dataset.name][4]
            return header[y_idx]
        except Exception as exc:
            logger.warning(f"Could not infer task names from raw dataset header: {exc}")
            return [f"task_{idx}" for idx in range(self.dataset.total_tasks)]

    def export_ltafm_modulation(self, save_dir, epoch, score=None, is_best=False):
        model = self.maml.module
        if not getattr(model, "use_ltafm", False) or model.ltafm is None:
            return None

        out_dir = os.path.join(save_dir, "ltafm_mechanistic")
        os.makedirs(out_dir, exist_ok=True)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            semantic_embeddings = model.semantic_embeddings.detach()
            modulation = model.ltafm.film.mlp(semantic_embeddings)
            gamma, beta = torch.chunk(modulation, 2, dim=-1)
            gamma = F.softplus(gamma) + 1e-8

            gamma_cpu = gamma.detach().cpu()
            beta_cpu = beta.detach().cpu()
            task_names = self._get_task_names()
            if len(task_names) != gamma_cpu.shape[0]:
                logger.warning(
                    "Task-name count does not match L-TAFM gamma rows "
                    f"({len(task_names)} vs {gamma_cpu.shape[0]}). Falling back to task ids."
                )
                task_names = [f"task_{idx}" for idx in range(gamma_cpu.shape[0])]
            rows = []
            for task_id, task_name in enumerate(task_names):
                gamma_t = gamma_cpu[task_id]
                beta_t = beta_cpu[task_id]
                rows.append({
                    "episode": int(epoch),
                    "task_id": int(task_id),
                    "task_name": task_name,
                    "pathway": "NR" if str(task_name).startswith("NR-") else "SR",
                    "gamma_l2": float(torch.norm(gamma_t - 1.0, p=2).item()),
                    "gamma_abs_mean": float(torch.mean(torch.abs(gamma_t - 1.0)).item()),
                    "gamma_mean": float(torch.mean(gamma_t).item()),
                    "gamma_std": float(torch.std(gamma_t, unbiased=False).item()),
                    "gamma_var": float(torch.var(gamma_t, unbiased=False).item()),
                    "gamma_min": float(torch.min(gamma_t).item()),
                    "gamma_max": float(torch.max(gamma_t).item()),
                    "beta_l2": float(torch.norm(beta_t, p=2).item()),
                    "beta_abs_mean": float(torch.mean(torch.abs(beta_t)).item()),
                    "beta_mean": float(torch.mean(beta_t).item()),
                    "beta_std": float(torch.std(beta_t, unbiased=False).item()),
                    "score": "" if score is None else float(score),
                    "is_best": bool(is_best),
                })

            snapshot = {
                "episode": int(epoch),
                "score": None if score is None else float(score),
                "is_best": bool(is_best),
                "task_names": task_names,
                "semantic_embedding_path": getattr(self.args, "semantic_embedding_path", ""),
                "gamma": gamma_cpu,
                "beta": beta_cpu,
                "stats": rows,
            }

        if was_training:
            model.train()

        snapshot_path = os.path.join(out_dir, f"gamma_beta_epoch_{int(epoch):04d}.pt")
        latest_path = os.path.join(out_dir, "gamma_beta_latest.pt")
        torch.save(snapshot, snapshot_path)
        torch.save(snapshot, latest_path)
        if is_best:
            torch.save(snapshot, os.path.join(out_dir, "gamma_beta_best.pt"))

        trajectory_path = os.path.join(out_dir, "gamma_trajectory.csv")
        fieldnames = list(rows[0].keys()) if rows else []
        write_header = not os.path.exists(trajectory_path)
        with open(trajectory_path, "a", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

        return {
            "snapshot_path": snapshot_path,
            "latest_path": latest_path,
            "trajectory_path": trajectory_path,
            "best_path": os.path.join(out_dir, "gamma_beta_best.pt") if is_best else None,
        }

    @staticmethod
    def _scheduled_weight(epoch, warm, ramp, hold, decay, max_w):
        max_w = float(max_w)
        if epoch is None or max_w <= 0.0:
            return 0.0
        if epoch <= warm:
            return 0.0
        if epoch <= warm + ramp:
            return max_w * float(epoch - warm) / float(max(1, ramp))
        if epoch <= warm + ramp + hold or decay == 0:
            return max_w
        t = epoch - (warm + ramp + hold)
        if t >= decay:
            return 0.0
        return max_w * (1.0 - float(t) / float(max(1, decay)))

    def _compute_know_weight(self, epoch):
        return self._scheduled_weight(
            epoch,
            max(0, int(getattr(self.args, 'know_warmup', 0))),
            max(1, int(getattr(self.args, 'know_ramp', 1))),
            max(0, int(getattr(self.args, 'know_hold', 0))),
            max(0, int(getattr(self.args, 'know_decay', 0))),
            float(getattr(self.args, 'know_weight', 0.0)),
        )

    def _knowledge_mode(self):
        if self.args.relation_backend == 'krgts' and self.args.use_knowledge_edges == 'none':
            return 'both'
        return self.args.use_knowledge_edges

    def _build_knowledge(self, s_list, q_list):
        mode = self._knowledge_mode()
        if mode == 'none':
            return None
        split = self.args.relation_backend == 'krgts'
        return compute_knowledge_adj(
            s_list,
            q_list,
            which=mode,
            top_k=self.args.rel_top_k,
            device=self.device,
            split_components=split,
        )

    @staticmethod
    def _attach_knowledge(batch, know):
        if know is None:
            return
        if isinstance(know, tuple):
            sca, gro = know
            batch.__setattr__('know_sca', sca)
            batch.__setattr__('know_gro', gro)
        else:
            batch.__setattr__('know_adj', know)

    def update_inner(self, s_data, q_data, task_id, auxi_tasks, epoch):
        sampled_task = torch.tensor([task_id] + auxi_tasks).to(self.device)
        s_y, q_y = s_data.y[:, sampled_task], q_data.y[:, sampled_task]
        model = self.maml.clone()
        model.train()
        know_w = self._compute_know_weight(epoch)
        
        for _ in range(self.args.inner_update_step):
            result = model(
                s_data, q_data, s_y, q_y, sampled_task, know_w=know_w, know_temp=self.args.know_temp,
            )
            s_logit, q_logit, s_label, q_label, graph_f = result[:5]
            inner_loss = self.cls_criterion(s_logit, s_label)
            model.adapt(inner_loss)
        
        result = model(
            s_data, q_data, s_y, q_y, sampled_task, know_w=know_w, know_temp=self.args.know_temp,
        )
        s_logit, q_logit, s_label, q_label, graph_f, s_feat, q_feat, diversity_loss = result
        
        eval_loss = self.cls_criterion(q_logit, q_label)
        
        return eval_loss, graph_f, diversity_loss

    def train_step(self, epoch):
        selected_ids, selected_tasks, selected_prob = self.sample_tasks(epoch)
        eval_losses = []
        diversity_losses = []
        graph_f1s, graph_f2s = [], []
        
        for task_id, (s_data1, q_data1, s_data2, q_data2) in zip(selected_ids, selected_tasks):
            auxi_tasks = self.sample_auxiliary(task_id, self.train_task_range, self.train_auxi_task_num, epoch=epoch)
            eval_loss1, graph_f1, div_loss1 = self.update_inner(s_data1, q_data1, task_id, auxi_tasks, epoch)
            eval_loss2, graph_f2, div_loss2 = self.update_inner(s_data2, q_data2, task_id, auxi_tasks, epoch)
            eval_losses += [eval_loss1, eval_loss2]
            graph_f1s.append(graph_f1)
            graph_f2s.append(graph_f2)
            if div_loss1 is not None:
                diversity_losses.append(div_loss1)
            if div_loss2 is not None:
                diversity_losses.append(div_loss2)

        # tgt_f
        tgt_f1, tgt_f2 = torch.vstack(graph_f1s), torch.vstack(graph_f2s)
        loss_contr = self.nce_loss(tgt_f1, tgt_f2)

        loss_cls = torch.stack(eval_losses).mean()
        loss_div = torch.stack(diversity_losses).mean() if diversity_losses else loss_cls.new_zeros(())
        self.opt.zero_grad()
        total_loss = loss_cls + loss_contr * self.args.contr_w + loss_div
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.maml.parameters(), 1)
        self.opt.step()

        # update task selector:
        loss_task = -torch.log(selected_prob).sum()
        reward = loss_contr.item()
        
        # Semantic-guided reward bonus: reward selecting semantically relevant tasks
        if (self.semantic_sampler is not None and 
            epoch > self.semantic_warmup and 
            self.semantic_reward_bonus > 0):
            # Compute mean semantic similarity of selected tasks
            # For simplicity, use pairwise similarity within selected tasks
            if len(selected_ids) > 1:
                selected_tensor = torch.tensor(selected_ids, dtype=torch.long)
                # Get pairwise similarities
                pair_sims = []
                for i, tid in enumerate(selected_ids):
                    others = torch.tensor([sid for j, sid in enumerate(selected_ids) if j != i], dtype=torch.long)
                    if len(others) > 0:
                        sims = self.semantic_sampler.get_task_similarities(tid, others)
                        pair_sims.append(sims.mean().item())
                if pair_sims:
                    mean_semantic_sim = np.mean(pair_sims)
                    # Add bonus for high semantic coherence (negative of contrastive loss style)
                    reward = reward - self.semantic_reward_bonus * (1.0 - mean_semantic_sim)
        
        loss_task *= (reward - self.task_reward_avg.get_avg())
        self.task_reward_avg.update(reward)
        self.task_opt.zero_grad()
        loss_task.backward()
        torch.nn.utils.clip_grad_norm_(self.task_selector.parameters(), 1)
        self.task_opt.step()

        return {
            'loss_cls': loss_cls.item(),
            'loss_contr': loss_contr.item(),
            'loss_div': loss_div.item(),
            'loss_total': total_loss.item(),
        }

    def test_step(self, test_auxi_task_num=None):
        auc_scores = []
        
        for task_i in tqdm(self.test_task_range, desc='eval'):
            s_data, q_data = dataset_sampler(self.dataset, self.n_support, self.n_query,
                                             tgt_id=task_i, inductive=True,
                                             sampling=self.task_sampling)
            s_list = s_data
            s_data = Batch.from_data_list(s_list).to(self.device)
            test_auxi_task_num = self.test_auxi_task_num if test_auxi_task_num is None else test_auxi_task_num
            auxi_tasks = self.sample_auxiliary(task_i, self.train_task_range, test_auxi_task_num)
            sampled_task = torch.tensor([task_i] + auxi_tasks).to(self.device)
            s_y = s_data.y[:, sampled_task]
            model = self.maml.clone()
            model.train()
            # inner update
            adapt_q_iter = iter(DataLoader(q_data, batch_size=self.args.n_query, shuffle=True))
            for _ in range(self.args.inner_update_step):
                adapt_q_data = next(adapt_q_iter)
                know = self._build_knowledge(s_list, adapt_q_data.to_data_list())
                self._attach_knowledge(adapt_q_data, know)
                adapt_q_data = adapt_q_data.to(self.device)
                adapt_q_y = adapt_q_data.y[:, sampled_task]
                result = model(
                    s_data, adapt_q_data, s_y, adapt_q_y, sampled_task,
                    know_w=float(self.args.know_weight), know_temp=self.args.know_temp,
                )
                s_logit, q_logit, s_label, q_label = result[:4]
                inner_loss = self.cls_criterion(s_logit, s_label)
                model.adapt(inner_loss)
            model.eval()

            y_pred, y_true = [], []
            query_loader = DataLoader(q_data, batch_size=self.args.test_batch_size, num_workers=2, shuffle=False)
            with torch.no_grad():
                for iter_q_data in query_loader:
                    know = self._build_knowledge(s_list, iter_q_data.to_data_list())
                    self._attach_knowledge(iter_q_data, know)
                    iter_q_data = iter_q_data.to(self.device)
                    iter_q_y = iter_q_data.y[:, sampled_task]
                    result = model(
                        s_data, iter_q_data, s_y, iter_q_y, sampled_task,
                        know_w=float(self.args.know_weight), know_temp=self.args.know_temp,
                    )
                    _, q_logit, _, q_label, _, s_feat, q_feat, _ = result
                    q_logit = torch.sigmoid(q_logit).cpu().view(-1)
                    q_label = q_label.cpu().view(-1)
                    y_pred.append(q_logit)
                    y_true.append(q_label)

                y_true = torch.cat(y_true, dim=0).numpy()
                y_pred = torch.cat(y_pred, dim=0).numpy()
                score = roc_auc_score(y_true, y_pred)
                auc_scores.append(score)
        return np.mean(auc_scores)

    def sample_tasks(self, epoch):
        def sample_data(tgt_id):
            s_list1, q_list1 = dataset_sampler(
                self.dataset, self.n_support, self.n_query, tgt_id, sampling=self.task_sampling
            )
            s_list2, q_list2 = dataset_sampler(
                self.dataset, self.n_support, self.n_query, tgt_id, sampling=self.task_sampling
            )
            # build batches
            s_data1 = Batch.from_data_list(s_list1).to(self.device)
            q_data1 = Batch.from_data_list(q_list1).to(self.device)
            s_data2 = Batch.from_data_list(s_list2).to(self.device)
            q_data2 = Batch.from_data_list(q_list2).to(self.device)
            know1 = self._build_knowledge(s_list1, q_list1)
            know2 = self._build_knowledge(s_list2, q_list2)
            self._attach_knowledge(q_data1, know1)
            self._attach_knowledge(q_data2, know2)
            return s_data1, q_data1, s_data2, q_data2


        model = self.maml.clone()
        model.eval()
        tasks_pool = []
        graph_fs = []
        # pool_num must be an integer; cast defensively to avoid accidental floats
        tasks_pool_ids = np.random.choice(self.train_task_range, int(self.args.pool_num), replace=False)
        for task_id in tasks_pool_ids:
            s_data1, q_data1, s_data2, q_data2 = sample_data(task_id)
            tasks_pool.append((s_data1, q_data1, s_data2, q_data2))
            with torch.no_grad():
                sampled_task = torch.tensor([task_id]).to(self.device)
                sy1, qy1 = s_data1.y[:, sampled_task], q_data1.y[:, sampled_task]
                sy2, qy2 = s_data2.y[:, sampled_task], q_data2.y[:, sampled_task]
                kw = self._compute_know_weight(epoch)
                result1 = model(s_data1, q_data1, sy1, qy1, sampled_task,
                               know_w=kw, know_temp=self.args.know_temp)
                result2 = model(s_data2, q_data2, sy2, qy2, sampled_task,
                               know_w=kw, know_temp=self.args.know_temp)
                graph_f1 = result1[4].detach()
                graph_f2 = result2[4].detach()
                graph_fs += [graph_f1, graph_f2]
        graph_fs = torch.stack(graph_fs)
        w = self.task_selector(graph_fs, epoch)  # [n_pool*2]
        w = w.reshape(-1, 2).sum(-1)  # [n_pool]
        sample_size = min(self.args.inner_tasks // 2, len(w))
        sample_size = max(sample_size, 1)
        selected_index = self.task_selector.sample(w.cpu().tolist(), sample_size)
        selected_prob = w[selected_index]
        selected_tasks, selected_ids = [], []
        for idx in selected_index:
            selected_tasks.append(tasks_pool[idx])
            selected_ids.append(tasks_pool_ids[idx])

        return selected_ids, selected_tasks, selected_prob

    def sample_auxiliary(self, tgt_task_id, auxi_task_range, auxi_task_num, epoch=None):
        if tgt_task_id in auxi_task_range:
            auxi_task_range = deepcopy(auxi_task_range)
            auxi_task_range.remove(tgt_task_id)
        
        # Semantic-guided sampling: prefer tasks semantically similar to target
        if (self.semantic_sampler is not None and 
            epoch is not None and 
            epoch > self.semantic_warmup):
            
            candidate_indices = torch.tensor(auxi_task_range, dtype=torch.long)
            
            # Get semantic similarities between target and candidates
            similarities = self.semantic_sampler.get_task_similarities(
                tgt_task_id, candidate_indices
            )
            
            # Convert to sampling probabilities (softmax with temperature)
            temp = self.args.semantic_temperature
            probs = torch.softmax(similarities / temp, dim=0).cpu().numpy()
            
            # Mix with uniform distribution for exploration
            mix_weight = self.args.semantic_similarity_weight
            uniform = np.ones_like(probs) / len(probs)
            mixed_probs = mix_weight * probs + (1 - mix_weight) * uniform
            mixed_probs = mixed_probs / mixed_probs.sum()  # Normalize
            
            # Sample according to mixed probabilities
            selected_ids = np.random.choice(
                auxi_task_range, 
                size=min(auxi_task_num, len(auxi_task_range)), 
                replace=False, 
                p=mixed_probs
            ).tolist()
        else:
            # Default: uniform random sampling
            selected_ids = np.random.choice(auxi_task_range, auxi_task_num, replace=False).tolist()
        
        return selected_ids
