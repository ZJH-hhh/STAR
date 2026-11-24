import os
import warnings
# Suppress noisy 3rd‑party warnings as early as possible (before heavy imports)
warnings.filterwarnings(
    "ignore",
    message=r"Gym has been unmaintained since 2022*",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from args_parser import args_parser
from explight import initialize_exp, set_seed, get_dump_path, describe_model
import logging
from tqdm import tqdm
import time
logger = logging.getLogger()



class Runner:
    def __init__(self, args, logger_path, writer):
        from meta_learner import MetaLearner
        self.args = args
        self.writer = writer
        self.meta_learner = MetaLearner(args)
        describe_model(self.meta_learner.maml.module, logger_path, name='model')
        describe_model(self.meta_learner.task_selector, logger_path, name='task_selector')
        self.logger_path = logger_path

    def run(self):
        if not self.args.attribution_episodes and not os.path.isfile(self.args.eval_manifest):
            raise FileNotFoundError(f"Missing revision evaluation manifest: {self.args.eval_manifest}")
        if self.args.eval_checkpoint:
            checkpoint = torch.load(self.args.eval_checkpoint, map_location=self.meta_learner.device)
            if checkpoint["epoch"] != self.args.episode:
                raise ValueError("Checkpoint epoch differs from the requested endpoint")
            for field in ("dataset", "n_support", "random_seed", "relation_backend", "use_ltafm",
                          "use_semantic_sampler", "use_knowledge_edges", "rel_top_k"):
                if checkpoint["args"][field] != getattr(self.args, field):
                    raise ValueError("Checkpoint configuration mismatch: " + field)
            saved_args = checkpoint["args"]
            saved_count = checkpoint.get("effective_train_auxi_task_num", saved_args.get("train_auxi_task_num"))
            if saved_count is not None:
                if self.args.train_auxi_task_num not in (None, saved_count):
                    raise ValueError("Training auxiliary assay count differs from the checkpoint")
                self.meta_learner.train_auxi_task_num = saved_count
            if saved_args.get("conditioning", "text") != self.args.conditioning:
                raise ValueError("Checkpoint conditioning differs from the requested model")
            if self.args.conditioning == "support_adapted_id" and saved_args.get("id_dim") != self.args.id_dim:
                raise ValueError("Checkpoint task-ID width differs from the requested width")
            if saved_args.get("rel_nan_w", 1.0) != self.args.rel_nan_w:
                raise ValueError("Select the checkpoint's missing-edge setting with --rel_nan_w")
            if self.args.use_ltafm and self.args.conditioning == "text":
                saved = checkpoint["model_state_dict"]["semantic_embeddings"]
                current = self.meta_learner.maml.module.semantic_embeddings
                if not torch.equal(saved, current):
                    raise ValueError("The selected assay embeddings differ from the checkpoint")
            for key, tasks in (("train_task_range", self.meta_learner.train_task_range),
                               ("test_task_range", self.meta_learner.test_task_range)):
                if list(checkpoint[key]) != list(tasks):
                    raise ValueError("Checkpoint task split differs from the dataset")
            self.meta_learner.maml.module.load_state_dict(checkpoint["model_state_dict"])
            if self.args.attribution_episodes:
                from attribution import run_attribution
                run_attribution(self.meta_learner, self.args.attribution_episodes, self.logger_path)
                return
            self.meta_learner.test_step(self.logger_path)
            return
        pbar = tqdm(range(1, self.args.episode + 1))
        cost_time_ls = []

        for epoch in pbar:
            start = time.time()
            loss_stats = self.meta_learner.train_step(epoch)
            cost_time = time.time() - start
            cost_time_ls.append(cost_time)
            self.writer.add_scalar('loss-cls', loss_stats['loss_cls'], epoch)
            self.writer.add_scalar('loss-contr', loss_stats['loss_contr'], epoch)
            self.writer.add_scalar('loss-div', loss_stats['loss_div'], epoch)
            self.writer.add_scalar('loss-total', loss_stats['loss_total'], epoch)

            pbar.set_description(f"loss={loss_stats['loss_total']:.4f}")
            
            # 每 50 步清理一次显存碎片，防止 OOM
            if epoch % 50 == 0:
                torch.cuda.empty_cache()
            
        path = self.meta_learner.save_checkpoint(self.logger_path, epoch=epoch, score=None)
        logger.info(f"Saved fixed-endpoint checkpoint: {path}")
        score = self.meta_learner.test_step(self.logger_path)
        self.writer.add_scalar("endpoint/auroc", score, epoch)
        self.meta_learner.export_ltafm_modulation(self.logger_path, epoch=epoch, score=score)
        logger.info(f"Fixed-endpoint AUROC: {score:.5f}")
        logger.info(f"time cost: {np.mean(cost_time_ls):.5f}s")



def main():
    args = args_parser()
    # Capture and filter noisy third-party warnings (e.g., Gym / NumPy compatibility)
    logging.captureWarnings(True)
    if int(args.gpu) >= 0:
        torch.cuda.set_device(int(args.gpu))
    set_seed(args.random_seed)
    logger = initialize_exp(args)
    logger_path = get_dump_path(args)
    writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard'))

    runner = Runner(args, logger_path, writer)
    runner.run()
    writer.close()


if __name__ == '__main__':
    main()
