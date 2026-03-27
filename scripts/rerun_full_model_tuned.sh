#!/bin/bash
# 重新运行完整模型 - 优化LLM超参数版本

#SBATCH --chdir=/home/zjhzjh/workdir/GSKLTA
#SBATCH --job-name=full_model_tuned
#SBATCH --partition=gpu-l20      # 请求的分区
#SBATCH --nodes=1                         # 请求一个节点
#SBATCH --ntasks=1                        # 请求一个任务
#SBATCH --cpus-per-task=8                # 每个任务使用一个 CPU
#SBATCH --gres=gpu:1

source /home/zjhzjh/miniconda3/etc/profile.d/conda.sh
conda activate pyg

GPU=0
DATASET="tox21"
N_SUPPORT=10

COMMON="--dataset ${DATASET} --n_support ${N_SUPPORT} --gpu ${GPU} \
--n_query 8 --episode 2000 --eval_step 100 --random_seed 123 \
--mol_num_layer 5 --emb_dim 300 --mol_gnn_type gin --mol_graph_pooling mean \
--inner_lr 0.05 --meta_lr 2e-4 --inner_update_step 3 --inner_tasks 10 \
--rel_layer 2 --rel_hidden_dim 128 --pool_num 20"

echo "=========================================="
echo "测试不同LLM Sampler强度的完整模型"
echo "=========================================="

# 配置1: 激进语义采样 (80% semantic)
echo "▶ Running: Full Model (Aggressive Semantic, 80%)"
python run.py ${COMMON} \
    --relation_backend krgts --use_knowledge_edges both --know_weight 0.4 \
    --use_semantic_sampler 1 \
    --semantic_temperature 0.05 \
    --semantic_similarity_weight 0.8 \
    --semantic_min_similarity 0.3 \
    --use_ltafm 1 --ltafm_hidden_dim 512 --ltafm_diversity_weight 0.005 \
    --exp_name full_aggressive --exp_id tuned_full_aggressive

# 配置2: 纯语义采样 (100% semantic)
echo "▶ Running: Full Model (Pure Semantic, 100%)"
python run.py ${COMMON} \
    --relation_backend krgts --use_knowledge_edges both --know_weight 0.4 \
    --use_semantic_sampler 1 \
    --semantic_temperature 0.05 \
    --semantic_similarity_weight 1.0 \
    --semantic_min_similarity 0.3 \
    --use_ltafm 1 --ltafm_hidden_dim 512 --ltafm_diversity_weight 0.005 \
    --exp_name full_pure_semantic --exp_id tuned_full_pure

# 配置3: 完全禁用LLM（验证KRGTS+L-TAFM就是最优）
echo "▶ Running: KRGTS+L-TAFM (No LLM)"
python run.py ${COMMON} \
    --relation_backend krgts --use_knowledge_edges both --know_weight 0.4 \
    --use_semantic_sampler 0 \
    --use_ltafm 1 --ltafm_hidden_dim 512 --ltafm_diversity_weight 0.005 \
    --exp_name krgts_ltafm_verified --exp_id tuned_krgts_ltafm

echo ""
echo "✅ 完成！对比三个配置的结果"
echo "如果Pure Semantic仍然不如KRGTS+L-TAFM，说明LLM模块确实不适合10-shot场景"
