#!/bin/bash
# =============================================================================
# 消融实验自动化脚本
# =============================================================================
# 用途：系统化地执行消融实验，评估每个创新点的独立贡献和协同效果
# 
# 实验矩阵：
#   Exp-1: Baseline (GS-Meta原始)
#   Exp-2: +KRGTS (知识边)
#   Exp-3: +LLM Sampler (语义采样)
#   Exp-4: +L-TAFM (特征调制)
#   Exp-5: KRGTS+LLM (知识+语义)
#   Exp-6: KRGTS+L-TAFM (知识+调制)
#   Exp-7: LLM+L-TAFM (语义+调制)
#   Exp-8: Full Model (完整模型)
# =============================================================================

#SBATCH --chdir=/home/zjhzjh/workdir/GSKLTA
#SBATCH --job-name=ablation           # 任务名称
#SBATCH --partition=gpu-l20      # 请求的分区
#SBATCH --nodes=1                         # 请求一个节点
#SBATCH --ntasks=1                        # 请求一个任务
#SBATCH --cpus-per-task=8                # 每个任务使用一个 CPU
#SBATCH --gres=gpu:1

source /home/zjhzjh/miniconda3/etc/profile.d/conda.sh
conda activate pyg

# =============================================================================
# 全局配置
# =============================================================================
GPU=0
DATASET="tox21"              # 可选: sider, tox21, muv, pcba
N_SUPPORT=10                 # Support set大小: 1 或 10
N_QUERY=8
EPISODE=2000
EVAL_STEP=100
RANDOM_SEED=0             # 固定随机种子确保可复现性

# 实验命名
EXP_NAME="ablation_${DATASET}_${N_SUPPORT}shot"

# =============================================================================
# 通用参数（所有实验共享）
# =============================================================================
COMMON_ARGS="--dataset ${DATASET} \
    --gpu ${GPU} \
    --random_seed ${RANDOM_SEED} \
    --n_support ${N_SUPPORT} \
    --n_query ${N_QUERY} \
    --episode ${EPISODE} \
    --eval_step ${EVAL_STEP} \
    --test_batch_size 128 \
    --mol_num_layer 5 \
    --emb_dim 300 \
    --mol_dropout 0.1 \
    --mol_graph_pooling mean \
    --mol_gnn_type gin \
    --rel_layer 2 \
    --rel_hidden_dim 128 \
    --rel_dropout 0.1 \
    --rel_pre_dropout 0.1 \
    --rel_norm 1 \
    --rel_top_k 1 \
    --num_relation_attr 3 \
    --inner_lr 0.005 \
    --meta_lr 2e-4 \
    --weight_decay 5e-5 \
    --inner_update_step 3 \
    --inner_tasks 10 \
    --pool_num 20 \
    --task_lr 5e-4 \
    --task_hid_dim 10 \
    --nce_t 0.08 \
    --contr_w 0.05"

# KRGTS相关参数（当启用时）
KRGTS_ARGS="--use_knowledge_edges both \
    --know_weight 0.4 \
    --know_temp 1.5 \
    --know_warmup 100 \
    --know_ramp 100 \
    --know_hold 0 \
    --know_decay 0"

# LLM语义采样参数（当启用时）
LLM_SAMPLER_ARGS="--semantic_temperature 0.1 \
    --semantic_similarity_weight 0.5 \
    --semantic_adjust_mode additive \
    --semantic_reward_bonus 0.1 \
    --semantic_warmup 0"

# L-TAFM特征调制参数（当启用时）
LTAFM_ARGS="--ltafm_hidden_dim 512 \
    --ltafm_gamma_init 0.0 \
    --ltafm_diversity_weight 0.005"

# =============================================================================
# 实验执行函数
# =============================================================================
run_experiment() {
    local exp_id=$1
    local relation_backend=$2
    local use_semantic=$3
    local use_ltafm=$4
    local additional_args=$5
    
    echo ""
    echo "=========================================="
    echo "Running: ${exp_name}"
    echo "Exp ID: ${exp_id}"
    echo "Config: relation_backend=${relation_backend}, semantic=${use_semantic}, ltafm=${use_ltafm}"
    echo "=========================================="
    
    python run.py ${COMMON_ARGS} \
        --exp_name ${EXP_NAME} \
        --exp_id ${exp_id} \
        --relation_backend ${relation_backend} \
        --use_semantic_sampler ${use_semantic} \
        --use_ltafm ${use_ltafm} \
        ${additional_args}
    
    echo "Completed: ${exp_name}"
    echo ""
}

# =============================================================================
# 开始消融实验
# =============================================================================
echo "=========================================="
echo "开始消融实验"
echo "数据集: ${DATASET}"
echo "Shot数: ${N_SUPPORT}"
echo "训练轮次: ${EPISODE}"
echo "随机种子: ${RANDOM_SEED}"
echo "=========================================="

# # -----------------------------------------------------------------------------
# # Exp-1: Baseline (GS-Meta 原始)
# # -----------------------------------------------------------------------------
# run_experiment \
#     "exp1_baseline" \
#     "baseline" \
#     0 \
#     0 \
#     ""

# # -----------------------------------------------------------------------------
# # Exp-2: +KRGTS (知识增强关系网络)
# # -----------------------------------------------------------------------------
# run_experiment \
#     "exp2_krgts" \
#     "krgts" \
#     0 \
#     0 \
#     "${KRGTS_ARGS}"

# # -----------------------------------------------------------------------------
# # Exp-3: +LLM Sampler (语义引导采样)
# # -----------------------------------------------------------------------------
# run_experiment \
#     "exp3_llm_sampler" \
#     "baseline" \
#     1 \
#     0 \
#     "${LLM_SAMPLER_ARGS}"

# # -----------------------------------------------------------------------------
# # Exp-4: +L-TAFM (任务自适应特征调制)
# # -----------------------------------------------------------------------------
# run_experiment \
#     "exp4_ltafm" \
#     "baseline" \
#     0 \
#     1 \
#     "${LTAFM_ARGS}"

# # -----------------------------------------------------------------------------
# # Exp-5: KRGTS + LLM Sampler (知识+语义)
# # -----------------------------------------------------------------------------
# run_experiment \
#     "exp5_krgts_llm" \
#     "krgts" \
#     1 \
#     0 \
#     "${KRGTS_ARGS} ${LLM_SAMPLER_ARGS}"

# # -----------------------------------------------------------------------------
# # Exp-6: KRGTS + L-TAFM (知识+调制)
# # -----------------------------------------------------------------------------
# run_experiment \
#     "exp6_krgts_ltafm" \
#     "krgts" \
#     0 \
#     1 \
#     "${KRGTS_ARGS} ${LTAFM_ARGS}"

# # -----------------------------------------------------------------------------
# # Exp-7: LLM Sampler + L-TAFM (语义+调制)
# # -----------------------------------------------------------------------------
# run_experiment \
#     "exp7_llm_ltafm" \
#     "baseline" \
#     1 \
#     1 \
#     "${LLM_SAMPLER_ARGS} ${LTAFM_ARGS}"

# -----------------------------------------------------------------------------
# Exp-8: Full Model (完整模型)
# -----------------------------------------------------------------------------
run_experiment \
    "pre_train" \
    "krgts" \
    1 \
    1 \
    "${KRGTS_ARGS} ${LLM_SAMPLER_ARGS} ${LTAFM_ARGS}"

# =============================================================================
# 实验完成
# =============================================================================
echo ""
echo "=========================================="
echo "消融实验全部完成！"
echo "=========================================="
echo ""
echo "结果保存在: logs/ablation_*/"
echo ""
echo "下一步："
echo "  1. 运行结果汇总脚本: python scripts/summarize_ablation_results.py"
echo "  2. 查看性能对比表格和可视化图表"
echo ""
