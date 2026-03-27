#!/bin/bash
# =============================================================================
# 快速启动消融实验 - 单个数据集
# =============================================================================
# 使用方法:
#   bash scripts/run_ablation_quick.sh tox21 10 0
#   参数1: 数据集名称 (tox21/sider/muv/pcba)
#   参数2: Shot数 (1/10)
#   参数3: GPU编号 (0/1/2/3)
# =============================================================================

# 参数解析
DATASET=${1:-"tox21"}
N_SUPPORT=${2:-10}
GPU=${3:-0}

echo "=========================================="
echo "快速消融实验启动"
echo "数据集: ${DATASET}"
echo "Shot数: ${N_SUPPORT}"
echo "GPU: ${GPU}"
echo "=========================================="

# 激活环境
source /home/zjhzjh/miniconda3/etc/profile.d/conda.sh
conda activate pyg

# 基础参数
COMMON="--dataset ${DATASET} --n_support ${N_SUPPORT} --gpu ${GPU} \
--n_query 8 --episode 2000 --eval_step 100 --random_seed 123 \
--mol_num_layer 5 --emb_dim 300 --mol_gnn_type gin \
--inner_lr 0.05 --meta_lr 2e-4 --inner_update_step 3"

EXP_PREFIX="quick_abl_${DATASET}_${N_SUPPORT}shot"

# Exp-1: Baseline
echo "▶ Running Exp-1: Baseline..."
python run.py ${COMMON} \
    --relation_backend baseline \
    --use_semantic_sampler 0 --use_ltafm 0 \
    --exp_name exp1_baseline --exp_id ${EXP_PREFIX}_exp1

# Exp-8: Full Model (对比最优配置)
echo "▶ Running Exp-8: Full Model..."
python run.py ${COMMON} \
    --relation_backend krgts --use_knowledge_edges both --know_weight 0.4 \
    --use_semantic_sampler 1 --semantic_temperature 0.1 --semantic_similarity_weight 0.5 \
    --use_ltafm 1 --ltafm_hidden_dim 512 --ltafm_diversity_weight 0.005 \
    --exp_name exp8_full --exp_id ${EXP_PREFIX}_exp8

echo ""
echo "✅ 快速对比实验完成！"
echo "查看结果:"
echo "  Baseline: logs/exp1_baseline/${EXP_PREFIX}_exp1/train.log"
echo "  Full:     logs/exp8_full/${EXP_PREFIX}_exp8/train.log"
