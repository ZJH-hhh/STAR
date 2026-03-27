# 消融实验完整指南

## 📋 目录
1. [实验设计原理](#实验设计原理)
2. [快速开始](#快速开始)
3. [详细实验配置](#详细实验配置)
4. [结果分析](#结果分析)
5. [论文撰写建议](#论文撰写建议)

---

## 实验设计原理

### 核心研究问题
**如何量化评估每个创新点的独立贡献和协同效果？**

本项目包含三个核心创新：
1. **KRGTS关系网络**: 使用HMRGNN + scaffold/group知识边替代原始关系网络
2. **LLM语义引导采样**: 利用LLM嵌入智能选择相关辅助任务
3. **L-TAFM特征调制**: 基于FiLM机制的任务自适应特征调制

### 消融实验矩阵

| 实验组 | KRGTS | LLM采样 | L-TAFM | 目的 |
|-------|-------|---------|--------|------|
| **Exp-1** | ❌ | ❌ | ❌ | **Baseline性能基线** |
| **Exp-2** | ✅ | ❌ | ❌ | 评估知识边独立贡献 |
| **Exp-3** | ❌ | ✅ | ❌ | 评估语义采样独立贡献 |
| **Exp-4** | ❌ | ❌ | ✅ | 评估特征调制独立贡献 |
| **Exp-5** | ✅ | ✅ | ❌ | 评估知识+语义协同 |
| **Exp-6** | ✅ | ❌ | ✅ | 评估知识+调制协同 |
| **Exp-7** | ❌ | ✅ | ✅ | 评估语义+调制协同 |
| **Exp-8** | ✅ | ✅ | ✅ | **完整模型最优性能** |

### 预期分析指标
- **独立贡献**: `Exp-2/3/4 - Exp-1` (单个组件提升)
- **协同增益**: `Exp-5/6/7 - (Exp-2+Exp-3)` (是否存在1+1>2)
- **完整模型优势**: `Exp-8 vs 其他所有配置`

---

## 🚀 快速开始

### 方式1: 自动化批量实验（推荐）

```bash
# 1. 确保环境激活
conda activate pyg

# 2. 提交SLURM任务（适合集群环境）
sbatch scripts/ablation_study.sh

# 或者直接运行（本地环境）
bash scripts/ablation_study.sh
```

**预计时间**: 约16-24小时（8个实验 × 2-3小时/实验）

### 方式2: 快速对比（Baseline vs Full）

```bash
# 只运行Exp-1和Exp-8对比，快速验证整体提升
bash scripts/run_ablation_quick.sh tox21 10 0
# 参数: 数据集 Shot数 GPU编号
```

### 方式3: 手动单独运行实验

```bash
# Exp-1: Baseline
python run.py --dataset tox21 --n_support 10 --episode 2000 \
    --relation_backend baseline \
    --use_semantic_sampler 0 --use_ltafm 0 \
    --exp_name exp1_baseline --exp_id ablation_tox21_10shot_exp1 \
    --gpu 0
```

---

## ⚙️ 详细实验配置

### 全局超参数（所有实验共享）

```bash
--dataset tox21              # 数据集
--n_support 10               # Support set大小
--episode 2000               # 训练轮次
--random_seed 123            # 固定随机种子
--mol_num_layer 5            # GNN层数
--emb_dim 300                # 嵌入维度
--inner_lr 0.05              # 内循环学习率
--meta_lr 2e-4               # 元学习率
```

### 各实验组特定配置

#### Exp-1: Baseline
```bash
--relation_backend baseline
--use_semantic_sampler 0
--use_ltafm 0
```

#### Exp-2: +KRGTS
```bash
--relation_backend krgts
--use_knowledge_edges both
--know_weight 0.4
--use_semantic_sampler 0
--use_ltafm 0
```

#### Exp-3: +LLM Sampler
```bash
--relation_backend baseline
--use_semantic_sampler 1
--semantic_temperature 0.1
--semantic_similarity_weight 0.5
--use_ltafm 0
```

#### Exp-4: +L-TAFM
```bash
--relation_backend baseline
--use_semantic_sampler 0
--use_ltafm 1
--ltafm_hidden_dim 512
--ltafm_diversity_weight 0.005
```

#### Exp-8: Full Model
```bash
--relation_backend krgts
--use_knowledge_edges both --know_weight 0.4
--use_semantic_sampler 1
--semantic_temperature 0.1 --semantic_similarity_weight 0.5
--use_ltafm 1
--ltafm_hidden_dim 512 --ltafm_diversity_weight 0.005
```

---

## 📊 结果分析

### 自动化结果汇总

```bash
python scripts/summarize_ablation_results.py \
    --log_dir logs/ablation_0305 \
    --dataset tox21 \
    --n_support 10 \
    --output_dir ablation_results
```

**输出内容**:
- ✅ `ablation_table.tex`: LaTeX格式表格
- ✅ `ablation_bar_chart.png`: 柱状图对比
- ✅ 终端统计摘要

### 关键指标计算

#### 1. 独立贡献（Δ）
```
Δ_KRGTS = AUROC(Exp-2) - AUROC(Exp-1)
Δ_LLM   = AUROC(Exp-3) - AUROC(Exp-1)
Δ_TAFM  = AUROC(Exp-4) - AUROC(Exp-1)
```

#### 2. 协同增益（Synergy）
```
Synergy(KRGTS+LLM) = AUROC(Exp-5) - [AUROC(Exp-2) + AUROC(Exp-3) - AUROC(Exp-1)]
```

- Synergy > 0：正协同（1+1>2）
- Synergy ≈ 0：独立作用
- Synergy < 0：负干扰

#### 3. 完整模型优势
```
Overall Gain = AUROC(Exp-8) - AUROC(Exp-1)
Relative Improvement = (AUROC(Exp-8) / AUROC(Exp-1) - 1) × 100%
```

---

## 📝 论文撰写建议

### 实验章节结构

```markdown
### 5.2 Ablation Study

**Table 1: Component Contribution Analysis**
[插入自动生成的LaTeX表格]

**关键发现**:
1. 单独组件贡献：
   - KRGTS: +X.XX% (知识边显著改善结构感知)
   - LLM Sampler: +X.XX% (语义引导过滤噪声任务)
   - L-TAFM: +X.XX% (任务自适应调制提升特征利用)

2. 协同效应分析：
   - KRGTS+LLM: +X.XX% (知识与语义互补)
   - 完整模型: +X.XX% (三组件协同达到最优)

3. 统计显著性：
   - 使用配对t检验验证完整模型vs Baseline (p < 0.01)
```

### 可视化建议

1. **Figure 1: 消融实验柱状图**
   - 展示8个配置的AUROC对比
   - 用颜色区分baseline、单组件、组合、完整模型

2. **Figure 2: 跨数据集泛化性**
   - 展示Tox21/SIDER/MUV上的消融结果
   - 验证创新点的普适性

### 结果讨论要点

1. **为什么L-TAFM有效？**
   - 不同化学属性需要关注分子不同子结构
   - FiLM调制使GNN能够根据任务动态调整特征提取策略

2. **为什么需要语义采样？**
   - 随机采样可能引入语义无关任务
   - LLM嵌入捕获化学属性的高层语义

3. **三组件如何协同？**
   - KRGTS提供结构知识
   - LLM Sampler过滤任务
   - L-TAFM任务感知
   - 三者形成闭环优化

---

## 🔧 高级技巧

### 1. 多数据集验证

修改脚本中的`DATASET`变量：

```bash
for dataset in tox21 sider muv; do
    # 修改 scripts/ablation_study.sh 中的 DATASET 变量
    # 然后运行
    bash scripts/ablation_study.sh
done
```

### 2. 不同Shot数对比

```bash
for n_support in 1 10; do
    bash scripts/run_ablation_quick.sh tox21 $n_support 0
done
```

---

## ⚠️ 故障排查

### 问题1: 日志文件未生成
**原因**: 实验可能崩溃或未完成  
**解决**: 检查SLURM输出或终端错误

### 问题2: 分数异常低（< 0.5）
**原因**: LLM嵌入文件缺失  
**解决**:
```bash
# 检查嵌入文件
ls data/embeddings/tox21/tox21_property_embeddings.pt

# 如果缺失，运行预计算脚本
python scripts/precompute_property_embeddings.py --dataset tox21
```

### 问题3: GPU内存不足
**解决**: 降低`--test_batch_size`或`--n_support`

---

## ✅ 检查清单

运行消融实验前，确认：

- [ ] 环境激活（`conda activate pyg`）
- [ ] LLM嵌入文件存在
- [ ] GPU可用（>=16GB显存）
- [ ] 固定随机种子
- [ ] 日志目录可写
- [ ] 至少16GB磁盘空间

---

## 📞 文件概览

- `scripts/ablation_study.sh` - 全自动8组实验
- `scripts/run_ablation_quick.sh` - 快速对比实验
- `scripts/summarize_ablation_results.py` - 结果汇总工具
- `ABLATION_STUDY_GUIDE.md` - 本指南

---

**祝实验顺利！🚀**
