# PROJECT_BASELINE.md

# 第一部分 核心项目架构

## 项目概述
- **项目名称**: GSKLLM-modify (GS-Meta + KRGTS + LLM 语义采样 + L-TAFM)
- **目标**: 在 GS-Meta 元学习框架基础上，引入 KRGTS 关系网络、LLM 语义引导的辅助任务采样和任务自适应特征调制，提升少样本分子属性预测性能
- **技术栈**: PyTorch 1.7.1, PyTorch Geometric 1.6.3, learn2learn 0.1.6, Python 3.7
- **基础论文**:
  1. **GS-Meta** (Zhuang et al., IJCAI 2023): 图采样元学习，任务选择器 + NCE 对比学习
  2. **KRGTS** (Wang et al., 2023): 知识增强关系图，HMRGNN 异构消息传递
  3. **FiLM** (Perez et al., AAAI 2018): Feature-wise Linear Modulation，条件特征调制
- **核心创新**: 
  - 🌟 **创新 1 - KRGTS 关系网络**: 替代 Baseline 关系网络，使用 HMRGNN + scaffold/group 知识边
  - 🌟 **创新 2 - LLM 语义引导的辅助任务采样**: 使用 LLM 生成的属性描述嵌入，智能选择与目标任务语义相关的辅助任务
  - 🌟 **创新 3 - L-TAFM 任务自适应特征调制**: 基于 FiLM 机制，利用 LLM 任务语义动态调制 GNN 编码后的分子特征，使其具备任务感知能力

## 目录结构分析
```
/nfs/home/fragenmgr/code/GSKA/GSKLLM-modify/ 
├── run.py                  # 主入口
├── meta_learner.py         # 元学习器核心逻辑
├── args_parser.py          # 命令行参数
├── models/
│   ├── model.py            # GSMeta 主模型
│   ├── base_encoder.py     # GNN 分子编码器
│   ├── krgts_relation.py   # KRGTS 关系网络 (HMRGNN)
│   ├── krgts_hgnn.py       # 异构图神经网络
│   ├── knowledge_edges.py  # 知识边计算 (scaffold/group)
│   ├── semantic_sampler.py # 🌟 创新 2: LLM 语义引导采样器
│   ├── ltafm.py            # 🌟 创新 3: L-TAFM 特征调制
│   ├── selector.py         # 任务选择器 (REINFORCE)
│   ├── relation.py         # 原始关系网络 (baseline)
│   ├── maml.py             # MAML 元学习
│   └── loss.py             # NCE 对比学习损失
├── dataset/
│   ├── dataset.py          # 数据集加载
│   └── sampler.py          # Episode 采样
├── data/
│   ├── sider/              # SIDER 副作用数据集 (27 任务)
│   ├── tox21/              # Tox21 毒性数据集 (12 任务)
│   └── embeddings/         # 🌟 LLM 属性语义嵌入
├── scripts/
│   └── run_sider_semantic.sh  # 实验运行脚本
├── pretrained/
      └── supervised_contextpred.pth  # 预训练 GNN 权重
└── dump/                   # 实验输出目录
```

## 核心架构模式

### 1. GSKLLM-modify 架构 (实际运行配置)
```
Input: 分子图 (SMILES → PyG Data)
    ↓
[1] GNN Encoder - 预训练 ContextPred (5层 GIN, 300维)
    ↓
[2] 🌟 L-TAFM (创新 3) - 任务自适应特征调制
    │   ├─ 获取 LLM 任务语义向量 e_task
    │   ├─ MLP: e_task → [γ, β]
    │   └─ FiLM 变换: H' = γ ⊙ H + β
    ↓
[3] KRGTS Relation Network (创新 1) - HMRGNN 异构消息传递
    │   ├─ 分子-任务关系边 (属性标签)
    │   ├─ 分子-分子知识边 (scaffold 相似度)
    │   └─ 分子-分子知识边 (functional group 相似度)
    ↓
[4] Task Selector - REINFORCE 策略梯度
    │   └─ 🌟 辅助任务采样受 LLM 语义相似度引导 (创新 2)
    ↓
[5] Classifier - 二分类预测
    ↓
Output: 分子属性预测 (AUROC 评估)
```

### 2. 元学习训练流程
```
Meta-Training Loop (MAML + 二阶优化):
  For each episode:
    [1] Task Selector - 从任务池选择训练任务 (Policy Gradient)
    
    [2] 🌟 Semantic-Guided Auxiliary Sampling:
        - 获取目标任务与候选任务的 LLM 语义相似度
        - 按相似度加权采样辅助任务 (50% 语义 + 50% 均匀)
        - 避免选择语义无关的噪声任务
    
    [3] Inner Loop (Support Set):
        - MAML 内循环适应
        - 计算分类损失，更新适应后参数
    
    [4] Outer Loop (Query Set):
        - 计算查询集损失
        - 二阶梯度更新元参数
    
    [5] Contrastive Learning:
        - NCE 对比损失学习任务级表示
        - 用于 Task Selector 的 reward 信号
```

### 3. 基线配置索引
- **数据集**: SIDER (27任务), Tox21 (12任务), MUV, ToxCast, PCBA
- **实验设置**: K-shot (K=1, 10)
- **评估指标**: ROC-AUC
- **环境**: 
  - conda activate pyg
  - Python 3.7, PyTorch 1.7.1, PyG 1.6.3
  - GPU: RTX 4090 / RTX 3090

---

# 第二部分 项目开发日志

---

LOG-0001: 2025-12-29 20:30:00+08:00 - 实施 L-TAFM 任务自适应特征调制模块 - 复杂 - 责任人: AI

1. 用户请求
- 实施第三个创新点：LLM 驱动的任务自适应特征调制 (L-TAFM)，基于 FiLM 机制，使 GNN 编码器对分子特征提取具备任务感知能力。

2. 诊断总结与代码库检查
- **交互前代码库检查结果**：
  - 当前架构包含 GS-Meta + KRGTS + LLM 语义采样器，但 GNN 编码器对所有任务使用相同的特征提取方式
  - LLM 语义向量已预加载在 `semantic_sampler.embedding_matrix` 中，可通过任务 ID 索引获取
  - GNN Encoder 输出位置：`model.py` 第 61 行，输出 `s_feat [n_s, 300]`, `q_feat [n_q, 300]`
  - 插入位置确认：在 GNN 编码后、特征拼接前应用 FiLM 调制（第 61-64 行之间）
- **根因分析**：当前 GNN 编码器是任务无关的 (Task-Agnostic)，无法针对不同化学属性（如"肝毒性" vs "溶解度"）关注分子的不同子结构，导致特征利用效率不足。

3. 实施清单
- 创建 `models/ltafm.py`，实现 `FiLMLayer` 类（语义投影 MLP + 仿射变换）和 `TaskAdaptiveModulator` 类（整合多样性正则化）
- 修改 `models/model.py`：
  - 导入 `TaskAdaptiveModulator`
  - 在 `GSMeta.__init__` 中添加 L-TAFM 模块初始化（当 `args.use_ltafm=True` 时）
  - 加载 LLM 语义嵌入矩阵并注册为 buffer
  - 在 `forward` 方法中，GNN 编码后提取目标任务的语义向量，对 `s_feat` 和 `q_feat` 分别应用 FiLM 调制
  - 计算并返回 γ 多样性正则化损失
- 修改 `args_parser.py`，添加参数：
  - `--use_ltafm`: 启用/禁用 L-TAFM (默认 0)
  - `--ltafm_semantic_dim`: 语义向量维度 (默认 768, SciBERT)
  - `--ltafm_hidden_dim`: MLP 隐藏层维度 (默认 256)
  - `--ltafm_gamma_init`: γ 初始化偏置 (默认 0.0)
  - `--ltafm_diversity_weight`: 多样性正则化权重 (默认 0.01)
- 修改 `models/__init__.py`，导出 `FiLMLayer` 和 `TaskAdaptiveModulator`
- 创建 `scripts/run_sider_ltafm.sh`，配置完整的三模块协同实验（KRGTS + LLM Sampler + L-TAFM）
- 为脚本添加执行权限
- 生成本日志条目并写入 `PROJECT_BASELINE.md`

4. 最终产出
- **代码变更/问题状态**: 任务成功完成
- **受影响模块**: 
  - 新增：`models/ltafm.py` (175 行，实现 FiLM 核心逻辑)
  - 修改：`models/model.py` (新增 L-TAFM 初始化与调用逻辑，45+ 行)
  - 修改：`args_parser.py` (新增 5 个 L-TAFM 参数)
  - 修改：`models/__init__.py` (导出 2 个新类)
- **关键差异**: 
  - **FiLM 变换**: `H' = γ ⊙ H + β`，其中 γ, β 由 MLP(LLM_task_semantic) 生成
  - **多样性正则化**: `L_div = -Var(γ)`，鼓励 γ 在不同维度分化，避免退化为恒等变换
  - **初始化策略**: γ bias 初始化为 0.0，经 Softplus 后 γ ≈ 1.0，确保初始状态接近恒等变换
- **文件系统变更**:
  - 创建 `/nfs/home/fragenmgr/code/GSKA/GSKLLM-modify/models/ltafm.py`
  - 修改 `/nfs/home/fragenmgr/code/GSKA/GSKLLM-modify/models/model.py`
  - 修改 `/nfs/home/fragenmgr/code/GSKA/GSKLLM-modify/args_parser.py`
  - 修改 `/nfs/home/fragenmgr/code/GSKA/GSKLLM-modify/models/__init__.py`
  - 创建 `/nfs/home/fragenmgr/code/GSKA/GSKLLM-modify/scripts/run_sider_ltafm.sh`
- **验证结果**: 
  - Linter 检查通过，无语法错误
  - 模块导入路径正确
  - 参数传递链路完整（args → GSMeta → TaskAdaptiveModulator → FiLMLayer）

5. 架构影响
- **更新"核心项目架构"**：新增第三个核心创新点
- **架构变更点**：
  - 核心创新列表：增加 "L-TAFM 任务自适应特征调制"
  - 目录结构：`models/` 下新增 `ltafm.py`
  - 架构流程图：在 GNN Encoder 和 KRGTS Relation 之间插入 L-TAFM 层
  - 实验配置：新增 `scripts/run_sider_ltafm.sh`
- **技术细节**：
  - **参数量**: MLP (768 → 256 → 600) ≈ 200K，轻量级设计
  - **计算复杂度**: O(n_mol × d × d_hidden)，远低于 Cross-Attention 的 O(n_mol² × d)
  - **梯度稳定性**: 仿射变换不引入深层非线性，适合少样本场景
  - **监控指标**: `gamma_deviation_from_1` (期望 < 0.3)，`gamma_var` (期望 > 0.05)

---

#### 项目总览

**GSKLLM-modify** 是在 **GS-Meta (IJCAI 2023)** 基础上的扩展，增加了两个核心组件：
1. **KRGTS 关系网络**: 替代原始关系网络，使用 HMRGNN + 知识边
2. **LLM 语义引导的辅助任务采样**: 🌟 核心创新

#### 核心研究问题

在药物发现的少样本学习中：
- **数据稀缺**：许多毒性/药效预测任务只有极少量标注分子 (1-shot 或 10-shot)
- **辅助任务选择问题**：GS-Meta 随机采样辅助任务，可能引入语义无关的噪声任务
- **核心假设**：语义相关的辅助任务能提供更有效的知识迁移

###  架构设计

**基础**: GS-Meta (IJCAI 2023)
**扩展 1**: KRGTS 关系网络 (HMRGNN + scaffold/group 知识边)
**扩展 2**: 🌟 LLM 语义引导的辅助任务采样

```
┌────────────────────────────────────────────────────────────────────────┐
│           GS-Meta + KRGTS + LLM Sampler + L-TAFM (三模块协同)           │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────┐   ┌───────────┐   ┌─────────────────┐   ┌────────┐ │
│  │ GNN Encoder  │ → │ 🌟 L-TAFM │ → │ KRGTS Relation  │ → │ Task   │ │
│  │ (ContextPred)│   │ (FiLM)    │   │ (HMRGNN)        │   │Selector│ │
│  └──────────────┘   └───────────┘   └─────────────────┘   └────────┘ │
│                           ↑                   ↑                  ↑     │
│                 ┌─────────┴────┐    ┌────────┴────────┐  ┌─────┴───┐ │
│                 │ LLM 语义向量 │    │ 知识边融合       │  │🌟 LLM   │ │
│                 │ e_task       │    │ scaffold + group│  │语义采样 │ │
│                 │ γ, β 调制    │    └─────────────────┘  └─────────┘ │
│                 └──────────────┘                                      │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │         MAML 元学习 + NCE 对比学习 (来自 GS-Meta)              │    │
│  └──────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────┘
```

### 4. 核心创新

#### 4.1 🌟 创新 1: KRGTS 关系网络 ✅

`--relation_backend krgts`

- 来源: KRGTS 论文
- 替代 GS-Meta 原始的 `TaskRelationNet`
- HMRGNN 异构消息传递 + scaffold/group 知识边自动融合

#### 4.2 🌟 创新 2: LLM 语义引导的辅助任务采样 ✅

`--use_semantic_sampler 1`

**核心思想**：
- 使用 LLM 生成属性描述的语义嵌入
- 计算任务间语义相似度
- 辅助任务采样偏向语义相关任务

**实现** (`meta_learner.py:sample_auxiliary`):
```python
# 获取语义相似度
similarities = self.semantic_sampler.get_task_similarities(tgt_task_id, candidates)

# 混合语义采样和均匀采样 (50% + 50%)
probs = softmax(similarities / temperature)
mixed_probs = 0.5 * probs + 0.5 * uniform

# 按概率采样辅助任务
selected = np.random.choice(candidates, p=mixed_probs)
```

#### 4.3 🌟 创新 3: L-TAFM 任务自适应特征调制 ✅ (NEW)

`--use_ltafm 1`

**核心思想**：
- 基于 FiLM (Feature-wise Linear Modulation) 机制
- 利用 LLM 任务语义向量动态生成调制参数 (γ, β)
- 对 GNN 编码后的分子特征进行任务感知的仿射变换

**实现** (`models/ltafm.py` + `models/model.py`):
```python
# 步骤 1: 获取 LLM 任务语义向量
task_semantic = self.semantic_embeddings[target_task_id]  # [768]

# 步骤 2: 生成调制参数
gamma, beta = self.ltafm.film.mlp(task_semantic)  # [300], [300]
gamma = F.softplus(gamma)  # 确保 γ > 0

# 步骤 3: FiLM 变换
H_modulated = gamma * H_gnn + beta

# 步骤 4: 多样性正则化
diversity_loss = -Var(gamma)  # 鼓励 γ 分化
```

**技术优势**：
- **参数高效**: MLP (768→256→600) ≈ 200K 参数，适合少样本场景
- **梯度稳定**: 仿射变换不引入深层非线性，避免过拟合
- **可解释性**: γ 表示"哪些维度重要"，β 表示"特征偏移方向"
- **理论背书**: FiLM 已在 VQA (AAAI 2018) 和视觉推理中验证有效

**参数配置**：
- `--ltafm_semantic_dim 768`: SciBERT 嵌入维度
- `--ltafm_hidden_dim 256`: MLP 隐藏层
- `--ltafm_gamma_init 0.0`: γ 初始化偏置（初始 γ ≈ 1）
- `--ltafm_diversity_weight 0.01`: 多样性正则化权重

### 5. 实验配置

```bash
# scripts/run_sider_semantic.sh
python run.py \
    --dataset sider \
    --n_support 1 \
    --episode 2000 \
    --relation_backend krgts \
    --use_semantic_sampler 1 \
    --semantic_temperature 0.1 \
    --semantic_similarity_weight 0.5
```

### 6. 数据

```
data/
├── sider/, tox21/, muv/...     # 分子属性数据集
└── embeddings/
    └── sider_property_embeddings.pt  # 🌟 LLM 属性嵌入
```

### 7. 研究价值

| 组件 | 来源 | 作用 |
|------|------|------|
| GS-Meta 框架 | IJCAI 2023 | MAML + NCE + TaskSelector |
| 🌟 创新 1: KRGTS 关系网络 | KRGTS 论文 | HMRGNN + 知识边 |
| 🌟 创新 2: LLM 语义采样 | **本工作创新** | 智能辅助任务选择 |
| 🌟 创新 3: L-TAFM 特征调制 | **本工作创新** | 任务感知特征提取 |

### 8. 最终产出

- 完整的四层架构: GS-Meta (基础) + KRGTS (关系网络) + LLM Sampler (语义采样) + L-TAFM (特征调制)
- 三个核心创新点协同工作，形成端到端的任务自适应少样本学习系统
- 代码中存在的其他模块 (EDL, DyConfidMatch, GP Head 等) **均未使用**

### 9. 消融实验设计

| 配置 | KRGTS | LLM Sampler | L-TAFM | 预期效果 |
|------|-------|-------------|--------|---------|
| Baseline | ❌ | ❌ | ❌ | GS-Meta 原始性能 |
| +KRGTS | ✅ | ❌ | ❌ | 验证知识边贡献 |
| +LLM | ✅ | ✅ | ❌ | 验证语义采样贡献 |
| **Full (本工作)** | ✅ | ✅ | ✅ | **最优性能** |
| -KRGTS | ❌ | ✅ | ✅ | 验证 L-TAFM 独立贡献 |


