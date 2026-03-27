"""
LLM-Driven Task-Adaptive Feature Modulation (L-TAFM)
=====================================================

基于 FiLM (Feature-wise Linear Modulation) 的任务自适应特征调制模块。

核心思想：
- 利用 LLM 任务语义向量生成调制参数 (γ, β)
- 对 GNN 编码后的分子特征进行仿射变换，使其具备任务感知能力
- 轻量级设计，参数量小，适合少样本场景

Reference:
- FiLM: Visual Reasoning with a General Conditioning Layer (AAAI 2018)
- Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation Layer
    
    给定分子特征 H 和任务语义向量 e_text，生成调制参数 γ 和 β，
    执行仿射变换: H' = γ ⊙ H + β
    
    Args:
        feature_dim: 分子特征维度 (GNN 输出维度)
        semantic_dim: LLM 语义向量维度
        hidden_dim: MLP 隐藏层维度
        gamma_init: γ 的初始化偏置（推荐 0.0，使初始 γ ≈ 1）
        dropout: Dropout 概率
    """
    
    def __init__(
        self,
        feature_dim: int,
        semantic_dim: int,
        hidden_dim: int = 256,
        gamma_init: float = 0.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.semantic_dim = semantic_dim
        self.hidden_dim = hidden_dim
        
        # 语义向量投影 MLP: e_text -> [γ, β]
        # 使用 LayerNorm 替代 BatchNorm，因为语义嵌入的 batch_size 可能为 1
        self.mlp = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * feature_dim),  # 输出 [γ, β] 拼接
        )
        
        # 初始化：确保初始调制接近恒等变换 (γ ≈ 1, β ≈ 0)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias[:feature_dim], gamma_init)  # γ 部分
        nn.init.zeros_(self.mlp[-1].bias[feature_dim:])  # β 部分
        
    def forward(
        self,
        features: torch.Tensor,
        semantic_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: 分子特征 [batch_size, feature_dim] 或 [batch_size, n_mols, feature_dim]
            semantic_emb: 任务语义向量 [batch_size, semantic_dim] 或 [semantic_dim]
            
        Returns:
            modulated_features: 调制后特征 [batch_size, feature_dim] 或 [batch_size, n_mols, feature_dim]
            gamma: 缩放因子 [batch_size, feature_dim]
            beta: 偏移因子 [batch_size, feature_dim]
        """
        # 处理语义向量的维度
        if semantic_emb.dim() == 1:
            semantic_emb = semantic_emb.unsqueeze(0)  # [1, semantic_dim]
        
        # 生成调制参数
        modulation = self.mlp(semantic_emb)  # [batch_size, 2 * feature_dim]
        gamma, beta = torch.chunk(modulation, 2, dim=-1)  # 各 [batch_size, feature_dim]
        
        # 应用 Softplus 确保 γ > 0 (避免负缩放)
        gamma = F.softplus(gamma) + 1e-8
        
        # 扩展维度以匹配 features
        if features.dim() == 3:  # [batch_size, n_mols, feature_dim]
            gamma = gamma.unsqueeze(1)  # [batch_size, 1, feature_dim]
            beta = beta.unsqueeze(1)
        
        # FiLM 变换: H' = γ ⊙ H + β
        modulated_features = gamma * features + beta
        
        return modulated_features, gamma.squeeze(1) if gamma.dim() == 3 else gamma, beta.squeeze(1) if beta.dim() == 3 else beta


class TaskAdaptiveModulator(nn.Module):
    """
    完整的任务自适应调制器
    
    整合 FiLM 层和多样性正则化，用于主模型集成。
    
    Args:
        feature_dim: 分子特征维度
        semantic_dim: LLM 语义向量维度
        hidden_dim: MLP 隐藏层维度
        gamma_init: γ 初始化偏置
        diversity_weight: 多样性正则化权重
        dropout: Dropout 概率
    """
    
    def __init__(
        self,
        feature_dim: int,
        semantic_dim: int,
        hidden_dim: int = 256,
        gamma_init: float = 0.0,
        diversity_weight: float = 0.01,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.film = FiLMLayer(
            feature_dim=feature_dim,
            semantic_dim=semantic_dim,
            hidden_dim=hidden_dim,
            gamma_init=gamma_init,
            dropout=dropout,
        )
        
        self.diversity_weight = diversity_weight
        
    def forward(
        self,
        features: torch.Tensor,
        semantic_emb: torch.Tensor,
        compute_diversity_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            features: 分子特征 [batch_size, feature_dim] 或 [batch_size, n_mols, feature_dim]
            semantic_emb: 任务语义向量 [batch_size, semantic_dim]
            compute_diversity_loss: 是否计算多样性正则化损失
            
        Returns:
            modulated_features: 调制后特征
            diversity_loss: 多样性正则化损失 (如果 compute_diversity_loss=True)
        """
        modulated_features, gamma, beta = self.film(features, semantic_emb)
        
        diversity_loss = None
        if compute_diversity_loss and self.diversity_weight > 0:
            # 计算 γ 的方差，鼓励不同维度的分化
            # L_div = -Var(γ)，使用负方差作为损失（方差越大越好）
            gamma_var = torch.var(gamma, dim=-1).mean()  # [batch_size] -> scalar
            diversity_loss = -self.diversity_weight * gamma_var
        
        return modulated_features, diversity_loss
    
    def get_modulation_stats(self, semantic_emb: torch.Tensor) -> dict:
        """
        获取调制参数的统计信息（用于监控）
        
        Args:
            semantic_emb: 任务语义向量 [batch_size, semantic_dim]
            
        Returns:
            stats: 包含 γ 和 β 统计信息的字典
        """
        with torch.no_grad():
            modulation = self.film.mlp(semantic_emb)
            gamma, beta = torch.chunk(modulation, 2, dim=-1)
            gamma = F.softplus(gamma) + 1e-8
            
            stats = {
                'gamma_mean': gamma.mean().item(),
                'gamma_std': gamma.std().item(),
                'gamma_min': gamma.min().item(),
                'gamma_max': gamma.max().item(),
                'gamma_deviation_from_1': torch.abs(gamma - 1.0).mean().item(),
                'beta_mean': beta.mean().item(),
                'beta_std': beta.std().item(),
            }
        
        return stats

