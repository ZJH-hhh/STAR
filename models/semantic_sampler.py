"""
Semantic-Guided Task Sampler
============================

基于属性语义相似度的辅助任务采样模块。

核心思想：
- 使用预计算的属性描述嵌入来衡量任务间的语义相关性
- 辅助任务的选择受语义相似度引导，避免选择语义无关的噪声任务
- 与 KRGTS 采样器无缝集成

Usage:
    semantic_sampler = SemanticGuidedSampler(
        embedding_path="data/embeddings/sider_property_embeddings.pt",
        temperature=0.1,
        similarity_weight=0.5,
    )
    
    # 获取目标任务与所有辅助任务的相似度
    similarities = semantic_sampler.get_task_similarities(target_task_idx, all_task_indices)
    
    # 用于修改采样器的 logits
    adjusted_logits = original_logits + similarity_weight * similarities
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from semantic_utils import load_task_embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_utils import normalize_semantic_encoder, validate_semantic_embedding_metadata


class SemanticGuidedSampler(nn.Module):
    """
    语义引导的辅助任务采样器
    
    通过预计算的属性语义嵌入来引导采样器选择与目标任务语义相关的辅助任务。
    """
    
    def __init__(
        self,
        embedding_path: str,
        temperature: float = 0.1,
        similarity_weight: float = 0.5,
        use_softmax: bool = True,
        min_similarity: float = -1.0,
        property_names: Optional[List[str]] = None,
        expected_encoder: Optional[str] = None,
    ):
        """
        Args:
            embedding_path: 预计算嵌入文件路径 (.pt)
            temperature: 相似度 softmax 温度（越低越尖锐）
            similarity_weight: 相似度权重（用于与原始 logits 混合）
            use_softmax: 是否对相似度应用 softmax
            min_similarity: 最小相似度阈值（低于此值的任务被强惩罚）
            property_names: 属性名称列表（如果与嵌入文件中的顺序不同）
        """
        super().__init__()
        
        self.temperature = temperature
        self.similarity_weight = similarity_weight
        self.use_softmax = use_softmax
        self.min_similarity = min_similarity
        self.expected_encoder = expected_encoder
        
        # 加载预计算嵌入
        self._load_embeddings(embedding_path, property_names)
        
    def _load_embeddings(self, embedding_path: str, property_names: Optional[List[str]] = None):
        """加载预计算的属性嵌入"""
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        
        data = load_task_embeddings(embedding_path)
        loaded_encoder = validate_semantic_embedding_metadata(
            data=data,
            expected_encoder=self.expected_encoder,
            embedding_path=embedding_path,
            consumer="SemanticSampler",
        )
        
        # 获取嵌入矩阵和属性名
        self.embedding_matrix = data['embedding_matrix']  # [num_properties, embedding_dim]
        self.property_names = data.get('sorted_names', data.get('property_names', []))
        self.num_properties, self.embedding_dim = self.embedding_matrix.shape
        
        # L2 归一化（确保余弦相似度计算正确）
        self.embedding_matrix = F.normalize(self.embedding_matrix, p=2, dim=1)
        
        # 预计算相似度矩阵
        self.register_buffer(
            'similarity_matrix',
            torch.mm(self.embedding_matrix, self.embedding_matrix.t())
        )
        
        # 属性名到索引的映射
        self.name_to_idx = {name: i for i, name in enumerate(self.property_names)}
        
        print(f"[SemanticSampler] Loaded {self.num_properties} property embeddings")
        print(f"[SemanticSampler] Embedding dim: {self.embedding_dim}")
        if loaded_encoder is not None:
            print(f"[SemanticSampler] Embedding encoder: {loaded_encoder}")
        
        # 如果提供了外部属性名列表，建立映射
        if property_names is not None:
            self._build_external_mapping(property_names)
    
    def _build_external_mapping(self, external_names: List[str]):
        """建立外部属性名到内部索引的映射"""
        self.external_to_internal = {}
        
        for i, ext_name in enumerate(external_names):
            # 尝试精确匹配
            if ext_name in self.name_to_idx:
                self.external_to_internal[i] = self.name_to_idx[ext_name]
            else:
                # 尝试模糊匹配（忽略大小写和空格）
                ext_clean = ext_name.lower().strip()
                matched = False
                for int_name, int_idx in self.name_to_idx.items():
                    if int_name.lower().strip() == ext_clean:
                        self.external_to_internal[i] = int_idx
                        matched = True
                        break
                
                if not matched:
                    print(f"[SemanticSampler] Warning: Property '{ext_name}' not found in embeddings")
                    # 使用默认映射（假设顺序一致）
                    if i < self.num_properties:
                        self.external_to_internal[i] = i
    
    def get_similarity(self, task_idx_a: int, task_idx_b: int) -> float:
        """获取两个任务之间的语义相似度"""
        return self.similarity_matrix[task_idx_a, task_idx_b].item()
    
    def get_task_similarities(
        self,
        target_task_idx: int,
        candidate_task_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        获取目标任务与候选任务之间的语义相似度
        
        Args:
            target_task_idx: 目标任务索引
            candidate_task_indices: 候选任务索引张量 [num_candidates]
        
        Returns:
            similarities: 相似度张量 [num_candidates]
        """
        # 获取目标任务的相似度向量
        target_similarities = self.similarity_matrix[target_task_idx]  # [num_properties]
        
        # 索引候选任务的相似度
        similarities = target_similarities[candidate_task_indices]
        
        return similarities
    
    def adjust_sampling_logits(
        self,
        original_logits: torch.Tensor,
        target_task_idx: int,
        candidate_task_indices: torch.Tensor,
        mode: str = 'additive',
    ) -> torch.Tensor:
        """
        根据语义相似度调整采样 logits
        
        Args:
            original_logits: 原始采样 logits [num_candidates]
            target_task_idx: 目标任务索引
            candidate_task_indices: 候选任务索引
            mode: 调整模式
                - 'additive': logits + weight * similarities
                - 'multiplicative': logits * (1 + weight * similarities)
                - 'gating': logits * sigmoid(weight * similarities)
        
        Returns:
            adjusted_logits: 调整后的 logits [num_candidates]
        """
        # 获取语义相似度
        similarities = self.get_task_similarities(target_task_idx, candidate_task_indices)
        
        # 应用最小相似度阈值（强惩罚低相似度任务）
        if self.min_similarity > -1.0:
            mask = similarities < self.min_similarity
            similarities = similarities.clone()
            similarities[mask] = -10.0  # 强惩罚
        
        # 温度缩放
        scaled_similarities = similarities / self.temperature
        
        if self.use_softmax:
            scaled_similarities = F.softmax(scaled_similarities, dim=0)
        
        # 应用调整
        if mode == 'additive':
            adjusted_logits = original_logits + self.similarity_weight * scaled_similarities
        elif mode == 'multiplicative':
            adjusted_logits = original_logits * (1.0 + self.similarity_weight * scaled_similarities)
        elif mode == 'gating':
            gate = torch.sigmoid(self.similarity_weight * scaled_similarities)
            adjusted_logits = original_logits * gate
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return adjusted_logits
    
    def compute_semantic_reward_bonus(
        self,
        selected_task_indices: torch.Tensor,
        target_task_idx: int,
    ) -> torch.Tensor:
        """
        计算基于语义相似度的采样奖励加成
        
        用于 REINFORCE 更新时给予语义相关任务额外奖励
        
        Args:
            selected_task_indices: 被选中的辅助任务索引
            target_task_idx: 目标任务索引
        
        Returns:
            reward_bonus: 奖励加成标量
        """
        similarities = self.get_task_similarities(target_task_idx, selected_task_indices)
        
        # 平均相似度作为奖励加成
        reward_bonus = similarities.mean() * self.similarity_weight
        
        return reward_bonus
    
    def get_top_k_similar_tasks(
        self,
        target_task_idx: int,
        k: int = 5,
        exclude_self: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取与目标任务最相似的 k 个任务
        
        Args:
            target_task_idx: 目标任务索引
            k: 返回的任务数量
            exclude_self: 是否排除目标任务自身
        
        Returns:
            (task_indices, similarities)
        """
        similarities = self.similarity_matrix[target_task_idx].clone()
        
        if exclude_self:
            similarities[target_task_idx] = -float('inf')
        
        top_k_sims, top_k_indices = torch.topk(similarities, k)
        
        return top_k_indices, top_k_sims
    
    def visualize_task_relationships(self, output_path: Optional[str] = None):
        """可视化任务间的语义关系"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib/seaborn not available for visualization")
            return
        
        sim_matrix = self.similarity_matrix.cpu().numpy()
        
        # 截断长名称
        short_names = [n[:20] + '...' if len(n) > 20 else n for n in self.property_names]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            sim_matrix,
            xticklabels=short_names,
            yticklabels=short_names,
            cmap='RdYlBu_r',
            center=0,
            vmin=-0.5,
            vmax=1.0,
            annot=False,
        )
        plt.title('Task Semantic Similarity Matrix')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def forward(
        self,
        task_logits: torch.Tensor,
        target_task_idx: int,
        candidate_indices: torch.Tensor,
        return_similarities: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播：调整任务采样 logits
        
        Args:
            task_logits: 原始任务 logits [batch_size, num_candidates] 或 [num_candidates]
            target_task_idx: 目标任务索引
            candidate_indices: 候选任务索引
            return_similarities: 是否返回原始相似度
        
        Returns:
            adjusted_logits: 调整后的 logits
            similarities (optional): 原始相似度
        """
        adjusted = self.adjust_sampling_logits(
            task_logits, target_task_idx, candidate_indices
        )
        
        if return_similarities:
            sims = self.get_task_similarities(target_task_idx, candidate_indices)
            return adjusted, sims
        
        return adjusted


class SemanticSamplerConfig:
    """语义采样器配置"""
    
    def __init__(
        self,
        enabled: bool = True,
        embedding_path: str = "data/embeddings/sider_property_embeddings.pt",
        encoder: str = "tfidf",
        temperature: float = 0.1,
        similarity_weight: float = 0.5,
        min_similarity: float = -1.0,
        use_softmax: bool = True,
        adjust_mode: str = 'additive',
        reward_bonus_weight: float = 0.1,
    ):
        self.enabled = enabled
        self.embedding_path = embedding_path
        self.encoder = normalize_semantic_encoder(encoder)
        self.temperature = temperature
        self.similarity_weight = similarity_weight
        self.min_similarity = min_similarity
        self.use_softmax = use_softmax
        self.adjust_mode = adjust_mode
        self.reward_bonus_weight = reward_bonus_weight
    
    @classmethod
    def from_args(cls, args) -> 'SemanticSamplerConfig':
        """从 argparse 参数创建配置"""
        return cls(
            enabled=getattr(args, 'use_semantic_sampler', False),
            embedding_path=getattr(args, 'semantic_embedding_path', 
                                   f"data/embeddings/{args.dataset}_property_embeddings.pt"),
            encoder=getattr(args, 'semantic_encoder', 'tfidf'),
            temperature=getattr(args, 'semantic_temperature', 0.1),
            similarity_weight=getattr(args, 'semantic_similarity_weight', 0.5),
            min_similarity=getattr(args, 'semantic_min_similarity', -1.0),
            use_softmax=getattr(args, 'semantic_use_softmax', True),
            adjust_mode=getattr(args, 'semantic_adjust_mode', 'additive'),
            reward_bonus_weight=getattr(args, 'semantic_reward_bonus', 0.1),
        )


def create_semantic_sampler(args) -> Optional[SemanticGuidedSampler]:
    """
    工厂函数：根据参数创建语义采样器
    
    Args:
        args: argparse 参数对象
    
    Returns:
        SemanticGuidedSampler 实例或 None（如果禁用）
    """
    if not getattr(args, 'use_semantic_sampler', False):
        return None
    
    config = SemanticSamplerConfig.from_args(args)
    
    # 构建嵌入文件路径
    embedding_path = config.embedding_path
    if not os.path.isabs(embedding_path):
        # 相对路径：从项目根目录解析
        project_root = Path(__file__).parent.parent
        embedding_path = str(project_root / embedding_path)
    
    try:
        sampler = SemanticGuidedSampler(
            embedding_path=embedding_path,
            temperature=config.temperature,
            similarity_weight=config.similarity_weight,
            use_softmax=config.use_softmax,
            min_similarity=config.min_similarity,
            expected_encoder=config.encoder,
        )
        return sampler
    except FileNotFoundError as e:
        print(f"[Warning] Semantic sampler disabled: {e}")
        return None


if __name__ == "__main__":
    # 测试
    import sys
    
    project_root = Path(__file__).parent.parent
    embedding_path = project_root / "data/embeddings/sider_property_embeddings.pt"
    
    if not embedding_path.exists():
        print(f"Please run precompute_property_embeddings.py first")
        sys.exit(1)
    
    print("=" * 60)
    print("Testing SemanticGuidedSampler")
    print("=" * 60)
    
    sampler = SemanticGuidedSampler(
        embedding_path=str(embedding_path),
        temperature=0.1,
        similarity_weight=0.5,
    )
    
    print(f"\nProperty names: {sampler.property_names[:5]}...")
    print(f"Similarity matrix shape: {sampler.similarity_matrix.shape}")
    
    # 测试相似度查询
    print("\n--- Top-5 similar tasks to 'Cardiac disorders' ---")
    cardiac_idx = sampler.name_to_idx.get("Cardiac disorders", 0)
    top_indices, top_sims = sampler.get_top_k_similar_tasks(cardiac_idx, k=5)
    for idx, sim in zip(top_indices, top_sims):
        print(f"  {sim.item():.3f}: {sampler.property_names[idx]}")
    
    # 测试 logits 调整
    print("\n--- Testing logits adjustment ---")
    original_logits = torch.randn(10)
    candidate_indices = torch.arange(10)
    adjusted = sampler.adjust_sampling_logits(original_logits, cardiac_idx, candidate_indices)
    print(f"Original logits: {original_logits[:5]}")
    print(f"Adjusted logits: {adjusted[:5]}")
    
    print("\n✅ All tests passed!")
