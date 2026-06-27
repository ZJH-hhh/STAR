#!/usr/bin/env python
"""
预计算属性描述嵌入
==================

为 SIDER 和 Tox21 数据集的属性描述生成语义嵌入向量。
支持多种编码器后端：
1. BERT (本地/缓存)
2. SentenceTransformers (如果可用)
3. 简单的 TF-IDF + SVD (无需深度学习模型)

Usage:
    python scripts/precompute_property_embeddings.py --dataset sider --encoder bert
    python scripts/precompute_property_embeddings.py --dataset tox21 --encoder tfidf
    python scripts/precompute_property_embeddings.py --dataset all --encoder bert

Output:
    data/embeddings/sider_property_embeddings.pt
    data/embeddings/tox21_property_embeddings.pt
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.property_descriptions import (SIDER_DESCRIPTIONS, TOX21_DESCRIPTIONS, MUV_DESCRIPTIONS,
                                        PCBA_DESCRIPTIONS, get_all_descriptions)
from semantic_utils import SEMANTIC_EMBEDDING_DIRS, normalize_semantic_encoder


def encode_with_bert(descriptions: dict, model_name: str = "bert-base-uncased") -> dict:
    """
    使用 BERT 模型编码属性描述
    
    Args:
        descriptions: {property_name: description_text}
        model_name: HuggingFace 模型名称
    
    Returns:
        {property_name: embedding_tensor}
    """
    try:
        # 尝试绕过安全检查（仅用于可信模型）
        os.environ['TRANSFORMERS_ALLOW_TORCH_LOAD'] = '1'
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError("请安装 transformers: pip install transformers")
    
    print(f"Loading BERT model: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
    except Exception as e:
        print(f"无法加载 {model_name}: {e}")
        print("尝试使用 TF-IDF 编码器作为备选...")
        return encode_with_tfidf(descriptions)
    
    embeddings = {}
    
    with torch.no_grad():
        for name, desc in descriptions.items():
            # Tokenize
            inputs = tokenizer(
                desc,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Encode
            outputs = model(**inputs)
            
            # 使用 [CLS] token 的输出作为句子表示
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
            
            # 也可以使用 mean pooling
            # attention_mask = inputs['attention_mask']
            # mean_embedding = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
            
            embeddings[name] = cls_embedding.cpu()
            print(f"  Encoded: {name[:50]}... -> shape {cls_embedding.shape}")
    
    return embeddings


def encode_with_sentence_transformers(descriptions: dict, model_name: str = "all-MiniLM-L6-v2") -> dict:
    """
    使用 SentenceTransformers 编码（更简单的 API）
    
    Args:
        descriptions: {property_name: description_text}
        model_name: SentenceTransformers 模型名称
    
    Returns:
        {property_name: embedding_tensor}
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("SentenceTransformers 未安装，使用 TF-IDF 编码器...")
        return encode_with_tfidf(descriptions)
    
    print(f"Loading SentenceTransformer: {model_name}")
    model = SentenceTransformer(model_name)
    
    names = list(descriptions.keys())
    texts = list(descriptions.values())
    
    print(f"Encoding {len(texts)} descriptions...")
    vectors = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    
    embeddings = {}
    for name, vec in zip(names, vectors):
        embeddings[name] = vec.cpu()
        print(f"  Encoded: {name[:50]}... -> shape {vec.shape}")
    
    return embeddings


def encode_with_tfidf(descriptions: dict, n_components: int = 128) -> dict:
    """
    使用 TF-IDF + SVD 编码（无需深度学习模型）
    
    这是最简单的备选方案，不需要任何预训练模型。
    
    Args:
        descriptions: {property_name: description_text}
        n_components: SVD 降维后的维度
    
    Returns:
        {property_name: embedding_tensor}
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    
    names = list(descriptions.keys())
    texts = list(descriptions.values())
    
    # 检查是否需要SVD降维
    n_samples = len(texts)
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples for TF-IDF encoding, got {n_samples}")
    
    # TF-IDF 向量化
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # 使用 unigram 和 bigram
        stop_words='english'
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # SVD 降维（如果样本数足够多）
    max_components = min(n_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1])
    
    if max_components < 1:
        # 样本太少，直接使用TF-IDF向量（密集化）
        print(f"Using TF-IDF vectors directly (no SVD, samples={n_samples})")
        reduced_matrix = tfidf_matrix.toarray()
        variance_explained = 1.0
    else:
        print(f"Using TF-IDF + SVD (n_components={max_components})")
        svd = TruncatedSVD(n_components=max_components, random_state=42)
        reduced_matrix = svd.fit_transform(tfidf_matrix)
        variance_explained = svd.explained_variance_ratio_.sum()
    
    print(f"  Explained variance ratio: {variance_explained:.2%}")
    
    # 转换为 torch tensor
    embeddings = {}
    for i, name in enumerate(names):
        vec = torch.tensor(reduced_matrix[i], dtype=torch.float32)
        # L2 归一化
        vec = vec / (vec.norm() + 1e-8)
        embeddings[name] = vec
        print(f"  Encoded: {name[:50]}... -> shape {vec.shape}")
    
    return embeddings


def save_embeddings(embeddings: dict, output_path: Path, metadata: dict = None):
    """
    保存嵌入到 .pt 文件
    
    Args:
        embeddings: {property_name: embedding_tensor}
        output_path: 输出文件路径
        metadata: 额外的元数据
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 构建保存数据
    save_data = {
        'embeddings': embeddings,
        'property_names': list(embeddings.keys()),
        'embedding_dim': next(iter(embeddings.values())).shape[0],
        'num_properties': len(embeddings),
    }
    
    if metadata:
        save_data['metadata'] = metadata
    
    # 构建嵌入矩阵（按属性名排序）
    sorted_names = sorted(embeddings.keys())
    embedding_matrix = torch.stack([embeddings[name] for name in sorted_names])
    save_data['embedding_matrix'] = embedding_matrix
    save_data['sorted_names'] = sorted_names
    
    # 保存
    torch.save(save_data, output_path)
    print(f"\nSaved embeddings to: {output_path}")
    print(f"  - Number of properties: {len(embeddings)}")
    print(f"  - Embedding dimension: {save_data['embedding_dim']}")
    print(f"  - Matrix shape: {embedding_matrix.shape}")
    
    # 同时保存属性名映射为 JSON（便于查看）
    json_path = output_path.with_suffix('.json')
    json_data = {
        'property_names': sorted_names,
        'embedding_dim': save_data['embedding_dim'],
        'name_to_index': {name: i for i, name in enumerate(sorted_names)},
    }
    if metadata:
        json_data['metadata'] = metadata
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"  - Index mapping saved to: {json_path}")


def compute_similarity_matrix(embeddings: dict) -> tuple:
    """
    计算属性间的余弦相似度矩阵
    
    Returns:
        (similarity_matrix, sorted_names)
    """
    sorted_names = sorted(embeddings.keys())
    vectors = torch.stack([embeddings[name] for name in sorted_names])
    
    # L2 归一化
    vectors_norm = vectors / (vectors.norm(dim=1, keepdim=True) + 1e-8)
    
    # 余弦相似度
    similarity = torch.mm(vectors_norm, vectors_norm.t())
    
    return similarity, sorted_names


def visualize_similarity(similarity: torch.Tensor, names: list, output_path: Path = None):
    """
    可视化相似度矩阵（如果 matplotlib 可用）
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn 未安装，跳过可视化")
        return
    
    # 截断过长的名称
    short_names = [n[:25] + '...' if len(n) > 25 else n for n in names]
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        similarity.numpy(),
        xticklabels=short_names,
        yticklabels=short_names,
        cmap='RdYlBu_r',
        center=0,
        vmin=-0.5,
        vmax=1.0,
        annot=False,
    )
    plt.title('Property Semantic Similarity Matrix')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  - Similarity heatmap saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def get_property_names_from_csv(dataset: str, data_root: Path) -> list:
    """从CSV文件提取属性名称列表"""
    import pandas as pd
    
    csv_path = data_root / dataset / f"{dataset}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        # 读取CSV的第一行（列名）
        df = pd.read_csv(csv_path, nrows=0)
        columns = list(df.columns)
        
        # 移除SMILES列
        property_cols = [col for col in columns if col.lower() not in ['smiles', 'mol_id', 'id']]
        
        return property_cols
    except Exception as e:
        print(f"Error reading CSV for {dataset}: {e}")
        print("Trying alternative method...")
        
        # 备选方法：直接读取第一行
        with open(csv_path, 'r') as f:
            header = f.readline().strip()
            if 'git-lfs' in header.lower():
                raise ValueError(f"CSV file for {dataset} is a Git LFS pointer. Please run 'git lfs pull' first.")
            columns = header.split(',')
            property_cols = [col for col in columns if col.lower() not in ['smiles', 'mol_id', 'id']]
            return property_cols


def main():
    parser = argparse.ArgumentParser(description="预计算属性描述嵌入")
    parser.add_argument("--dataset", type=str, default="sider",
                        help="数据集名称 (sider, tox21, muv, pcba, toxcast-*, 或 all)")
    parser.add_argument("--encoder", type=str, default="tfidf",
                        choices=["bert", "sbert", "sentence-transformers", "tfidf"],
                        help="编码器类型")
    parser.add_argument("--model_name", type=str, default=None,
                        help="模型名称（覆盖默认值）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录")
    parser.add_argument("--data_root", type=str, default="data",
                        help="数据根目录")
    parser.add_argument("--visualize", action="store_true",
                        help="生成相似度矩阵可视化")
    
    args = parser.parse_args()
    args.encoder = normalize_semantic_encoder(args.encoder)
    
    # 确定要处理的数据集
    if args.dataset == "all":
        # 扫描data目录下的所有数据集
        data_root = PROJECT_ROOT / args.data_root
        datasets = []
        for item in data_root.iterdir():
            if item.is_dir() and item.name not in ['embeddings', '__pycache__']:
                csv_file = item / f"{item.name}.csv"
                if csv_file.exists():
                    datasets.append(item.name)
        print(f"Found {len(datasets)} datasets: {', '.join(datasets)}")
    else:
        datasets = [args.dataset]
    
    # 确定编码器
    encoder_fn = {
        "bert": encode_with_bert,
        "sbert": encode_with_sentence_transformers,
        "tfidf": encode_with_tfidf,
    }[args.encoder]
    
    # 默认模型名
    default_models = {
        "bert": "bert-base-uncased",
        "sbert": "all-MiniLM-L6-v2",
        "tfidf": None,
    }
    model_name = args.model_name or default_models[args.encoder]
    
    if args.output_dir is None:
        args.output_dir = SEMANTIC_EMBEDDING_DIRS[args.encoder]
    output_dir = PROJECT_ROOT / args.output_dir
    data_root = PROJECT_ROOT / args.data_root
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset.upper()} dataset")
        print(f"{'='*60}")
        
        # 获取描述
        if dataset == "sider":
            descriptions = SIDER_DESCRIPTIONS
        elif dataset == "tox21":
            descriptions = TOX21_DESCRIPTIONS
        elif dataset == "muv":
            descriptions = MUV_DESCRIPTIONS
        elif dataset == "pcba":
            descriptions = PCBA_DESCRIPTIONS
            print(f"Using pre-generated descriptions for {len(descriptions)} properties")
        else:
            # 自动生成描述（用于pcba和toxcast-*）
            try:
                property_names = get_property_names_from_csv(dataset, data_root)
                if len(property_names) < 2:
                    print(f"Warning: Only {len(property_names)} properties found (likely Git LFS issue)")
                    print(f"Skipping {dataset}... (run 'git lfs pull' to download large CSV files)")
                    continue
                descriptions = get_all_descriptions(dataset, property_names)
                print(f"Auto-generated descriptions for {len(descriptions)} properties")
            except Exception as e:
                print(f"Error processing {dataset}: {e}")
                print(f"Skipping {dataset}...")
                continue
        
        print(f"Total {len(descriptions)} properties")
        
        # 编码
        if model_name:
            embeddings = encoder_fn(descriptions, model_name)
        else:
            embeddings = encoder_fn(descriptions)
        
        # 保存到数据集专属子目录
        dataset_output_dir = output_dir / dataset
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = dataset_output_dir / f"{dataset}_property_embeddings.pt"
        metadata = {
            "encoder": args.encoder,
            "model_name": model_name,
            "dataset": dataset,
        }
        save_embeddings(embeddings, output_path, metadata)
        
        # 计算并展示相似度
        similarity, names = compute_similarity_matrix(embeddings)
        
        print(f"\nTop similar property pairs:")
        n = len(names)
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                pairs.append((names[i], names[j], similarity[i, j].item()))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        for p1, p2, sim in pairs[:5]:
            print(f"  {sim:.3f}: {p1[:30]} <-> {p2[:30]}")
        
        # 可视化
        if args.visualize:
            viz_path = dataset_output_dir / f"{dataset}_similarity_matrix.png"
            visualize_similarity(similarity, names, viz_path)
    
    print(f"\n{'='*60}")
    print("Done! Embeddings saved to:", output_dir)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
