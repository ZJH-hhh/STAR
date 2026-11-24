#!/usr/bin/env python
"""Build frozen assay-text resources from electronic Table S27, in dataset-local task order.

For full descriptions, literal names and fixed random/shuffled controls, use
--dataset all --fit_scope both --text_table <S27.csv> --split_table <S26.csv>.
Use --output_dir to select a new text_embeddings directory. Each full/name
resource records its fitted width before any zero padding. Exact historical
replay uses the native resource identified by Table S28.
"""

import os
import sys
import argparse
import json
import hashlib
import pickle
import re
from collections import Counter
from pathlib import Path

import torch
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import csv
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
    sorted_names = list(embeddings)  # S27 order follows dataset-local task IDs
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
    sorted_names = list(embeddings)  # S27 order follows dataset-local task IDs
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


def read_text_table(path):
    by_dataset = {}
    with open(path, newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            by_dataset.setdefault(row["dataset"], []).append(row)
    return by_dataset


def ordered_rows(records):
    rows = sorted(records, key=lambda row: int(row["task_id"]))
    if not rows or [int(row["task_id"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Expected consecutive dataset-local task IDs starting at zero")
    for row in rows:
        if row["split"] not in {"train", "test"}:
            raise ValueError(f"Invalid split for task {row['task_id']}")
        digest = row.get("processed_text_sha256")
        if digest and hashlib.sha256(row["processed_text"].encode("utf-8")).hexdigest() != digest:
            raise ValueError(f"S27 text hash mismatch for task {row['task_id']}")
    return rows


def normalize_rows(matrix):
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(lengths > 0, lengths, 1)


def numpy_tfidf_svd(texts):
    """Rebuild the all-corpus protocol. Vocabulary tie order and SVD signs are explicit."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    counts = []
    total = Counter()
    for text in texts:
        words = [w for w in re.findall(r"(?u)\b\w\w+\b", text.lower()) if w not in ENGLISH_STOP_WORDS]
        terms = Counter(words)
        terms.update(a + " " + b for a, b in zip(words, words[1:]))
        counts.append(terms)
        total.update(terms)
    vocab = sorted(sorted(total, key=lambda term: (-total[term], term))[:5000])
    width = min(128, len(texts) - 1, len(vocab))
    if width < 1:
        raise ValueError("TF-IDF/SVD requires at least two fitted texts and a nonempty vocabulary")
    indices = {term: i for i, term in enumerate(vocab)}
    tf = np.zeros((len(texts), len(vocab)))
    for row, terms in enumerate(counts):
        for term, count in terms.items():
            if term in indices:
                tf[row, indices[term]] = count
    idf = np.log((1 + len(texts)) / (1 + np.count_nonzero(tf, axis=0))) + 1
    matrix = normalize_rows(tf * idf)
    rng = np.random.RandomState(42)
    subspace = rng.normal(size=(len(vocab), min(width + 10, *matrix.shape)))
    q = np.linalg.qr(matrix @ subspace, mode="reduced")[0]
    for _ in range(5):
        right = np.linalg.qr(matrix.T @ q, mode="reduced")[0]
        q = np.linalg.qr(matrix @ right, mode="reduced")[0]
    _, singular, right = np.linalg.svd(q.T @ matrix, full_matrices=False)
    components = right[:width]
    pivots = components[np.arange(width), np.argmax(np.abs(components), axis=1)]
    components *= np.where(pivots < 0, -1., 1.)[:, None]
    embedding = normalize_rows(matrix @ components.T).astype(np.float32)
    return embedding, dict(vocabulary=np.asarray(vocab), idf=idf, components=components,
                           singular_values=singular[:width])


def train_only_tfidf_svd(texts, train):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    if len(train) < 2:
        raise ValueError("Train-only TF-IDF/SVD requires at least two training tasks")
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    fitted = vectorizer.fit_transform([texts[i] for i in train])
    all_rows = vectorizer.transform(texts)
    width = min(128, len(train) - 1, fitted.shape[1])
    if fitted.shape[1] == 1:
        # For a single feature the right singular vector is the positive unit scalar.
        svd = None
        projected = all_rows.toarray()
    else:
        svd = TruncatedSVD(n_components=width, n_iter=5, random_state=42)
        svd.fit(fitted)
        projected = svd.transform(all_rows)
    return normalize_rows(projected).astype(np.float32), dict(vectorizer=vectorizer, svd=svd)


def derangement(size, rng):
    if size < 2:
        raise ValueError("A shuffled-description control requires at least two assays per shuffled pool")
    ids = np.arange(size)
    while True:
        shuffled = rng.permutation(size)
        if np.all(shuffled != ids):
            return shuffled


def build_revision_resources(rows, directory, scope, seeds=(0, 1, 2)):
    """Generate new resource files in S28 layout. Input must be one physical dataset."""
    import sklearn
    rows = ordered_rows(rows)
    names = [r["task_name"] for r in rows]
    ids = np.arange(len(rows))
    train = [i for i, r in enumerate(rows) if r["split"] == "train"]
    test = [i for i, r in enumerate(rows) if r["split"] == "test"]
    all_corpus = scope == "all-corpus"
    if scope not in {"all-corpus", "meta-train-only"}:
        raise ValueError(f"Unknown fit scope: {scope}")
    dest = directory / "all_corpus" if all_corpus else directory
    dest.mkdir(parents=True, exist_ok=True)
    meta = dict(dataset=rows[0]["dataset"], encoder="tfidf", fit_scope=scope,
                fit_task_ids=ids.tolist() if all_corpus else train,
                implementation="numpy" if all_corpus else "scikit-learn",
                numpy_version=np.__version__, sklearn_version=sklearn.__version__,
                provenance="Recomputed from supplied S27 texts; use archived objects for exact replay.")

    def save(stem, matrix, settings, fitted=None):
        details = dict(meta, variant=stem, output_embedding_dim=matrix.shape[1], **settings)
        if all_corpus:
            np.savez_compressed(dest / (stem + ".npz"), task_ids=ids, task_names=np.asarray(names),
                                embedding_matrix=matrix, **(fitted or {}))
        else:
            torch.save(dict(task_ids=ids.tolist(), property_names=names,
                            embedding_matrix=torch.from_numpy(matrix), metadata=details),
                       dest / (stem + ".pt"))
            if fitted is not None:
                with (dest / (stem + ".pkl")).open("wb") as stream:
                    pickle.dump(dict(fitted, metadata=details), stream)
        with (dest / (stem + ".json")).open("w", encoding="utf-8") as stream:
            json.dump(details, stream, indent=2, ensure_ascii=False)

    for stem, column in [("full", "processed_text"), ("name", "names_only_text")]:
        texts = [r[column] for r in rows]
        if all_corpus:
            matrix, fitted = numpy_tfidf_svd(texts)
            v = len(fitted["vocabulary"])
        else:
            matrix, fitted = train_only_tfidf_svd(texts, train)
            v = len(fitted["vectorizer"].vocabulary_)
        fitted_width = matrix.shape[1]
        if stem == "full":
            full = matrix
        else:
            padding = full.shape[1] - fitted_width
            if padding < 0:
                raise ValueError("Names-only fit exceeds Full width; verify the S27 inputs")
            matrix = np.pad(matrix, ((0, 0), (0, padding)))
        save(stem, matrix, dict(fitted_dimension=fitted_width, vocabulary_size=v, svd_seed=42), fitted)
    for seed in seeds:
        rng = np.random.RandomState(seed)
        perm = ids.copy()
        pools = [ids] if all_corpus else [np.asarray(train), np.asarray(test)]
        for pool in pools:
            if len(pool):
                perm[pool] = pool[derangement(len(pool), rng)]
        save(f"shuffled_seed{seed}", full[perm],
             dict(control_seed=seed, source_task_ids=perm.tolist(),
                  permutation_scope="partition" if all_corpus else "within each split"))
        random_matrix = normalize_rows(np.random.RandomState(seed).normal(size=full.shape))
        save(f"random_seed{seed}", random_matrix.astype(np.float32),
             dict(control_seed=seed, distribution="L2-normalized standard Gaussian"))
    return full


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
    
    parser.add_argument("--text_table", default="Supplementary_Processed_Texts.csv",
                        help="Electronic Table S27, in its original CSV format")
    parser.add_argument("--split_table", help="Optional S26 CSV for checking the split labels in S27")
    parser.add_argument("--fit_scope", choices=["all-corpus", "meta-train-only", "both", "legacy"],
                        default="all-corpus", help="TF-IDF resource to generate. Legacy exports the old PT layout.")
    args = parser.parse_args()
    by_dataset = read_text_table(args.text_table)
    args.encoder = normalize_semantic_encoder(args.encoder)
    
    # 确定要处理的数据集
    if args.dataset == "all":
        datasets = sorted(by_dataset)
    else:
        datasets = [args.dataset]

    if args.encoder == "tfidf" and args.fit_scope != "legacy":
        split_data = read_text_table(args.split_table) if args.split_table else None
        scopes = ["all-corpus", "meta-train-only"] if args.fit_scope == "both" else [args.fit_scope]
        output = Path(args.output_dir or PROJECT_ROOT / "recomputed" / "text_embeddings")
        for dataset in datasets:
            rows = ordered_rows(by_dataset[dataset])
            if split_data is not None:
                labels = [(int(r["task_id"]), r["split"]) for r in split_data[dataset]]
                if sorted(labels) != [(int(r["task_id"]), r["split"]) for r in rows]:
                    raise ValueError(f"S26 and S27 task IDs or splits differ for {dataset}")
            for scope in scopes:
                full = build_revision_resources(rows, output / dataset, scope)
                if args.visualize:
                    visualize_similarity(torch.from_numpy(full @ full.T), [r["task_name"] for r in rows],
                                         output / dataset / f"{scope}_similarity.png")
        return
    
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
        
        rows = sorted(by_dataset[dataset], key=lambda row: int(row["task_id"]))
        if [int(row["task_id"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"Invalid dataset-local task IDs for {dataset}")
        descriptions = {row["task_name"]: row["processed_text"] for row in rows}
        if len(descriptions) != len(rows):
            raise ValueError(f"Duplicate task name in {dataset}")
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
