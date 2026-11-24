from __future__ import annotations

from typing import Any, Optional


SEMANTIC_ENCODER_ALIASES = {
    "tfidf": "tfidf",
    "bert": "bert",
    "sbert": "sbert",
    "sentence-transformers": "sbert",
    "sentence_transformers": "sbert",
}

SEMANTIC_EMBEDDING_DIRS = {
    "tfidf": "data/embeddings",
    "bert": "data/embeddings_bert",
    "sbert": "data/embeddings_sbert",
}


def normalize_semantic_encoder(name: Optional[str], default: str = "tfidf") -> str:
    key = default if name is None else str(name).strip().lower()
    key = key.replace("_", "-")
    normalized = SEMANTIC_ENCODER_ALIASES.get(key)
    if normalized is None:
        valid = ", ".join(sorted(SEMANTIC_EMBEDDING_DIRS))
        raise ValueError(f"Unknown semantic encoder '{name}'. Valid options: {valid}")
    return normalized


def default_semantic_embedding_path(dataset: str, encoder: Optional[str]) -> str:
    normalized = normalize_semantic_encoder(encoder)
    base_dir = SEMANTIC_EMBEDDING_DIRS[normalized]
    return f"{base_dir}/{dataset}/{dataset}_property_embeddings.pt"


def validate_semantic_embedding_metadata(
    data: dict[str, Any],
    expected_encoder: Optional[str],
    embedding_path: str,
    consumer: str,
) -> Optional[str]:
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    actual_encoder = metadata.get("encoder")
    if actual_encoder is None:
        return None

    try:
        normalized_actual = normalize_semantic_encoder(actual_encoder)
    except ValueError:
        print(
            f"[{consumer}] Warning: embedding metadata in '{embedding_path}' "
            f"contains unknown encoder '{actual_encoder}'."
        )
        return str(actual_encoder)

    if expected_encoder is not None:
        normalized_expected = normalize_semantic_encoder(expected_encoder)
        if normalized_actual != normalized_expected:
            print(
                f"[{consumer}] Warning: requested semantic encoder "
                f"'{normalized_expected}', but '{embedding_path}' was generated "
                f"with '{normalized_actual}'."
            )

    return normalized_actual

def load_task_embeddings(path):
    """Read native S28 resources in dataset-local task order."""
    import numpy as np
    import torch
    if str(path).lower().endswith(".npz"):
        with np.load(path, allow_pickle=False) as archive:
            ids = archive["task_ids"]
            if not np.array_equal(ids, np.arange(len(ids))):
                raise ValueError("Embedding rows must follow dataset-local task IDs")
            matrix = torch.as_tensor(archive["embedding_matrix"], dtype=torch.float32)
            names = archive["task_names"].tolist()
            if matrix.ndim != 2 or matrix.shape[0] != len(ids) or len(names) != len(ids):
                raise ValueError("Inconsistent embedding dimensions")
            return {"embedding_matrix": matrix, "property_names": names,
                    "metadata": {"encoder": "tfidf"}}
    data = torch.load(path, map_location="cpu")
    if isinstance(data, torch.Tensor):
        return {"embedding_matrix": data}
    return data
