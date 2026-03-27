import torch


def _topk_mask(adj: torch.Tensor, k: int) -> torch.Tensor:
    """Row-wise top-k mask for a [B, N, N] adjacency (zeroes diagonal)."""
    if k is None or k < 0:
        return (adj > 0).to(adj.dtype)
    B, N, _ = adj.shape
    k = min(k, N)
    tmp = adj.reshape(B * N, N)
    topk = torch.topk(tmp, k=k, dim=-1).indices
    mask = torch.zeros_like(tmp)
    mask.scatter_(1, topk, 1.0)
    mask = mask.reshape(B, N, N)
    return mask


def compute_knowledge_adj(s_list, q_list, which: str = 'both', top_k: int = 5,
                          device=None, split_components: bool = False):
    """
    Build scaffold/group similarity adjacency for each query sample, over (support+query) nodes.
    Returns a dense tensor of shape [n_q, n_s+1, n_s+1] with zeros on diagonal and values in [0,1].
    """
    # gather fingerprints / groupprints
    s_fp = torch.stack([d.fingerprint.float() for d in s_list])  # [n_s, 2214]
    q_fp = torch.stack([d.fingerprint.float() for d in q_list])  # [n_q, 2214]
    s_gp = torch.stack([d.groupprint.float() for d in s_list])   # [n_s, 49]
    q_gp = torch.stack([d.groupprint.float() for d in q_list])   # [n_q, 49]
    if device is not None:
        s_fp, q_fp, s_gp, q_gp = s_fp.to(device), q_fp.to(device), s_gp.to(device), q_gp.to(device)

    # stack into [n_q, n_s+1, D]
    fp = torch.cat([s_fp.unsqueeze(0).repeat(q_fp.shape[0], 1, 1), q_fp.unsqueeze(1)], dim=1)
    gp = torch.cat([s_gp.unsqueeze(0).repeat(q_gp.shape[0], 1, 1), q_gp.unsqueeze(1)], dim=1)

    # binary complement for denominator
    fp_ = 1.0 - fp
    gp_ = 1.0 - gp
    # similarity as in KRGTS: <x,x^T> / (D - <(1-x),(1-x)^T> + eps)
    sca_sim = fp @ fp.transpose(1, 2)
    sca_den = (fp.shape[-1] - (fp_ @ fp_.transpose(1, 2))).clamp_min(1e-9)
    sca = (sca_sim / sca_den).clamp(0.0, 1.0)

    gro_sim = gp @ gp.transpose(1, 2)
    gro_den = (gp.shape[-1] - (gp_ @ gp_.transpose(1, 2))).clamp_min(1e-9)
    gro = (gro_sim / gro_den).clamp(0.0, 1.0)

    # zero diagonal, apply top-k per row
    eye = torch.eye(s_fp.shape[0] + 1, device=fp.device).unsqueeze(0)
    sca = sca * (1 - eye)
    gro = gro * (1 - eye)

    def keep_topk(adj):
        if top_k is None or top_k < 0:
            return adj
        mask = _topk_mask(adj, top_k)
        return adj * mask

    sca = keep_topk(sca)
    gro = keep_topk(gro)

    if split_components:
        def maybe(tensor, allow):
            return tensor if allow else torch.zeros_like(tensor)
        sca_out = maybe(sca, which in ('sca', 'both'))
        gro_out = maybe(gro, which in ('gro', 'both'))
        return sca_out, gro_out

    if which == 'sca':
        return sca
    if which == 'gro':
        return gro
    return torch.maximum(sca, gro)
