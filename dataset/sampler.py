import numpy as np
import random
from collections import defaultdict


def _group_by_scaffold(dataset, indices):
    groups = defaultdict(list)
    if hasattr(dataset, 'get_scaffold_id'):
        for idx in indices:
            groups[dataset.get_scaffold_id(idx)].append(idx)
    else:
        for idx in indices:
            groups[idx].append(idx)
    return [grp for grp in groups.values() if grp]


def _scaffold_balanced_sample(dataset, candi_list, num):
    groups = _group_by_scaffold(dataset, candi_list)
    for grp in groups:
        random.shuffle(grp)
    random.shuffle(groups)
    selected = []
    while groups and len(selected) < num:
        next_groups = []
        for grp in groups:
            if grp:
                selected.append(grp.pop())
                if len(selected) == num:
                    break
            if grp:
                next_groups.append(grp)
        groups = next_groups
    if len(selected) < num:
        remaining = [idx for grp in groups for idx in grp if idx not in selected]
        pool = [idx for idx in candi_list if idx not in selected]
        if remaining:
            pool = remaining
        if pool:
            need = num - len(selected)
            selected.extend(random.sample(pool, min(len(pool), need)))
    return selected


def sample_from_candi(dataset, candi_list, num, mode='random'):
    if len(candi_list) == 0 or num <= 0:
        return []
    if len(candi_list) <= num:
        return list(candi_list)
    if mode == 'scaffold':
        return _scaffold_balanced_sample(dataset, candi_list, num)
    return np.random.choice(candi_list, num, replace=False).tolist()


def dataset_sampler(dataset, n_support, n_query, tgt_id, inductive=False, sampling='random'):
    tgt_index_list = dataset.index_list[tgt_id]
    class0_num, class1_num = len(tgt_index_list[0]), len(tgt_index_list[1])
    if class0_num > n_support and class1_num > n_support:
        support_list_i_0 = sample_from_candi(dataset, tgt_index_list[0], n_support, sampling)
        support_list_i_1 = sample_from_candi(dataset, tgt_index_list[1], n_support, sampling)
    elif class0_num <= n_support < class1_num:
        # class 0 not enough
        support_list_i_0 = sample_from_candi(dataset, tgt_index_list[0], max(class0_num - 1, 1), sampling)
        support_list_i_1 = sample_from_candi(dataset, tgt_index_list[1], 2 * n_support - class0_num + 1, sampling)
    else:
        support_list_i_0 = sample_from_candi(dataset, tgt_index_list[0], 2 * n_support - class1_num + 1, sampling)
        support_list_i_1 = sample_from_candi(dataset, tgt_index_list[1], max(class1_num - 1, 1), sampling)
    support_list = support_list_i_0 + support_list_i_1

    if not inductive:
        query_candi_i_0 = [idx for idx in tgt_index_list[0] if idx not in support_list]
        query_candi_i_1 = [idx for idx in tgt_index_list[1] if idx not in support_list]
        query_list = sample_from_candi(dataset, query_candi_i_0, 1, sampling) + \
                     sample_from_candi(dataset, query_candi_i_1, 1, sampling)
        query_candi = [idx for idx in query_candi_i_0 + query_candi_i_1 if idx not in query_list]
        query_list += sample_from_candi(dataset, query_candi, n_query - 2, sampling)
    else:
        query_list = [idx for idx in tgt_index_list[0] + tgt_index_list[1]
                      if idx not in support_list]
    return dataset[support_list], dataset[query_list]
