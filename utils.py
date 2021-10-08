# -*- coding: utf-8 -*-
from argparse import Namespace
import numpy as np
import torch

from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr, spearmanr


class Config:
    def __init__(self, initial_data: dict) -> None:
        for key in initial_data:
            if hasattr(self, key):
                setattr(self, key, initial_data[key])

    def namespace(self) -> Namespace:
        return Namespace(
            **{
                name: getattr(self, name)
                for name in dir(self)
                if not callable(getattr(self, name)) and not name.startswith("__")
            }
        )


def read_qe_files(path: str, inference=False, reverse_mt_and_src=False, only_src=False):
    def read_file(fpath, transform=lambda x: x):
        data = []
        with open(fpath, 'r', encoding='utf8') as f:
            for line in f:
                data.append(transform(line.strip()))
        return data
    if inference:
        data = {
            'mt': read_file(path + '.mt', transform=lambda x: x.strip()),
            'src': read_file(path + '.src', transform=lambda x: x.strip()),
        }
        if only_src:
            data['mt'] = data['src']
        if reverse_mt_and_src:
            data['mt'], data['src'] = data['src'], data['mt']
    else:
        data = {
            'score': read_file(path + '.da', transform=lambda x: float(x)),
            'hter': read_file(path + '.hter', transform=lambda x: float(x)),
            'mt': read_file(path + '.mt', transform=lambda x: x.strip()),
            'pe': read_file(path + '.pe', transform=lambda x: x.strip()),
            'src': read_file(path + '.src', transform=lambda x: x.strip()),
            'src_tags': read_file(path + '.src-tags', transform=lambda x: list(map(int, x.split()))),
            'tgt_tags': read_file(path + '.tgt-tags', transform=lambda x: list(map(int, x.split()))),
            # 'src_mt_alignments': read_file(path + '.src-mt.alignments', transform=lambda x: list(map(lambda y: list(map(int, y.split('-'))), x.split()))), # noqa
        }
        if only_src:
            data['mt'] = data['src']
            data['tgt_tags'] = data['src_tags']
        if reverse_mt_and_src:
            data['mt'], data['src'] = data['src'], data['mt']
            data['tgt_tags'], data['src_tags'] = data['src_tags'], data['tgt_tags']

    return [dict(zip(data.keys(), t)) for t in list(zip(*data.values()))]


def validate_word_level_data(gold_explanations, model_explanations):
    valid_gold, valid_model = [], []
    for gold_expl, model_expl in zip(gold_explanations, model_explanations):
        if sum(gold_expl) == 0 or sum(gold_expl) == len(gold_expl):
            continue
        else:
            valid_gold.append(gold_expl)
            valid_model.append(model_expl)
    return valid_gold, valid_model


def compute_auc_score(gold_explanations, model_explanations):
    res = 0
    for i in range(len(gold_explanations)):
        res += roc_auc_score(gold_explanations[i], model_explanations[i])
    return res / len(gold_explanations)


def compute_ap_score(gold_explanations, model_explanations):
    res = 0
    for i in range(len(gold_explanations)):
        res += average_precision_score(gold_explanations[i], model_explanations[i])
    return res / len(gold_explanations)


def compute_rec_topk(gold_explanations, model_explanations):
    res = 0
    for i in range(len(gold_explanations)):
        idxs = np.argsort(model_explanations[i])[::-1][:sum(gold_explanations[i])]
        res += len([idx for idx in idxs if gold_explanations[i][idx] == 1])/sum(gold_explanations[i])
    return res / len(gold_explanations)


def evaluate_word_level(gold_explanations, model_explanations):
    gold_explanations, model_explanations = validate_word_level_data(gold_explanations, model_explanations)
    auc_score = compute_auc_score(gold_explanations, model_explanations)
    ap_score = compute_ap_score(gold_explanations, model_explanations)
    rec_topk = compute_rec_topk(gold_explanations, model_explanations)
    print('AUC score: {:.4f}'.format(auc_score))
    print('AP score: {:.4f}'.format(ap_score))
    print('Recall at top-K: {:.4f}'.format(rec_topk))
    return auc_score, ap_score, rec_topk


def evaluate_sentence_level(y_gold, y_hat):
    y_gold = np.array(y_gold)
    y_hat = np.array(y_hat)
    pearson = pearsonr(y_gold, y_hat)[0]
    spearman = spearmanr(y_gold, y_hat)[0]
    mae = np.abs(y_gold - y_hat).mean()
    rmse = ((y_gold - y_hat) ** 2).mean() ** 0.5
    print("Pearson: {:.4f}".format(pearson))
    print("Spearman: {:.4f}".format(spearman))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    return pearson, spearman, mae, rmse


def aggregate_pieces(x, mask, reduction='first'):
    """
    :param x: tensor of shape (seq_len) or (seq_len, hdim)
    :param mask: bool tensor of shape (seq_len)
    :param reduction: aggregation strategy (first, max, sum, mean)

    :returns: <s> word_1 word_2 ... </s>
    where word_i = aggregate(piece_i_1, piece_i_2, ...)
    """
    # mark <s> and </s> as True
    special_mask = mask.clone()
    special_mask[0] = special_mask[-1] = True

    if reduction == 'first':
        return x[special_mask.bool()]

    elif reduction == 'sum' or reduction == 'mean' or reduction == 'max':
        idx = special_mask.long().cumsum(dim=-1) - 1
        idx_unique_count = torch.bincount(idx)
        num_unique = idx_unique_count.shape[-1]
        if reduction == 'sum':
            res = torch.zeros(num_unique, device=x.device, dtype=x.dtype).scatter_add(0, idx, x)
        elif reduction == 'mean':
            res = torch.zeros(num_unique, device=x.device, dtype=x.dtype).scatter_add(0, idx, x)
            res /= idx_unique_count.float()
        else:
            res = torch.stack([x[idx == i].max() for i in range(num_unique)]).to(x.device)
        return res.float()

    else:
        return x
