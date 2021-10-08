# -*- coding: utf-8 -*-
import torch


def lengths_to_mask(*lengths, **kwargs):
    lengths = [l.squeeze().tolist() if torch.is_tensor(l) else l for l in lengths]

    # For cases where length is a scalar, this needs to convert it to a list.
    lengths = [l if isinstance(l, list) else [l] for l in lengths]
    assert all(len(l) == len(lengths[0]) for l in lengths)
    batch_size = len(lengths[0])
    other_dimensions = tuple([int(max(l)) for l in lengths])
    mask = torch.zeros(batch_size, *other_dimensions, **kwargs)
    for i, length in enumerate(zip(*tuple(lengths))):
        mask[i][[slice(int(l)) for l in length]].fill_(1)
    return mask.bool()


def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    """ Moves a sample to cuda. Works with dictionaries, tensors and lists. """

    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def move_to_cpu(sample):
    """ Moves a sample to cuda. Works with dictionaries, tensors and lists. """

    def _move_to_cpu(tensor):
        return tensor.cpu()

    return apply_to_sample(_move_to_cpu, sample)


def masked_average(tensor, mask):
    """ Performs masked average of a given tensor at time dim. """
    tensor_sum = (tensor * mask.float().unsqueeze(-1)).sum(1)
    tensor_mean = tensor_sum / mask.sum(-1).float().unsqueeze(-1)
    return tensor_mean


def pad_sequence(list_of_tensors, padding_value=-100, end=True):
    if end:
        return torch.nn.utils.rnn.pad_sequence(list_of_tensors, batch_first=True, padding_value=padding_value)
    bs = len(list_of_tensors)
    seq_len = max([t.shape[0] for t in list_of_tensors])
    t = list_of_tensors[0]
    if t.dim() > 1:
        new_tensor = torch.zeros(bs, seq_len, t.shape[-1], dtype=t.dtype, device=t.device) + padding_value
    else:
        new_tensor = torch.zeros(bs, seq_len, dtype=t.dtype, device=t.device) + padding_value
    for i, t in enumerate(list_of_tensors):
        new_tensor[i, -t.shape[0]:] = t
    return new_tensor


def confusion_matrix(y_pred, y_true):
    device = y_pred.device
    labels = max(y_pred.max().item() + 1, y_true.max().item() + 1)

    return (
        (
                torch.stack((y_true, y_pred), -1).unsqueeze(-2).unsqueeze(-2)
                == torch.stack(
            (
                torch.arange(labels, device=device).unsqueeze(-1).repeat(1, labels),
                torch.arange(labels, device=device).unsqueeze(-2).repeat(labels, 1),
            ),
            -1,
        )
        )
            .all(-1)
            .sum(-3)
    )


def accuracy_precision_recall_f1(y_pred, y_true, average=True):
    M = confusion_matrix(y_pred, y_true)

    tp = M.diagonal(dim1=-2, dim2=-1).float()

    precision_den = M.sum(-2)
    precision = torch.where(
        precision_den == 0, torch.zeros_like(tp), tp / precision_den
    )

    recall_den = M.sum(-1)
    recall = torch.where(recall_den == 0, torch.ones_like(tp), tp / recall_den)

    f1_den = precision + recall
    f1 = torch.where(
        f1_den == 0, torch.zeros_like(tp), 2 * (precision * recall) / f1_den
    )

    return ((y_pred == y_true).float().mean(-1),) + (
        tuple(e.mean(-1) for e in (precision, recall, f1))
        if average
        else (precision, recall, f1)
    )
