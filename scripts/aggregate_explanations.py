import argparse
import os

import numpy as np
import torch

from utils import aggregate_pieces


def save_list(fpath, list_of_vals):
    print('Saving to {}'.format(fpath))
    with open(fpath, 'w') as f:
        for i, v in enumerate(list_of_vals):
            if isinstance(v, (int, float)):
                s = '{}'.format(v)
            else:
                s = ' '.join(['{}'.format(v_) for v_ in v])
            f.write(s + '\n')


def read_sentence_data(model_sent_fh):
    model_scores = [float(line.strip()) for line in open(model_sent_fh, 'r')]
    return model_scores


def read_word_data(model_explanations_fh):
    model_explanations = [list(map(float, line.replace('OK', '0').replace('BAD', '1').split()))
                          for line in open(model_explanations_fh, 'r')]
    return model_explanations


def read_fp_mask_data(fp_mask_fh):
    fp_mask = [list(map(lambda x: int(x.replace('False', '0').replace('True', '1')), line.split()))
               for line in open(fp_mask_fh, 'r')]
    return fp_mask


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def softmax(x):
    return np.exp(x) / np.exp(x).sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_explanations_dname', type=str, required=True)
    parser.add_argument('-r', '--reduction', default='sum', choices=['none', 'first', 'sum', 'mean', 'max'])
    parser.add_argument('-t', '--transform', default='pre', choices=['none', 'pre', 'pos', 'pre_neg'])
    args = parser.parse_args()

    src_model_explanations = read_word_data(os.path.join(args.model_explanations_dname, 'source_scores.txt'))
    mt_model_explanations = read_word_data(os.path.join(args.model_explanations_dname, 'mt_scores.txt'))
    model_scores = read_sentence_data(os.path.join(args.model_explanations_dname, 'sentence_scores.txt'))
    num_samples = len(model_scores)
    src_fp_mask = [[1] for _ in range(num_samples)]
    mt_fp_mask = [[1] for _ in range(num_samples)]
    if os.path.exists(os.path.join(args.model_explanations_dname, 'source_fp_mask.txt')):
        src_fp_mask = read_fp_mask_data(os.path.join(args.model_explanations_dname, 'source_fp_mask.txt'))
        mt_fp_mask = read_fp_mask_data(os.path.join(args.model_explanations_dname, 'mt_fp_mask.txt'))

    # hack to fix transquest attn expls since the input is: <s> <src> <mt> </s> instead of <s> <mt> <src> </s>
    if 'transquest' in args.model_explanations_dname and 'attn' in args.model_explanations_dname:
        src_model_explanations, mt_model_explanations = mt_model_explanations, src_model_explanations
        src_fp_mask, mt_fp_mask = mt_fp_mask, src_fp_mask

    is_rembert = 'rembert' in args.model_explanations_dname

    # scores for each word and also for <s> and </s>
    for i in range(num_samples):
        src_model_explanations[i] = torch.tensor(src_model_explanations[i])
        mt_model_explanations[i] = torch.tensor(mt_model_explanations[i])
        src_fp_mask[i] = torch.tensor(src_fp_mask[i])
        mt_fp_mask[i] = torch.tensor(mt_fp_mask[i])

        if args.transform == 'pre':
            src_model_explanations[i] = torch.sigmoid(torch.abs(src_model_explanations[i]))
            mt_model_explanations[i] = torch.sigmoid(torch.abs(mt_model_explanations[i]))

        if args.transform == 'pre_neg':
            src_model_explanations[i] = - src_model_explanations[i]
            mt_model_explanations[i] = - mt_model_explanations[i]

        src_model_explanations[i] = aggregate_pieces(src_model_explanations[i], src_fp_mask[i], args.reduction)
        mt_model_explanations[i] = aggregate_pieces(mt_model_explanations[i], mt_fp_mask[i], args.reduction)

        if args.transform == 'pos':
            src_model_explanations[i] = torch.sigmoid(torch.abs(src_model_explanations[i]))
            mt_model_explanations[i] = torch.sigmoid(torch.abs(mt_model_explanations[i]))

        # remove <s> and </s>
        a = 0 if args.reduction == 'none' else 1
        b = None if args.reduction == 'none' else -1
        src_model_explanations[i] = src_model_explanations[i][a:b].tolist()
        a = 0 if is_rembert else a
        mt_model_explanations[i] = mt_model_explanations[i][a:b].tolist()

    print('Saving aggregated explanations...')
    save_list(os.path.join(args.model_explanations_dname, 'aggregated_source_scores.txt'), src_model_explanations)
    save_list(os.path.join(args.model_explanations_dname, 'aggregated_mt_scores.txt'), mt_model_explanations)


if __name__ == '__main__':
    main()
