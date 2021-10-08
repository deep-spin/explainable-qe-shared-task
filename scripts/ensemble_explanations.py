import argparse
import os

import torch


def save_list(fpath, list_of_vals):
    print('Saving to {}'.format(fpath))
    with open(fpath, 'w') as f:
        for i, v in enumerate(list_of_vals):
            if isinstance(v, (int, float)):
                s = '{}'.format(v)
            else:
                s = ' '.join(['{}'.format(v_) for v_ in v])
            f.write(s + '\n')


def read_sentence_data(sent_filename):
    return [float(line.strip()) for line in open(sent_filename, 'r')]


def read_word_data(explanations_filename):
    return [list(map(float, line.split())) for line in open(explanations_filename, 'r')]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--explainers", type=str, required=True, nargs='+', help="Dirs of explainers")
    parser.add_argument("-w", "--weights", type=str, default='uniform', nargs='+', help="Weights of each explainer")
    parser.add_argument("-s", "--save", type=str, required=True, help="Dir where to save the new explanations")
    args = parser.parse_args()

    explainers = args.explainers
    if args.weights == 'uniform':
        weights = torch.tensor(1) / float(len(explainers))
    else:
        weights = torch.tensor(list(map(float, args.weights)))
        assert len(weights) == len(explainers)

    all_sentence_scores = []
    all_mt_scores = []
    all_source_scores = []
    for expl_dir in explainers:
        all_sentence_scores.append(read_sentence_data(os.path.join(expl_dir, 'sentence_scores.txt')))
        all_mt_scores.append(read_word_data(os.path.join(expl_dir, 'aggregated_mt_scores.txt')))
        all_source_scores.append(read_word_data(os.path.join(expl_dir, 'aggregated_source_scores.txt')))

    w = weights.unsqueeze(-1)
    N = len(all_sentence_scores[0])
    E = len(explainers)
    sentence_scores = []
    mt_scores = []
    source_scores = []
    for i in range(N):
        exs_sent = []
        exs_mt = []
        exs_src = []
        for e in range(E):
            ex_sent = all_sentence_scores[e][i]
            ex_mt = torch.tensor(all_mt_scores[e][i])
            ex_src = torch.tensor(all_source_scores[e][i])
            if 'gxi' in explainers[e]:
                ex_mt = ex_mt.abs().sigmoid()
                ex_src = ex_src.abs().sigmoid()
            if 'hidden' in explainers[e]:
                ex_mt = torch.softmax(ex_mt, dim=-1)
                ex_src = torch.softmax(ex_src, dim=-1)
            exs_sent.append(ex_sent)
            exs_mt.append(ex_mt)
            exs_src.append(ex_src)
        sent_expl = (torch.tensor(exs_sent) * w).sum(0)
        mt_expl = (torch.stack(exs_mt) * w).sum(0)
        src_expl = (torch.stack(exs_src) * w).sum(0)
        sentence_scores.append(sent_expl.item())
        mt_scores.append(mt_expl.tolist())
        source_scores.append(src_expl.tolist())

    print('Saving ensembled explanations...')
    if not os.path.exists(args.save):
        os.mkdir(args.save)
    save_list(os.path.join(args.save, 'sentence_scores.txt'), sentence_scores)
    save_list(os.path.join(args.save, 'aggregated_source_scores.txt'), source_scores)
    save_list(os.path.join(args.save, 'aggregated_mt_scores.txt'), mt_scores)
