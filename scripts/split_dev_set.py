import argparse
import os

import numpy as np

from sklearn.model_selection import train_test_split


def read_sentence_data(gold_sent_fh):
    gold_scores = [float(line.strip()) for line in open(gold_sent_fh, 'r')]
    return gold_scores


def read_data(fname):
    data = [line.strip() for line in open(fname, 'r')]
    return data


def save_data(fname, data):
    f = open(fname, 'w', encoding='utf8')
    for line in data:
        f.write(line + '\n')
    f.close()


def split_data(fname, idxs_first, idxs_second):
    data = read_data(fname)
    data_first = [data[i] for i in idxs_first]
    data_second = [data[i] for i in idxs_second]
    new_name_first = fname.replace('dev.', 'dev.set1.')
    new_name_second = fname.replace('dev.', 'dev.set2.')
    save_data(new_name_first, data_first)
    save_data(new_name_second, data_second)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lp-dir', type=str, required=True, default='data/ro-en/')
    args = parser.parse_args()

    scores = read_sentence_data(os.path.join(args.lp_dir, 'dev.da'))
    scores = np.array(scores)
    min = np.amin(scores)
    max = np.amax(scores)
    bins = np.linspace(min, max, 21)
    bins = bins[1:]  # ignore first bin to create the interval (0, bin[1])
    y = np.digitize(scores, bins, right=True)
    x = np.arange(len(y))
    idxs_first, idxs_second, _, _ = train_test_split(x, y, stratify=y, test_size=0.5)
    idxs_first = idxs_first.tolist()
    idxs_second = idxs_second.tolist()

    print(bins)
    print([(y == i).sum() for i in range(20)])
    print(len(idxs_first))
    print(len(idxs_second))

    all_fnames = ['dev.da', 'dev.mt', 'dev.src', 'dev.src-tags', 'dev.hter', 'dev.pe',
                  'dev.src-mt.alignments', 'dev.tgt-tags']
    for fname in all_fnames:
        split_data(os.path.join(args.lp_dir, fname), idxs_first, idxs_second)

    save_data(os.path.join(args.lp_dir, 'dev.set1.idxs'), list(map(str, idxs_first)))
    save_data(os.path.join(args.lp_dir, 'dev.set2.idxs'), list(map(str, idxs_second)))
