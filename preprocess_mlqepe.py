import argparse
import shutil

import pandas as pd
import os

from utils import read_qe_files


def save_data_to_file(fname, data):
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    with open(fname, 'w') as f:
        for x in data:
            f.write('{}\n'.format(x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ln -s /mnt/data-zeus2/mtreviso/data/corpus/mlqe-pe mlqe-pe
    # mkdir data/mlqepe
    parser.add_argument('--input-dir', help='Base dir for mlqe-pe dataset.', default='mlqe-pe/data/')
    parser.add_argument('--output-dir', help='Base dir to store LPs data.', default='data/mlqepe/')
    args = parser.parse_args()

    lps = ['en-de', 'en-zh', 'et-en', 'ne-en', 'ro-en', 'ru-en', 'si-en']
    splits = ['dev', 'train', 'test']

    # .da
    for split in splits:
        for lp in lps:
            # e.g. mlqe-pe/data/direct-assessments/dev/en-de-dev/dev.ende.df.short.tsv
            if split == 'test':
                ipath = os.path.join(args.input_dir, 'direct-assessments/{}/{}/{}20.{}.df.short.tsv'.format(
                    split, lp, split, lp.replace('-', '')
                ))
            else:
                ipath = os.path.join(args.input_dir, 'direct-assessments/{}/{}-{}/{}.{}.df.short.tsv'.format(
                    split, lp, split, split, lp.replace('-', '')
                ))
            opath = os.path.join(args.output_dir, '{}/{}.da'.format(lp, split))
            df = pd.read_csv(ipath, sep='\t', usecols=[1, 2, 3, 4, 5], quoting=3)  # some files have " tokens
            save_data_to_file(opath, df["mean"].tolist())

    # .hter, .mt, .pe, .src, .src-mt.alignments, .src_tags, .tgt_tags
    exts = {
        'hter': 'hter',
        'mt': 'mt',
        'pe': 'pe',
        'src': 'src',
        'src-mt.alignments': 'src-mt.alignments',
        'source_tags': 'src-tags',
        'tags': 'tgt-tags'
    }
    for split in splits:
        for lp in lps:
            split20 = 'test20' if split == 'test' else split
            for iext, oext in exts.items():
                ipath = os.path.join(args.input_dir, 'post-editing/{}/{}-{}/{}.{}'.format(
                    split, lp, split20, split20, iext
                ))
                opath = os.path.join(args.output_dir, '{}/{}.{}'.format(lp, split, oext))
                assert os.path.exists(ipath)
                shutil.copy(ipath, opath)

    # fix word-level tags:
    # OK -> 0 and BAD -> 1 in .src_tags and .tgt_tags
    # gaps are ignored in .tgt_tags
    for split in splits:
        for lp in lps:
            src_path = os.path.join(args.output_dir, '{}/{}.{}'.format(lp, split, 'src-tags'))
            tgt_path = os.path.join(args.output_dir, '{}/{}.{}'.format(lp, split, 'tgt-tags'))
            src_data = open(src_path, encoding='utf8').readlines()
            tgt_data = open(tgt_path, encoding='utf8').readlines()
            src_data = [x.strip().replace('OK', '0').replace('BAD', '1') for x in src_data]
            # x = <gap> <word> <gap> <word> <gap> ...
            # x[1::2] will select <word> tags
            tgt_data = [' '.join(x.strip().replace('OK', '0').replace('BAD', '1').split()[1::2]) for x in tgt_data]
            # sanity check: number of mt tags should match the number of mt tokens
            tgt_words = open(tgt_path.replace('tgt-tags', 'mt'), encoding='utf8').readlines()
            for x, y in zip(tgt_words, tgt_data):
                assert len(x.strip().split()) == len(y.split())
            save_data_to_file(src_path, src_data)
            save_data_to_file(tgt_path, tgt_data)

    # concat all LPs:
    def save_list_of_data_to_file(fpath, list_of_data, filter_lp_rule=lambda lp: True):
        with open(fpath, 'w') as f:
            for lp, data in list_of_data:
                if filter_lp_rule(lp):
                    for line in data:
                        f.write(line.strip() + '\n')

    try:
        os.makedirs(os.path.join(args.output_dir, 'all-en/'))
        os.makedirs(os.path.join(args.output_dir, 'en-all/'))
        os.makedirs(os.path.join(args.output_dir, 'all-all/'))
    except FileExistsError:
        pass
    exts = ['hter', 'mt', 'pe', 'src', 'src-mt.alignments', 'src-tags', 'tgt-tags', 'da']
    for split in splits:
        for ext in exts:
            all_data = []
            for lp in lps:
                ipath = os.path.join(os.path.join(args.output_dir, '{}/{}.{}'.format(lp, split, ext)))
                data = open(ipath, encoding='utf8').readlines()
                all_data.append((lp, data))
            opath_all_en = os.path.join(args.output_dir, 'all-en/{}.{}'.format(split, ext))
            opath_en_all = os.path.join(args.output_dir, 'en-all/{}.{}'.format(split, ext))
            opath_all_all = os.path.join(args.output_dir, 'all-all/{}.{}'.format(split, ext))
            save_list_of_data_to_file(opath_all_en, all_data, filter_lp_rule=lambda lp: lp.endswith('-en'))
            save_list_of_data_to_file(opath_en_all, all_data, filter_lp_rule=lambda lp: lp.startswith('en-'))
            save_list_of_data_to_file(opath_all_all, all_data, filter_lp_rule=lambda lp: True)
            del all_data
