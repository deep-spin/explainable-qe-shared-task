# pip3 install transquest
# python3 evaluate_transquest.py --testset data/ro-en/dev --checkpoint TransQuest/monotransquest-da-ro_en-wiki
# python3 evaluate_transquest.py --testset data/et-en/dev --checkpoint TransQuest/monotransquest-da-et_en-wiki

import torch
from evaluate_word_level import evaluate_word_level_qe, unroll
from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel


from scipy.stats import pearsonr, spearmanr
import numpy as np
import argparse

from utils import read_qe_files, evaluate_word_level

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Model checkpoint to be tested.")
    parser.add_argument("--testset", default="data/2020-da.qe.test20.csv",  help="Testset Path.")
    args = parser.parse_args()

    # Load model
    model = MicroTransQuestModel('xlmroberta', args.checkpoint, labels=["OK", "BAD"], use_cuda=torch.cuda.is_available())
    data = read_qe_files(args.testset)

    # Evaluate predictions on the dataset
    src = [d['src'] for d in data]
    mt = [d['mt'] for d in data]
    y = [d["score"] for d in data]
    w_src = [d["src_tags"] for d in data]
    w_mt = [d["tgt_tags"] for d in data]
    w_src_hat = []
    w_mt_hat = []
    w_src_hat, w_mt_hat = model.predict(list(map(list, zip(src, mt))))
    w_src_hat = [[int(w == 'BAD') for w in ws] for ws in w_src_hat]
    w_mt_hat = [[int(w == 'BAD') for w in ws] for ws in w_mt_hat]

    print('Src Word-level')
    src_mcc, src_f1_mult, src_f1_ok, src_f1_bad = evaluate_word_level_qe(w_src, w_src_hat)
    print('-----------------')
    print('Tgt Word-level')
    tgt_mcc, tgt_f1_mult, tgt_f1_ok, tgt_f1_bad = evaluate_word_level_qe(w_mt, w_mt_hat)
    print('-----------------')
    print('Expl MT word-level')
    mt_auc_score, mt_ap_score, mt_rec_topk = evaluate_word_level(w_mt, w_mt_hat)
    print('-----------------')
    print('Expl source word-level')
    src_auc_score, src_ap_score, src_rec_topk = evaluate_word_level(w_src, w_src_hat)
    print('-----------------')

    # print to google docs
    print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(0, 0, 0, 0), end='\t')
    print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(src_mcc, src_f1_mult, src_f1_ok, src_f1_bad), end='\t')
    print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(tgt_mcc, tgt_f1_mult, tgt_f1_ok, tgt_f1_bad), end='\t')
    print('{:.4f}\t{:.4f}\t{:.4f}'.format(mt_auc_score, mt_ap_score, mt_rec_topk), end='\t')
    print('{:.4f}\t{:.4f}\t{:.4f}'.format(src_auc_score, src_ap_score, src_rec_topk))
