import os

import torch
import numpy as np
import argparse

from utils import read_qe_files, evaluate_sentence_level, evaluate_word_level
from sklearn.metrics import matthews_corrcoef, f1_score


def unroll(list_of_lists):
    return [element for ell in list_of_lists for element in ell]


def evaluate_word_level_qe(golds, preds):
    preds_bin = [((np.array(p) > 0.5) * 1).tolist() for p in preds]
    golds_flattened = unroll(golds)
    preds_flattened = unroll(preds_bin)
    mcc = matthews_corrcoef(golds_flattened, preds_flattened)
    f1_bad = f1_score(golds_flattened, preds_flattened, average='binary', pos_label=1)
    f1_ok = f1_score(golds_flattened, preds_flattened, average='binary', pos_label=0)
    f1_mult = f1_bad * f1_ok
    print("MCC: {:.4f}".format(mcc))
    print("F1 MULT: {:.4f}".format(f1_mult))
    print("F1 OK: {:.4f}".format(f1_ok))
    print("F1 BAD: {:.4f}".format(f1_bad))
    return mcc, f1_mult, f1_ok, f1_bad


def save_list(fpath, list_of_vals):
    print('Saving to {}'.format(fpath))
    with open(fpath, 'w') as f:
        for i, v in enumerate(list_of_vals):
            if isinstance(v, (int, float)):
                s = '{}'.format(v)
            else:
                s = ' '.join(['{}'.format(v_) for v_ in v])
            f.write(s + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Model checkpoint to be tested.")
    parser.add_argument("--testset", default="data/2020-da.qe.test20.csv",  help="Testset Path.")
    parser.add_argument("--mc_dropout", type=int, default=0)
    parser.add_argument("-s", "--save", type=str, default=None, help="Dir where to save the new explanations")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mc_dropout = args.mc_dropout

    # 1) Load Checkpoint
    from model.xlm_roberta_word_level import load_checkpoint
    model = load_checkpoint(args.checkpoint)
    model.eval()
    model.zero_grad()
    model.to(device)

    # 2) Prepare TESTSET
    data = read_qe_files(args.testset)

    # 3) Run Predictions
    all_scores = []
    y = [d["score"] for d in data]
    w_src = [d["src_tags"] for d in data]
    w_mt = [d["tgt_tags"] for d in data]
    y_hat = []
    w_src_hat = []
    w_mt_hat = []
    batch_size = 1
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    for i, batched_sample in enumerate(batches):
        print('Predicting {}/{}...'.format(i+1, len(batches)), end='\r')

        # prepare sample
        batch = model.prepare_sample(batched_sample, cuda=True, inference=True)
        input_ids = batch['input_ids']
        attn_mask = batch['attention_mask']
        mt_eos_ids = batch['mt_eos_ids']
        fs_mask = batch['first_sentence_mask']
        fp_mask = batch['first_piece_mask']

        # predict
        with torch.no_grad():
            if mc_dropout > 0:
                model.train()
                pred_score, wl_scores = [], []
                for _ in range(mc_dropout):
                    ret0, ret1 = model.forward(**batch, return_attentions=False)
                    pred_score.append(ret0)
                    wl_scores.append(ret1)
                pred_score = torch.stack(pred_score).mean(0)
                wl_scores = torch.stack(wl_scores).mean(0)
            else:
                pred_score, wl_scores = model.forward(**batch, return_attentions=False)
            wl_probas = wl_scores[:, :, 1]

        wl_fs_mask = fs_mask.squeeze(0)[fp_mask.squeeze(0).bool()]
        mt_probas = wl_probas.squeeze(0)[wl_fs_mask.bool()].exp()
        src_probas = wl_probas.squeeze(0)[~wl_fs_mask.bool()].exp()

        y_hat.append(pred_score.squeeze().item())
        w_mt_hat.append(mt_probas.tolist())
        w_src_hat.append(src_probas.tolist())

    print('Sentence-level')
    pearson, spearman, mae, rmse = evaluate_sentence_level(y, y_hat)
    print('-----------------')
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
    print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(pearson, spearman, mae/100, rmse/100), end='\t')
    print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(src_mcc, src_f1_mult, src_f1_ok, src_f1_bad), end='\t')
    print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(tgt_mcc, tgt_f1_mult, tgt_f1_ok, tgt_f1_bad), end='\t')
    print('{:.4f}\t{:.4f}\t{:.4f}'.format(mt_auc_score, mt_ap_score, mt_rec_topk), end='\t')
    print('{:.4f}\t{:.4f}\t{:.4f}'.format(src_auc_score, src_ap_score, src_rec_topk))

    if args.save is not None:
        print('Saving word level explanations...')
        if not os.path.exists(args.save):
            os.mkdir(args.save)
        save_list(os.path.join(args.save, 'sentence_scores.txt'), y_hat)
        save_list(os.path.join(args.save, 'aggregated_source_scores.txt'), w_src_hat)
        save_list(os.path.join(args.save, 'source_scores.txt'), w_src_hat)
        save_list(os.path.join(args.save, 'aggregated_mt_scores.txt'), w_mt_hat)
        save_list(os.path.join(args.save, 'mt_scores.txt'), w_mt_hat)