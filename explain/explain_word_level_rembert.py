import os

import torch
import argparse

from utils import read_qe_files


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
    parser.add_argument("--revert-mt-src", action='store_true')
    parser.add_argument("-s", "--save", type=str, required=True, help="Dir where to save the new explanations")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mc_dropout = args.mc_dropout

    # 1) Load Checkpoint
    from model.rembert_word_level import load_checkpoint
    model = load_checkpoint(args.checkpoint)
    model.eval()
    model.zero_grad()
    model.to(device)

    # 2) Prepare TESTSET
    data = read_qe_files(args.testset, inference=True)

    # 3) Run Predictions
    all_scores = []
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
        fs_mask = ~batch['token_type_ids'].bool()
        fp_mask = batch['first_piece_mask'].bool()

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
        w_mt_hat.append(mt_probas.squeeze().tolist())
        w_src_hat.append(src_probas.squeeze().tolist())

    if args.revert_mt_src:
        w_src_hat, w_mt_hat = w_mt_hat, w_src_hat

    print('Saving word level explanations...')
    if not os.path.exists(args.save):
        os.mkdir(args.save)
    save_list(os.path.join(args.save, 'sentence_scores.txt'), y_hat)
    save_list(os.path.join(args.save, 'aggregated_source_scores.txt'), w_src_hat)
    save_list(os.path.join(args.save, 'aggregated_mt_scores.txt'), w_mt_hat)
