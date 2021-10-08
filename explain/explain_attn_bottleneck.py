import os

import torch
from model.xlm_roberta_bottlenecked import load_checkpoint

import argparse

from utils import read_qe_files


def save_list(fpath, list_of_vals):
    # save predictions and explanations
    if not os.path.exists(os.path.dirname(fpath)):
        os.makedirs(os.path.dirname(fpath))
    print('Saving to {}'.format(fpath))
    with open(fpath, 'w') as f:
        for i, v in enumerate(list_of_vals):
            if isinstance(v, (int, float)):
                s = '{}'.format(v)
            else:
                s = ' '.join(['{}'.format(v_) for v_ in v])
            f.write(s + '\n')


def get_valid_explanations(explanations, attn_mask, fs_mask, fp_mask):
    # input = <s>  mt  </s>    </s>  src  </s>
    #         -------------
    #         first sentence
    e_mt_hat = []
    e_src_hat = []
    e_mt_fp_mask = []
    e_src_fp_mask = []
    for j in range(explanations.shape[0]):
        seq_len = attn_mask[j].sum().item()
        explanations_j = explanations[j][:seq_len]
        fs_mask_j = fs_mask[j][:seq_len]
        fp_mask_j = fp_mask[j][:seq_len]

        mt_explanations = explanations_j[fs_mask_j.bool()].detach().squeeze()
        src_explanations = explanations_j[~fs_mask_j.bool()].detach().squeeze()
        mt_fp_mask = fp_mask_j[fs_mask_j.bool()].detach().squeeze()
        src_fp_mask = fp_mask_j[~fs_mask_j.bool()].detach().squeeze()

        # scores for each word and also for <s> and </s>
        # mt_explanations = aggregate_pieces(mt_explanations, mt_fp_mask, args.reduction)
        # src_explanations = aggregate_pieces(src_explanations, src_fp_mask, args.reduction)
        # cut out <s> and </s>
        # mt_explanations = mt_explanations[1:-1]
        # src_explanations = src_explanations[1:-1]

        # to cpu
        e_mt_hat.append(mt_explanations.cpu().numpy())
        e_src_hat.append(src_explanations.cpu().numpy())
        e_mt_fp_mask.append(mt_fp_mask.cpu().numpy())
        e_src_fp_mask.append(src_fp_mask.cpu().numpy())

    return list(zip(e_mt_hat, e_src_hat, e_mt_fp_mask, e_src_fp_mask))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", help="Model checkpoint to be tested.")
    parser.add_argument("-t", "--testset", default="data/2020-da.qe.test20.csv",  help="Testset Path.")
    parser.add_argument("-e", "--explainer", default="ig", help="Explainability method.")
    parser.add_argument("-r", "--reduction", default="sum", choices=['first', 'sum', 'mean', 'max'])
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-s", "--save", type=str, default="experiments/explanations/tmp/", help="save dir")
    parser.add_argument("--norm-attention", action='store_true')
    parser.add_argument("--norm-strategy", type=str, default='weighted_norm')
    parser.add_argument("--mc_dropout", default=False)
    # args.save = 'experiments/explanations/roen_attn'

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load Checkpoint
    model = load_checkpoint(args.checkpoint, output_norm=args.norm_attention, norm_strategy=args.norm_strategy)
    model.eval()
    model.zero_grad()
    model.to(device)

    aggregation = model.bottleneck.aggregation
    num_labels = model.num_labels
    target = None if num_labels == 1 else 0
    if args.mc_dropout:
        model.set_mc_dropout(int(args.mc_dropout))

    # 2) Prepare TESTSET
    data = read_qe_files(args.testset)

    # 3) Run Predictions
    y_gold = [sample["score"] for sample in data]
    y_hat = []
    e_mt_gold = [sample["tgt_tags"] for sample in data]
    e_src_gold = [sample["src_tags"] for sample in data]
    batches = [data[i:i + args.batch_size] for i in range(0, len(data), args.batch_size)]

    num_layers = 24
    num_heads = 16
    explanations_all = []
    explanations_layers = [[] for _ in range(num_layers)]
    explanations_heads = [[[] for _ in range(num_heads)] for _ in range(num_layers)]

    for i, batched_sample in enumerate(batches):
        print('Explaining {}/{}...'.format(i+1, len(batches)), end='\r')

        # prepare sample
        batch, targets = model.prepare_sample(batched_sample, cuda=True)
        input_ids = batch['input_ids']
        attn_mask = batch['attention_mask']
        mt_eos_ids = batch['mt_eos_ids']
        fs_mask = batch['first_sentence_mask']
        fp_mask = batch['first_piece_mask']
        gold_score = targets['score']
        gold_score_bin = targets['score_bin'] if num_labels > 1 else None

        # predict
        with torch.no_grad():
            pred_score, attn = model.forward(**batch, return_attentions=True)

        if num_labels == 1:
            pred_proba = None
            y_hat.extend(pred_score.view(-1).tolist())
        else:
            pred_proba = torch.exp(pred_score)
            y_hat.extend(pred_proba.argmax(-1).view(-1).tolist())

        # overall
        if aggregation != 'none':
            bounds, bounds_mask = model.bottleneck.get_bounds_from_first_piece_mask(fp_mask)
            attn_mask = bounds_mask
            r = torch.arange(bounds.shape[0], device=bounds.device).unsqueeze(-1)
            fs_mask = fs_mask[r, bounds]
            fp_mask = fp_mask[r, bounds]
            assert torch.allclose(fp_mask, torch.ones_like(fp_mask))

        attn_avg = attn.squeeze(1)
        explanations = get_valid_explanations(attn_avg, attn_mask, fs_mask, fp_mask)
        explanations_all.extend(explanations)

    # save stuff
    save_dir = args.save + '_all/'
    e_mt_hat, e_src_hat, e_mt_fp_mask, e_src_fp_mask = zip(*explanations_all)
    save_list(os.path.join(save_dir, 'sentence_scores.txt'), y_hat)
    save_list(os.path.join(save_dir, 'mt_scores.txt'), e_mt_hat)
    save_list(os.path.join(save_dir, 'source_scores.txt'), e_src_hat)
    save_list(os.path.join(save_dir, 'mt_fp_mask.txt'), e_mt_fp_mask)
    save_list(os.path.join(save_dir, 'source_fp_mask.txt'), e_src_fp_mask)
