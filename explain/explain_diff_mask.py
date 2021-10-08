import os

import torch
from model.xlm_roberta_diff_mask import load_checkpoint

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
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("-s", "--save", type=str, default="experiments/explanations/tmp/", help="save dir")
    parser.add_argument("-e", "--eval", action='store_true')
    # args.save = 'experiments/explanations/roen_attn'

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load Checkpoint
    model = load_checkpoint(args.checkpoint)
    model.eval()
    model.zero_grad()
    model.to(device)

    # 2) Prepare TESTSET
    data = read_qe_files(args.testset, inference=True)

    # 3) Run Predictions
    y_hat = []
    batches = [data[i:i + args.batch_size] for i in range(0, len(data), args.batch_size)]

    num_layers = model.qe_model.model.config.num_hidden_layers
    explanations_layers = [[] for _ in range(num_layers + 2)]

    for i, batched_sample in enumerate(batches):
        print('Explaining {}/{}...'.format(i+1, len(batches)), end='\r')

        # prepare sample
        batch = model.prepare_sample(batched_sample, cuda=True, inference=True)
        input_ids = batch['input_ids']
        attn_mask = batch['attention_mask']
        mt_eos_ids = batch['mt_eos_ids']
        fs_mask = batch['first_sentence_mask']
        fp_mask = batch['first_piece_mask']

        with torch.no_grad():
            # attributions = model.forward_explainer(batch, attribution=True).exp()
            logits, logits_orig, attributions = model.forward_explainer(batch, attribution=True, return_logits=True)
            y_hat.extend(logits.view(-1).tolist())
            attributions = attributions.exp().squeeze(0).t()

        # attributions.shape is
        # (num_layers+2, seq_len)
        for layer_id in range(num_layers + 2):
            attn_avg = attributions[layer_id].unsqueeze(0)
            explanations = get_valid_explanations(attn_avg, attn_mask, fs_mask, fp_mask)
            explanations_layers[layer_id].extend(explanations)

    # save stuff
    for layer_id in range(num_layers + 2):
        save_dir = args.save + '_layer_{}/'.format(layer_id)
        e_mt_hat, e_src_hat, e_mt_fp_mask, e_src_fp_mask = zip(*explanations_layers[layer_id])
        save_list(os.path.join(save_dir, 'sentence_scores.txt'), y_hat)
        save_list(os.path.join(save_dir, 'mt_scores.txt'), e_mt_hat)
        save_list(os.path.join(save_dir, 'source_scores.txt'), e_src_hat)
        save_list(os.path.join(save_dir, 'mt_fp_mask.txt'), e_mt_fp_mask)
        save_list(os.path.join(save_dir, 'source_fp_mask.txt'), e_src_fp_mask)
