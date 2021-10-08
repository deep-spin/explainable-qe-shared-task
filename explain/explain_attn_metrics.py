import os

import torch
from torchnlp.utils import collate_tensors
from model.tokenizer import Tokenizer
from model.sparse_xlmr import SparseXLMRobertaModel

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
    pretrained_model_fname = 'Unbabel/xlm-roberta-wmt-metrics-da'
    model = SparseXLMRobertaModel.from_pretrained(
        pretrained_model_fname,
        output_hidden_states=True,
        output_attentions=True,
        alpha=1.0,
        output_norm=args.norm_attention,
        norm_strategy=args.norm_strategy
    )
    model.eval()
    model.to(device)
    tokenizer = Tokenizer(pretrained_model_fname)

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
        collated_sample = collate_tensors(batched_sample)
        batch = tokenizer.batch_encode(collated_sample["src"], collated_sample["mt"])
        for key in batch.keys():
            batch[key] = batch[key].to(device)

        input_ids = batch['input_ids']
        attn_mask = batch['attention_mask']
        fs_mask = batch['first_sentence_mask']
        fp_mask = batch['first_piece_mask']

        # predict
        with torch.no_grad():
            out = model(**batch)
            pred_score = torch.sigmoid(out.pooler_output.mean(-1)) * 100  # fake prediction
            attn = torch.stack(out.attentions)
            y_hat.extend(pred_score.view(-1).detach().cpu().tolist())

        # (num_layers, bs, num_heads, seq_len, seq_len) -> (bs, nl, nh, seq_len, seq_len)
        attn = attn.transpose(0, 1)
        bs, seq_len = attn_mask.shape

        for layer_id in range(num_layers):
            if args.norm_strategy != 'summed_weighted_norm':
                for head_id in range(num_heads):
                    # head
                    attn_sum = (attn[:, layer_id, head_id] * attn_mask.unsqueeze(-1).float()).sum(1)
                    attn_avg = attn_sum / attn_mask.sum(-1).unsqueeze(-1).float()
                    explanations = get_valid_explanations(attn_avg, attn_mask, fs_mask, fp_mask)
                    explanations_heads[layer_id][head_id].extend(explanations)

            # layer
            attn_heads = attn[:, layer_id].mean(1)
            attn_sum = (attn_heads * attn_mask.unsqueeze(-1).float()).sum(1)
            attn_avg = attn_sum / attn_mask.sum(-1).unsqueeze(-1).float()
            explanations = get_valid_explanations(attn_avg, attn_mask, fs_mask, fp_mask)
            explanations_layers[layer_id].extend(explanations)

        # overall
        attn_layers = attn.mean(1)
        attn_heads = attn_layers.mean(1)
        attn_sum = (attn_heads * attn_mask.unsqueeze(-1).float()).sum(1)
        attn_avg = attn_sum / attn_mask.sum(-1).unsqueeze(-1).float()
        explanations = get_valid_explanations(attn_avg, attn_mask, fs_mask, fp_mask)
        explanations_all.extend(explanations)

    # save stuff
    for layer_id in range(num_layers):
        if args.norm_strategy != 'summed_weighted_norm':
            for head_id in range(num_heads):
                save_dir = args.save + '_head_{}_{}/'.format(layer_id, head_id)
                e_mt_hat, e_src_hat, e_mt_fp_mask, e_src_fp_mask = zip(*explanations_heads[layer_id][head_id])
                save_list(os.path.join(save_dir, 'sentence_scores.txt'), y_hat)
                save_list(os.path.join(save_dir, 'mt_scores.txt'), e_mt_hat)
                save_list(os.path.join(save_dir, 'source_scores.txt'), e_src_hat)
                save_list(os.path.join(save_dir, 'mt_fp_mask.txt'), e_mt_fp_mask)
                save_list(os.path.join(save_dir, 'source_fp_mask.txt'), e_src_fp_mask)

        save_dir = args.save + '_layer_{}/'.format(layer_id)
        e_mt_hat, e_src_hat, e_mt_fp_mask, e_src_fp_mask = zip(*explanations_layers[layer_id])
        save_list(os.path.join(save_dir, 'sentence_scores.txt'), y_hat)
        save_list(os.path.join(save_dir, 'mt_scores.txt'), e_mt_hat)
        save_list(os.path.join(save_dir, 'source_scores.txt'), e_src_hat)
        save_list(os.path.join(save_dir, 'mt_fp_mask.txt'), e_mt_fp_mask)
        save_list(os.path.join(save_dir, 'source_fp_mask.txt'), e_src_fp_mask)

    save_dir = args.save + '_all/'
    e_mt_hat, e_src_hat, e_mt_fp_mask, e_src_fp_mask = zip(*explanations_all)
    save_list(os.path.join(save_dir, 'sentence_scores.txt'), y_hat)
    save_list(os.path.join(save_dir, 'mt_scores.txt'), e_mt_hat)
    save_list(os.path.join(save_dir, 'source_scores.txt'), e_src_hat)
    save_list(os.path.join(save_dir, 'mt_fp_mask.txt'), e_mt_fp_mask)
    save_list(os.path.join(save_dir, 'source_fp_mask.txt'), e_src_fp_mask)
