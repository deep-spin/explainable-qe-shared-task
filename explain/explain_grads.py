import os

import torch
from captum.attr import (LayerIntegratedGradients, LayerGradientXActivation, FeatureAblation, Lime, KernelShap,
                         LayerConductance)
from model.utils import masked_average
from model.xlm_roberta import load_checkpoint

import numpy as np
import argparse
from zipfile import ZipFile

from utils import read_qe_files, evaluate_word_level, evaluate_sentence_level, aggregate_pieces


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
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("-s", "--save", type=str, default="experiments/explanations/tmp/", help="save dir")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # 1) Load Checkpoint
    model = load_checkpoint(args.checkpoint, output_hidden_states=True, output_attentions=False)
    model.eval()
    model.zero_grad()
    model.to(device)

    num_labels = model.num_labels
    target = None if num_labels == 1 else 0

    # 2) Prepare TESTSET
    data = read_qe_files(args.testset)

    # 3) Run Predictions
    y_gold = [sample["score"] for sample in data]
    y_hat = []
    e_mt_gold = [sample["tgt_tags"] for sample in data]
    e_src_gold = [sample["src_tags"] for sample in data]
    batches = [data[i:i + args.batch_size] for i in range(0, len(data), args.batch_size)]

    num_layers = model.model.config.num_hidden_layers + 2  # embeddings and scalar mix
    num_heads = model.model.config.num_attention_heads
    explanations_layers = [[] for _ in range(num_layers)]

    def forward_helper_fn1(input_ids, attn_mask, mt_eos_ids, fs_mask, fp_mask):
        model_output = model.model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=False)
        hidden_states = model.scalar_mix(torch.stack(model_output.hidden_states), attn_mask)
        mt_summary = masked_average(hidden_states, fs_mask.bool())
        src_summary = masked_average(hidden_states, ~fs_mask.bool())
        combined_summary = model.alpha_merge * mt_summary + (1 - model.alpha_merge) * src_summary
        mt_score = model.estimator(combined_summary)
        return mt_score

    def forward_helper_fn2(input_emb, attn_mask, mt_eos_ids, fs_mask, fp_mask):
        model_output = model.model(inputs_embeds=input_emb, attention_mask=attn_mask, output_attentions=False)
        hidden_states = model.scalar_mix(model_output.hidden_states, attn_mask)
        mt_summary = masked_average(hidden_states, fs_mask.bool())
        src_summary = masked_average(hidden_states, ~fs_mask.bool())
        combined_summary = model.alpha_merge * mt_summary + (1 - model.alpha_merge) * src_summary
        mt_score = model.estimator(combined_summary)
        return mt_score

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
            pred_score = model.forward(**batch, return_attentions=False)

        if num_labels == 1:
            pred_proba = None
            y_hat.extend(pred_score.view(-1).tolist())
        else:
            pred_proba = torch.exp(pred_score)
            y_hat.extend(pred_proba.argmax(-1).view(-1).tolist())

        # create a baseline input: "<s> <unk> <unk> ... </s> </s> <unk> <unk> ... </s>"
        unk_id = model.tokenizer.stoi['<unk>']
        mask_id = model.tokenizer.stoi['<mask>']
        baseline_input_ids = torch.zeros_like(input_ids) + unk_id
        baseline_input_ids[input_ids == model.tokenizer.stoi['<s>']] = model.tokenizer.stoi['<s>']
        baseline_input_ids[input_ids == model.tokenizer.stoi['</s>']] = model.tokenizer.stoi['</s>']
        baseline_input_ids[input_ids == model.tokenizer.stoi['<pad>']] = model.tokenizer.stoi['<pad>']

        for layer_id in range(num_layers):

            if layer_id == 0:
                ref_module = model.model.embeddings
            elif layer_id == num_layers - 1:
                ref_module = model.scalar_mix
            else:
                ref_module = model.model.encoder.layer[layer_id-1].output
                # ref_module = model.model.encoder.layer[layer_id-1].attention.output
                # ref_module = model.model.encoder.layer[layer_id-1].attention.self
                # ref_module = model.model.encoder.layer[layer_id-1].attention.self.transform_fn  # w.r.t. softmax

            if args.explainer == 'ig':
                baseline_input_ids = None
                expl = LayerIntegratedGradients(forward_helper_fn1, ref_module)
                attributions = expl.attribute(
                    inputs=input_ids,
                    baselines=baseline_input_ids,
                    target=target,
                    additional_forward_args=(attn_mask, mt_eos_ids, fs_mask, fp_mask),
                    n_steps=25,  # todo: try 50
                )
                # divide each vector by its L2 norm:
                attributions = attributions.sum(dim=-1).squeeze(0) / torch.norm(attributions, p=2, dim=-1)
                explanations = attributions.to(device)

            elif args.explainer == 'gxi':
                expl = LayerGradientXActivation(forward_helper_fn1, ref_module)
                attributions = expl.attribute(
                    inputs=input_ids,
                    target=target,
                    additional_forward_args=(attn_mask, mt_eos_ids, fs_mask, fp_mask),
                )
                attributions = attributions.sum(-1)
                explanations = attributions.to(device)

            else:
                expl = LayerConductance(forward_helper_fn2, ref_module)
                attributions = expl.attribute(
                    inputs=model.model.embeddings(input_ids.long()),
                    baselines=model.model.embeddings(baseline_input_ids.long()),
                    target=target,
                    additional_forward_args=(attn_mask, mt_eos_ids, fs_mask, fp_mask),
                    n_steps=10
                )
                # divide each vector by its L2 norm:
                attributions = attributions.sum(dim=-1).squeeze(0) / torch.norm(attributions, p=2, dim=-1)
                explanations = attributions.to(device)
                del attributions
                del expl
                torch.cuda.empty_cache()

            explanations = get_valid_explanations(explanations, attn_mask, fs_mask, fp_mask)
            explanations_layers[layer_id].extend(explanations)

    # save stuff
    for layer_id in range(num_layers):
        save_dir = args.save + '_layer_{}/'.format(layer_id)
        e_mt_hat, e_src_hat, e_mt_fp_mask, e_src_fp_mask = zip(*explanations_layers[layer_id])
        save_list(os.path.join(save_dir, 'sentence_scores.txt'), y_hat)
        save_list(os.path.join(save_dir, 'mt_scores.txt'), e_mt_hat)
        save_list(os.path.join(save_dir, 'source_scores.txt'), e_src_hat)
        save_list(os.path.join(save_dir, 'mt_fp_mask.txt'), e_mt_fp_mask)
        save_list(os.path.join(save_dir, 'source_fp_mask.txt'), e_src_fp_mask)
