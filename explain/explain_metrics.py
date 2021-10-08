import os

import torch
from captum.attr import (LayerIntegratedGradients, LayerGradientXActivation, FeatureAblation, Lime, KernelShap,
                         LayerConductance)
from model.sparse_xlmr import SparseXLMRobertaModel
from model.tokenizer import Tokenizer
from model.xlm_roberta import load_checkpoint

import numpy as np
import argparse
from zipfile import ZipFile

from torchnlp.utils import collate_tensors
from utils import read_qe_files, evaluate_word_level, evaluate_sentence_level, aggregate_pieces


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

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # 1) Load Checkpoint
    print('Loading pretrained model...')
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

    def fake_forward(*args, **kwrags):
        out = model(*args, **kwrags)
        return torch.sigmoid(out.pooler_output.mean(-1)) * 100  # fake prediction

    num_labels = 1
    target = None

    # 2) Prepare TESTSET
    data = read_qe_files(args.testset)

    # 3) Run Predictions
    y_gold = [sample["score"] for sample in data]
    y_hat = []
    e_mt_gold = [sample["tgt_tags"] for sample in data]
    e_mt_fp_mask = []
    e_mt_hat = []
    e_src_gold = [sample["src_tags"] for sample in data]
    e_src_fp_mask = []
    e_src_hat = []
    batches = [data[i:i + args.batch_size] for i in range(0, len(data), args.batch_size)]

    for i, batched_sample in enumerate(batches):
        print('Explaining {}/{}...'.format(i+1, len(batches)), end='\r')

        # prepare sample
        collated_sample = collate_tensors(batched_sample)
        batch = tokenizer.batch_encode(collated_sample["src"], collated_sample["mt"])
        for key in batch.keys():
            batch[key] = batch[key].to(device)

        input_ids = batch['input_ids']
        attn_mask = batch['attention_mask']
        mt_eos_ids = batch['mt_eos_ids']
        fs_mask = batch['first_sentence_mask']
        fp_mask = batch['first_piece_mask']
        gold_score = torch.tensor([b['score'] for b in batched_sample]).to(device)
        gold_score_bin = torch.tensor([int(b['score'] > 50) for b in batched_sample]).to(device)

        # predict
        with torch.no_grad():
            out = model(**batch)
            pred_score = torch.sigmoid(out.pooler_output.mean(-1)) * 100  # fake prediction
            attn = torch.stack(out.attentions)

        if num_labels == 1:
            pred_proba = None
            y_hat.extend(pred_score.view(-1).detach().cpu().tolist())
        else:
            pred_proba = torch.exp(pred_score)
            y_hat.extend(pred_proba.argmax(-1).view(-1).tolist())

        # create a baseline input: "<s> <unk> <unk> ... </s> </s> <unk> <unk> ... </s>"
        unk_id = tokenizer.stoi['<unk>']
        baseline_input_ids = torch.zeros_like(input_ids) + unk_id
        baseline_input_ids[input_ids == tokenizer.stoi['<s>']] = tokenizer.stoi['<s>']
        baseline_input_ids[input_ids == tokenizer.stoi['</s>']] = tokenizer.stoi['</s>']
        baseline_input_ids[input_ids == tokenizer.stoi['<pad>']] = tokenizer.stoi['<pad>']

        if args.explainer == 'ig':
            # w.r.t. the Embedding layer
            expl = LayerIntegratedGradients(fake_forward, model.embeddings)
            # w.r.t. word embedding vectors (i.e. ignore positional embeddings)
            # expl = LayerIntegratedGradients(model.forward_base, model.embeddings.word_embeddings)
            attributions = expl.attribute(
                inputs=input_ids,
                # baselines=baseline_input_ids,
                target=target,
                additional_forward_args=(attn_mask,),
                n_steps=10,
            )
            # divide each vector by its L2 norm:
            attributions = attributions.sum(dim=-1).squeeze(0) / torch.norm(attributions, p=2, dim=-1)
            explanations = attributions.to(device)

        elif args.explainer == 'gxi':
            expl = LayerGradientXActivation(fake_forward, model.embeddings)
            attributions = expl.attribute(
                inputs=input_ids,
                target=target,
                additional_forward_args=(attn_mask,),
            )
            # here, attributions = gradient .* input (element-wise)
            # so we .sum(-1) to get a dot product
            attributions = attributions.sum(-1)
            explanations = attributions.to(device)

        elif args.explainer == 'loo' or args.explainer == 'lpo':
            # leave-one-out or leave-pieces-out
            expl = FeatureAblation(model.forward)
            feature_mask = fp_mask.cumsum(-1).to(device) if args.explainer == 'lpo' else None
            attributions = expl.attribute(
                inputs=input_ids,
                baselines=unk_id,
                target=target,
                feature_mask=feature_mask,
                additional_forward_args=(attn_mask,),
            )
            explanations = attributions.to(device)

        elif args.explainer == 'lime' or args.explainer == 'limep':
            expl = Lime(model.forward)
            feature_mask = fp_mask.cumsum(-1).to(device) if args.explainer == 'limep' else None
            attributions = expl.attribute(
                inputs=input_ids,
                baselines=unk_id,
                target=target,
                feature_mask=feature_mask,
                n_samples=50,
                additional_forward_args=(attn_mask,),
            )
            explanations = attributions.to(device)

        elif args.explainer == 'kshap' or args.explainer == 'kshapp':
            expl = KernelShap(model.forward)
            feature_mask = fp_mask.cumsum(-1).to(device) if args.explainer == 'kshapp' else None
            attributions = expl.attribute(
                inputs=input_ids,
                baselines=unk_id,
                target=target,
                feature_mask=feature_mask,
                additional_forward_args=(attn_mask,),
            )
            explanations = attributions.to(device)

        elif args.explainer == 'erasure':
            # completely removed the ith token
            if num_labels == 1:
                base_loss = torch.nn.functional.mse_loss(pred_score.view(-1), gold_score, reduction='none')
            else:
                nll_loss_value = torch.nn.functional.nll_loss(pred_score, gold_score_bin, reduction='none')
                mse_loss_value = torch.nn.functional.mse_loss(pred_proba[:, 1], gold_score / 100, reduction='none')
                base_loss = nll_loss_value + model.mse_lbda * mse_loss_value
            bs, seq_len = input_ids.shape
            erase_mask = torch.zeros(seq_len, device=device, dtype=torch.bool)
            erase_mask[0] = True
            explanations = torch.zeros(bs, seq_len, device=device)
            for i in range(seq_len):
                input_ids_x = input_ids[:, ~erase_mask]
                attn_mask_x = attn_mask[:, ~erase_mask]
                mt_eos_ids_x = mt_eos_ids
                fs_mask_x = fs_mask[:, ~erase_mask]
                fp_mask_x = fp_mask[:, ~erase_mask]
                pred_score_x = model.forward(input_ids_x, attn_mask_x)
                if num_labels == 1:
                    new_loss = model.mse_loss_fn(pred_score_x.view(-1), gold_score, reduction='none')
                else:
                    pred_proba_x = torch.exp(pred_score_x)
                    nll_loss_value = torch.nn.functional.nll_loss(pred_score_x, gold_score_bin, reduction='none')
                    mse_loss_value = torch.nn.functional.mse_loss(pred_proba_x[:, 1], gold_score / 100, reduction='none')
                    new_loss = nll_loss_value + model.mse_lbda * mse_loss_value
                explanations[:, i] = base_loss - new_loss
                erase_mask = erase_mask.roll(1, -1)

        elif args.explainer == 'cond':
            expl = LayerConductance(fake_forward, model.embeddings)
            attributions = expl.attribute(
                inputs=input_ids,
                target=target,
                # baselines=baseline_input_ids,
                additional_forward_args=(attn_mask,),
            )
            attributions = attributions.sum(dim=-1).squeeze(0) / torch.norm(attributions, p=2, dim=-1)
            explanations = attributions.to(device)

        else:
            raise NotImplementedError

        # input = <s>  mt  </s>    </s>  src  </s>
        #         -------------
        #         first sentence
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

    # save predictions and explanations
    def save_list(fpath, list_of_vals):
        print('Saving to {}'.format(fpath))
        with open(fpath, 'w') as f:
            for i, v in enumerate(list_of_vals):
                if isinstance(v, (int, float)):
                    s = '{}'.format(v)
                else:
                    s = ' '.join(['{}'.format(v_) for v_ in v])
                f.write(s + '\n')

    save_list(os.path.join(args.save, 'sentence_scores.txt'), y_hat)
    save_list(os.path.join(args.save, 'mt_scores.txt'), e_mt_hat)
    save_list(os.path.join(args.save, 'source_scores.txt'), e_src_hat)
    save_list(os.path.join(args.save, 'mt_fp_mask.txt'), e_mt_fp_mask)
    save_list(os.path.join(args.save, 'source_fp_mask.txt'), e_src_fp_mask)
