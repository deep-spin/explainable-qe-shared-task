import os
from functools import partial

import torch
from captum.attr import (LayerIntegratedGradients, LayerGradientXActivation, FeatureAblation, Lime, KernelShap,
                         LayerConductance)
from torch.utils.data import SequentialSampler, DataLoader
from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel

import numpy as np
import argparse
from zipfile import ZipFile

from transquest.algo.sentence_level.monotransquest.utils import InputExample
from utils import read_qe_files, evaluate_word_level, evaluate_sentence_level, aggregate_pieces

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", default='TransQuest/monotransquest-da-ro_en-wiki')
    parser.add_argument("-t", "--testset", default="data/ro-en/dev",  help="Testset Path.")
    parser.add_argument("-e", "--explainer", default="ig", help="Explainability method.")
    parser.add_argument("-r", "--reduction", default="sum", choices=['first', 'sum', 'mean', 'max'])
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-s", "--save", type=str, default="experiments/explanations/tmp/", help="save dir")
    parser.add_argument("--mc_dropout", default=False)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # 1) Load Checkpoint
    tqmodel = MonoTransQuestModel('xlmroberta', args.checkpoint, num_labels=1, use_cuda=torch.cuda.is_available())
    tqmodel.model.eval()
    tqmodel.model.zero_grad()
    tqmodel.model.to(device)

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

    def transquest_predict_logits_fn(*args, **kwargs):
        return tqmodel.model.forward(*args, **kwargs)[0]

    def get_first_sentence_mask(input_ids):
        fp = torch.zeros_like(input_ids)
        for i, idxs in enumerate(input_ids):
            for j, idx in enumerate(idxs):
                fp[i, j] = 1
                if idx == 2:
                    break
        return fp.long()

    def get_first_piece_mask(input_ids):
        fp = torch.zeros_like(input_ids)
        for i, idxs in enumerate(input_ids.tolist()):
            for j, tk in enumerate(tqmodel.tokenizer.convert_ids_to_tokens(idxs)):
                fp[i, j] = int(tk.startswith('▁'))
        return fp.long()

    eval_examples = [InputExample(i, d['src'], d['mt'], d['score']) for i, d in enumerate(data)]
    eval_dataset = tqmodel.load_and_cache_examples(eval_examples, evaluate=True, multi_label=False, no_cache=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)

    for i, batch in enumerate(eval_dataloader):
        print('Explaining {}/{}...'.format(i+1, len(eval_dataloader)), end='\r')
        inputs = tqmodel._get_inputs_dict(batch)
        gold_score = inputs['labels'].to(device)
        input_ids = inputs['input_ids'].to(device)
        attn_mask = inputs['attention_mask'].to(device)
        fs_mask = get_first_sentence_mask(input_ids).to(device)
        fp_mask = get_first_piece_mask(input_ids).to(device)

        with torch.no_grad():
            outputs = tqmodel.model(**inputs)
            eval_loss, pred_score = outputs[:2]

        y_hat.extend(pred_score.view(-1).detach().cpu().tolist())

        # create a baseline input: "<unk> <unk> ..."
        unk_id = tqmodel.tokenizer.unk_token_id
        baseline_input_ids = torch.zeros_like(inputs['input_ids']) + unk_id
        baseline_input_ids[input_ids == tqmodel.tokenizer.cls_token_id] = tqmodel.tokenizer.cls_token_id
        baseline_input_ids[input_ids == tqmodel.tokenizer.sep_token_id] = tqmodel.tokenizer.sep_token_id
        baseline_input_ids[input_ids == tqmodel.tokenizer.pad_token_id] = tqmodel.tokenizer.pad_token_id

        if args.explainer == 'ig':
            # w.r.t. the Embedding layer
            expl = LayerIntegratedGradients(transquest_predict_logits_fn, tqmodel.model.roberta.embeddings)
            # w.r.t. word embedding vectors (i.e. ignore positional embeddings)
            # expl = LayerIntegratedGradients(model.forward_base, model.model.embeddings.word_embeddings)
            attributions = expl.attribute(
                inputs=input_ids,
                baselines=baseline_input_ids,
                additional_forward_args=(inputs['attention_mask'], inputs['token_type_ids']),
                n_steps=10
            )
            # divide each vector by its L2 norm:
            attributions = attributions.sum(dim=-1).squeeze(0) / torch.norm(attributions, p=2, dim=-1)
            explanations = attributions.to(device)

        elif args.explainer == 'gxi':
            expl = LayerGradientXActivation(transquest_predict_logits_fn, tqmodel.model.roberta.embeddings)
            attributions = expl.attribute(
                inputs=input_ids,
                additional_forward_args=(inputs['attention_mask'], inputs['token_type_ids']),
            )
            # here, attributions = gradient .* input (element-wise)
            # so we .sum(-1) to get a dot product
            attributions = attributions.sum(-1)
            explanations = attributions.to(device)

        elif args.explainer == 'loo' or args.explainer == 'lpo':
            # leave-one-out or leave-pieces-out
            expl = FeatureAblation(transquest_predict_logits_fn)
            feature_mask = fp_mask.cumsum(-1).to(device) if args.explainer == 'lpo' else None
            attributions = expl.attribute(
                inputs=input_ids,
                baselines=unk_id,
                feature_mask=feature_mask,
                additional_forward_args=(inputs['attention_mask'], inputs['token_type_ids']),
            )
            explanations = attributions.to(device)

        elif args.explainer == 'lime' or args.explainer == 'limep':
            expl = Lime(transquest_predict_logits_fn)
            feature_mask = fp_mask.cumsum(-1).to(device) if args.explainer == 'limep' else None
            attributions = expl.attribute(
                inputs=input_ids,
                baselines=unk_id,
                feature_mask=feature_mask,
                n_samples=50,
                additional_forward_args=(inputs['attention_mask'], inputs['token_type_ids']),
            )
            explanations = attributions.to(device)

        elif args.explainer == 'kshap' or args.explainer == 'kshapp':
            expl = KernelShap(transquest_predict_logits_fn)
            feature_mask = fp_mask.cumsum(-1).to(device) if args.explainer == 'kshapp' else None
            attributions = expl.attribute(
                inputs=input_ids,
                baselines=unk_id,
                feature_mask=feature_mask,
                additional_forward_args=(inputs['attention_mask'], inputs['token_type_ids']),
            )
            explanations = attributions.to(device)

        elif args.explainer == 'erasure':
            # completely removed the ith token
            pass

        elif args.explainer == 'cond':
            expl = LayerConductance(transquest_predict_logits_fn, tqmodel.model.roberta.embeddings)
            attributions = expl.attribute(
                inputs=input_ids,
                baselines=baseline_input_ids,
                additional_forward_args=(inputs['attention_mask'], inputs['token_type_ids']),
            )
            attributions = attributions.sum(dim=-1).squeeze(0) / torch.norm(attributions, p=2, dim=-1)
            explanations = attributions.to(device)

        elif args.explainer == 'lrp':
            # todo: implement rules for transformers (see Elena Voita's paper)
            raise NotImplementedError

        elif args.explainer == 'attn_head':
            # todo: implement aggregation strategies for heads / layers
            raise NotImplementedError

        else:
            raise NotImplementedError

        # input = <s>  src  </s>    </s>  mt  </s>
        #         -------------
        #         first sentence
        for j in range(explanations.shape[0]):
            explanations_j = explanations[j][attn_mask[j].bool()]
            fs_mask_j = fs_mask[j][attn_mask[j].bool()].bool()
            fp_mask_j = fp_mask[j][attn_mask[j].bool()].bool()

            mt_explanations = explanations_j[~fs_mask_j].detach().squeeze()
            src_explanations = explanations_j[fs_mask_j].detach().squeeze()
            mt_fp_mask = fp_mask_j[~fs_mask_j].detach().squeeze()
            src_fp_mask = fp_mask_j[fs_mask_j].detach().squeeze()

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
