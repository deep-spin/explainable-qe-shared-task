import os

import torch
import numpy as np
import networkx as nx
from model.xlm_roberta import load_checkpoint

import argparse

from utils import read_qe_files


def get_adjmat(mat, input_tokens):
    n_layers, length, _ = mat.shape
    adj_mat = np.zeros(((n_layers+1)*length, (n_layers+1)*length))
    labels_to_index = {}
    for k in np.arange(length):
        labels_to_index[str(k)+"_"+input_tokens[k]] = k

    for i in np.arange(1,n_layers+1):
        for k_f in np.arange(length):
            index_from = (i)*length+k_f
            label = "L"+str(i)+"_"+str(k_f)
            labels_to_index[label] = index_from
            for k_t in np.arange(length):
                index_to = (i-1)*length+k_t
                adj_mat[index_from][index_to] = mat[i-1][k_f][k_t]

    return adj_mat, labels_to_index


def compute_flows(G, labels_to_index, input_nodes, length):
    number_of_nodes = len(labels_to_index)
    flow_values=np.zeros((number_of_nodes,number_of_nodes))
    for key in labels_to_index:
        if key not in input_nodes:
            current_layer = int(labels_to_index[key] / length)
            pre_layer = current_layer - 1
            u = labels_to_index[key]
            for inp_node_key in input_nodes:
                v = labels_to_index[inp_node_key]
                flow_value = nx.maximum_flow_value(G,u,v, flow_func=nx.algorithms.flow.edmonds_karp)
                flow_values[u][pre_layer*length+v ] = flow_value
            flow_values[u] /= flow_values[u].sum()

    return flow_values


def compute_node_flow(G, labels_to_index, input_nodes, output_nodes,length):
    number_of_nodes = len(labels_to_index)
    flow_values=np.zeros((number_of_nodes,number_of_nodes))
    for key in output_nodes:
        if key not in input_nodes:
            current_layer = int(labels_to_index[key] / length)
            pre_layer = current_layer - 1
            u = labels_to_index[key]
            for inp_node_key in input_nodes:
                v = labels_to_index[inp_node_key]
                flow_value = nx.maximum_flow_value(G,u,v, flow_func=nx.algorithms.flow.edmonds_karp)
                flow_values[u][pre_layer*length+v ] = flow_value
            flow_values[u] /= flow_values[u].sum()

    return flow_values


def compute_joint_attention(att_mat, add_residual=True):
    if add_residual:
        residual_att = np.eye(att_mat.shape[1])[None,...]
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[...,None]
    else:
        aug_att_mat =  att_mat

    joint_attentions = np.zeros(aug_att_mat.shape)

    layers = joint_attentions.shape[0]
    joint_attentions[0] = aug_att_mat[0]
    for i in np.arange(1,layers):
        joint_attentions[i] = aug_att_mat[i].dot(joint_attentions[i-1])

    return joint_attentions


def convert_adjmat_tomats(adjmat, n_layers, l):
    mats = np.zeros((n_layers,l,l))

    for i in np.arange(n_layers):
        mats[i] = adjmat[(i+1)*l:(i+2)*l,i*l:(i+1)*l]

    return mats


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
    parser.add_argument("--method", default="rollout", help="rollout or flow.")
    parser.add_argument("--norm-attention", action='store_true')
    parser.add_argument("--only-cross-attention", action='store_true')
    parser.add_argument("--mc_dropout", default=False)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    only_cross_attention = args.only_cross_attention
    if only_cross_attention:
        assert args.batch_size == 1

    # 1) Load Checkpoint
    model = load_checkpoint(args.checkpoint)
    model.eval()
    model.zero_grad()
    model.to(device)

    num_labels = model.num_labels
    target = None if num_labels == 1 else 0

    # 2) Prepare TESTSET
    data = read_qe_files(args.testset, inference=True)

    # 3) Run Predictions
    y_hat = []
    batches = [data[i:i + args.batch_size] for i in range(0, len(data), args.batch_size)]

    num_layers = model.model.config.num_hidden_layers
    num_heads = model.model.config.num_attention_heads
    explanations_layers = [[] for _ in range(num_layers)]

    for i, batched_sample in enumerate(batches):
        print('Explaining {}/{}...'.format(i+1, len(batches)), end='\r')

        # prepare sample
        batch = model.prepare_sample(batched_sample, cuda=True, inference=True)
        input_ids = batch['input_ids']
        attn_mask = batch['attention_mask']
        mt_eos_ids = batch['mt_eos_ids']
        fs_mask = batch['first_sentence_mask']
        fp_mask = batch['first_piece_mask']

        # predict
        with torch.no_grad():
            pred_score, attn, hidden_states = model.forward(**batch, return_attentions=True, return_hidden_states=True)
            attn = torch.stack(attn)
            hidden_states = torch.stack(hidden_states)

        if num_labels == 1:
            pred_proba = None
            y_hat.extend(pred_score.view(-1).tolist())
        else:
            pred_proba = torch.exp(pred_score)
            y_hat.extend(pred_proba.argmax(-1).view(-1).tolist())

        # (num_layers, bs, num_heads, seq_len, seq_len) -> (bs, nl, nh, seq_len, seq_len)
        attn = attn.transpose(0, 1)
        # (num_layers+2, bs, num_heads, seq_len, seq_len) -> (bs, nl+2, nh, seq_len, seq_len)
        # layer  0 = embeddings
        # layer -1 = scalar mix (even if not used)
        hidden_states = hidden_states.transpose(0, 1)
        bs, seq_len = attn_mask.shape
        L = fs_mask.squeeze().sum().item()  # intended to be used when batch_size == 1

        for layer_id in range(num_layers):
            if args.norm_attention:
                self_attn_module = model.model.encoder.layer[layer_id].attention.self
                values = self_attn_module.transpose_for_scores(self_attn_module.value(hidden_states[:, layer_id+1]))
                values_norm = torch.norm(values.detach(), p=2, dim=-1)
                attn[:, layer_id] = attn[:, layer_id] * values_norm.unsqueeze(2)

        res_att_mat = attn.mean(2).squeeze(0)
        res_att_mat = res_att_mat + torch.eye(res_att_mat.shape[-1], device=res_att_mat.device).unsqueeze(0)
        res_att_mat = res_att_mat / res_att_mat.sum(-1).unsqueeze(-1)
        joint_attentions = compute_joint_attention(res_att_mat.cpu().numpy(), add_residual=False)

        if args.method == 'flow':
            tokens = list(map(str, input_ids.squeeze().tolist()))
            res_adj_mat, res_labels_to_index = get_adjmat(mat=res_att_mat, input_tokens=tokens)
            joint_att_adjmat, joint_labels_to_index = get_adjmat(mat=joint_attentions, input_tokens=tokens)
            output_nodes = []
            input_nodes = []
            for key in res_labels_to_index:
                if 'L{}'.format(num_layers) in key:
                    output_nodes.append(key)
                if res_labels_to_index[key] < attn.shape[-1]:
                    input_nodes.append(key)
            res_G = nx.from_numpy_matrix(joint_att_adjmat, create_using=nx.DiGraph())
            for k in np.arange(joint_att_adjmat.shape[0]):
                for j in np.arange(joint_att_adjmat.shape[1]):
                    nx.set_edge_attributes(res_G, {(k, j): joint_att_adjmat[k, j]}, 'capacity')
            flow_values = compute_node_flow(res_G, res_labels_to_index, input_nodes, output_nodes, length=attn.shape[-1])
            joint_attentions = flow_values
            # import ipdb; ipdb.set_trace()

        for layer_id in range(num_layers):
            attn_avg = torch.from_numpy(joint_attentions)[layer_id].mean(-1)  # assume batch size = 1
            attn_avg = attn_avg.unsqueeze(0)  # add batch dimension
            explanations = get_valid_explanations(attn_avg, attn_mask, fs_mask, fp_mask)
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
