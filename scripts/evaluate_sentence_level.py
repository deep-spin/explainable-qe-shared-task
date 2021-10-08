import torch
import numpy as np
import argparse

from utils import read_qe_files, evaluate_sentence_level

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Model checkpoint to be tested.")
    parser.add_argument("--testset", default="data/2020-da.qe.test20.csv",  help="Testset Path.")
    parser.add_argument("--model_type", default='XLMRobertaQE')
    parser.add_argument("--mc_dropout", default=False, help="Model checkpoint to be tested.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load Checkpoint
    model_kwargs = {}
    if args.model_type == 'XLMRobertaQE':
        from model.xlm_roberta import load_checkpoint as load_checkpoint_xlm_roberta
        model = load_checkpoint_xlm_roberta(args.checkpoint)
        model_kwargs = dict(reverse_mt_and_src=model.reverse_mt_and_src, only_src=model.only_src)
        print(model_kwargs)
    elif args.model_type == 'XLMRationalizer':
        from model.rationalizer import load_checkpoint as load_checkpoint_rationalizer
        model = load_checkpoint_rationalizer(args.checkpoint)
    elif args.model_type == "XLMRobertaBottlenecked":
        from model.xlm_roberta_bottlenecked import load_checkpoint as load_checkpoint_bottlenecked
        model = load_checkpoint_bottlenecked(args.checkpoint)
    elif args.model_type == "MBart":
        from model.mbart50 import load_checkpoint as load_checkpoint_mbart50
        model = load_checkpoint_mbart50(args.checkpoint)
    elif args.model_type == "ByT5":
        from model.byt5 import load_checkpoint as load_checkpoint_byt5
        model = load_checkpoint_byt5(args.checkpoint)
    elif args.model_type == "RemBERT":
        from model.rembert import load_checkpoint as load_checkpoint_rembert
        model = load_checkpoint_rembert(args.checkpoint)
    else:
        raise NotImplementedError
    model.eval()
    model.zero_grad()
    model.to(device)

    # 2) Prepare TESTSET
    data = read_qe_files(args.testset, **model_kwargs)

    if args.mc_dropout:
        model.set_mc_dropout(int(args.mc_dropout))

    # 3) Run Predictions
    all_scores = []
    y = [d["score"] for d in data]
    y = np.array(y)

    _, pred = model.predict(data, show_progress=True, cuda=True, batch_size=1)

    if model.num_labels == 1:
        pred = torch.tensor(pred)
        y_hat = pred.view(-1).numpy()
        y_gold = (y > 50) * 1
        y_pred = (y_hat > 50) * 1
        print('Acc:', np.mean(y_pred == y_gold))
        pearson, spearman, mae, rmse = evaluate_sentence_level(y, y_hat)
    else:
        pred = torch.tensor(pred)
        pred_proba = torch.exp(pred)
        y_class = pred_proba.argmax(dim=-1).tolist()
        y_pred = np.array(y_class)
        y_gold = (y > 50) * 1  # cast to int
        y_proba = pred_proba[:, 1] * 100
        y_hat = y_proba.view(-1).numpy()
        print('Acc:', np.mean(y_pred == y_gold))
        pearson, spearman, mae, rmse = evaluate_sentence_level(y, y_hat)

    # print to google docs
    print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(pearson, spearman, mae/100, rmse/100))
