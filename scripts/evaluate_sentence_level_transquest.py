# pip3 install transquest
# python3 evaluate_transquest.py --testset data/ro-en/dev --checkpoint TransQuest/monotransquest-da-ro_en-wiki
# python3 evaluate_transquest.py --testset data/et-en/dev --checkpoint TransQuest/monotransquest-da-et_en-wiki

import torch
from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel


from scipy.stats import pearsonr, spearmanr
import numpy as np
import argparse

from utils import read_qe_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Model checkpoint to be tested.")
    parser.add_argument("--testset", default="data/2020-da.qe.test20.csv",  help="Testset Path.")
    args = parser.parse_args()

    # Load model
    model = MonoTransQuestModel('xlmroberta', args.checkpoint, num_labels=1, use_cuda=torch.cuda.is_available())
    data = read_qe_files(args.testset)

    # Evaluate predictions on the dataset
    src = [d['src'] for d in data]
    mt = [d['mt'] for d in data]
    y = [d["score"] for d in data]
    y_hat, raw_outputs = model.predict(list(map(list, zip(src, mt))))

    y = np.array(y) / 100.0
    y_hat = np.array(y_hat)  # no need to divide by 100
    pearson = pearsonr(y, y_hat)[0]
    spearman = spearmanr(y, y_hat)[0]
    diff = np.array(y) - np.array(y_hat)
    mae = np.abs(diff).mean()
    rmse = (diff ** 2).mean() ** 0.5
    print(f"pearson: {pearson}")
    print(f"spearman: {spearman}")
    print(f"mae: {mae}")
    print(f"rmse: {rmse}")
