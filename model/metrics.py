# -*- coding: utf-8 -*-
import warnings

import torch
from pytorch_lightning.metrics import Metric
from scipy.stats import kendalltau, pearsonr
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score


class Kendall(Metric):
    def __init__(self, dist_sync_on_step=False, padding=None, ignore=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("predictions", default=[], dist_reduce_fx="sum")
        self.add_state("scores", default=[], dist_reduce_fx="sum")
        
    def update(self, predictions: torch.Tensor, scores: torch.Tensor):
        assert predictions.shape == scores.shape
        self.predictions += predictions.cpu().tolist() if predictions.is_cuda else predictions.tolist()
        self.scores += scores.cpu().tolist() if scores.is_cuda else predictions.tolist()

    def compute(self):
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            if len(self.predictions) == 0:
                self.predictions.append(0)
                self.scores.append(0)
            if len(self.predictions) == 1:
                self.predictions.append(0)
                self.scores.append(0)
            return torch.tensor(kendalltau(self.predictions, self.scores)[0], dtype=torch.float32)


class Pearson(Metric):
    def __init__(self, dist_sync_on_step=False, padding=None, ignore=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("predictions", default=[], dist_reduce_fx="sum")
        self.add_state("scores", default=[], dist_reduce_fx="sum")
        
    def update(self, predictions: torch.Tensor, scores: torch.Tensor):
        assert predictions.shape == scores.shape
        self.predictions += predictions.cpu().tolist() if predictions.is_cuda else predictions.tolist()
        self.scores += scores.cpu().tolist() if scores.is_cuda else predictions.tolist()

    def compute(self):
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            if len(self.predictions) == 0:
                self.predictions.append(0)
                self.scores.append(0)
            if len(self.predictions) == 1:
                self.predictions.append(0)
                self.scores.append(0)
            return torch.tensor(pearsonr(self.predictions, self.scores)[0], dtype=torch.float32)


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False, padding=None, ignore=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("predictions", default=[], dist_reduce_fx="sum")
        self.add_state("scores", default=[], dist_reduce_fx="sum")

    def update(self, predictions: torch.Tensor, scores: torch.Tensor):
        self.predictions += predictions.cpu().tolist() if predictions.is_cuda else predictions.tolist()
        self.scores += scores.cpu().tolist() if scores.is_cuda else predictions.tolist()

    def compute(self):
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            return torch.mean((torch.tensor(self.predictions) == torch.tensor(self.scores)).float())


class MCC(Metric):
    def __init__(self, dist_sync_on_step=False, padding=None, ignore=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("predictions", default=[], dist_reduce_fx="sum")
        self.add_state("scores", default=[], dist_reduce_fx="sum")

    def update(self, predictions: torch.Tensor, scores: torch.Tensor):
        self.predictions += predictions.flatten().cpu().tolist() if predictions.is_cuda else predictions.tolist()
        self.scores += scores.flatten().cpu().tolist() if scores.is_cuda else predictions.tolist()

    def compute(self):
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            return torch.tensor(matthews_corrcoef(self.scores, self.predictions))


class AUC(Metric):
    def __init__(self, dist_sync_on_step=False, padding=None, ignore=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("predictions", default=[], dist_reduce_fx="sum")
        self.add_state("scores", default=[], dist_reduce_fx="sum")

    def update(self, predictions: torch.Tensor, scores: torch.Tensor):
        self.predictions += predictions.cpu().tolist() if predictions.is_cuda else predictions.tolist()
        self.scores += scores.cpu().tolist() if scores.is_cuda else predictions.tolist()

    def compute(self):
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            try:
                return roc_auc_score(self.scores, self.predictions)
            except ValueError:
                return 0.0


class AveragePrecision(Metric):
    def __init__(self, dist_sync_on_step=False, padding=None, ignore=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("predictions", default=[], dist_reduce_fx="sum")
        self.add_state("scores", default=[], dist_reduce_fx="sum")

    def update(self, predictions: torch.Tensor, scores: torch.Tensor):
        self.predictions += predictions.cpu().tolist() if predictions.is_cuda else predictions.tolist()
        self.scores += scores.cpu().tolist() if scores.is_cuda else predictions.tolist()

    def compute(self):
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            try:
                return average_precision_score(self.scores, self.predictions)
            except ValueError:
                return 0.0
