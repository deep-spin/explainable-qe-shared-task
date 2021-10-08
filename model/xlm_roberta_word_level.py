# -*- coding: utf-8 -*-
import os
from argparse import Namespace
from typing import Dict, List, Optional, Tuple, Union

import click
import torch
import yaml
from model.bottleneck_layer import BottleneckSummary
from model.scalar_mix import ScalarMixWithDropout
from model.utils import move_to_cuda, move_to_cpu, pad_sequence
from model.xlm_roberta import XLMRobertaQE

from torch import nn
from torchnlp.utils import collate_tensors
from transformers import AdamW

from model.metrics import Pearson, Kendall, Accuracy, MCC, AUC, AveragePrecision
from model.tokenizer import Tokenizer
from model.utils import move_to_cpu, move_to_cuda, masked_average
from model.sparse_xlmr import SparseXLMRobertaModel

from tqdm import tqdm
from utils import Config


def load_checkpoint(checkpoint: str, **kwargs):
    if not os.path.exists(checkpoint):
        raise Exception(f"{checkpoint} file not found!")

    hparam_yaml_file = "/".join(checkpoint.split("/")[:-1] + ["hparams.yaml"])

    if os.path.exists(hparam_yaml_file):
        with open(hparam_yaml_file) as yaml_file:
            hparams = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
        model = XLMRobertaWithWordLevel.load_from_checkpoint(checkpoint, hparams={**hparams, **kwargs})
    else:
        raise Exception("hparams file not found.")

    model.eval()
    model.freeze()
    return model


class XLMRobertaWithWordLevel(XLMRobertaQE):

    class ModelConfig(Config):
        pretrained_model: str = "xlm-roberta-large"

        monitor: str = "pearson"
        metric_mode: str = "max"
        loss: str = "mse"

        # Optimizer
        learning_rate: float = 3.0e-5
        encoder_learning_rate: float = 1.0e-5
        keep_embeddings_frozen: bool = False
        nr_frozen_epochs: Union[float, int] = 0.3
        dropout: float = 0.1
        hidden_size: int = 2048
        use_adapters: bool = False

        # Data configs
        train_path: str = None
        val_path: str = None
        test_path: str = None
        load_weights_from_checkpoint: Union[str, bool] = False

        # Training details
        batch_size: int = 4
        num_labels: int = 1
        attention_alpha: float = 1.0
        mse_lbda: float = 1.0
        output_norm: bool = False
        norm_strategy: str = 'weighted_norm'
        effective: bool = False
        alpha_merge: float = 1.0
        return_hidden_states: bool = False

        # bottleneck
        aggregation: str = 'first'
        selected_layers: str = 'mix'
        word_level_lbda: float = 1.0

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)
        self.aggregation = self.hparams.get('aggregation', 'first')
        self.selected_layers = self.hparams.get('selected_layers', 'mix')
        self.word_level_lbda = self.hparams.get('word_level_lbda', 1.0)
        self.wl_estimator = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.hparams.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_size, 2),
            nn.LogSoftmax(dim=-1)
        )
        self.wl_loss = nn.NLLLoss(ignore_index=-100, reduction='none')
        self.train_mcc = MCC()
        self.train_auc = AUC()
        self.train_ap = AveragePrecision()
        self.dev_mcc = MCC()
        self.dev_auc = AUC()
        self.dev_ap = AveragePrecision()
        self._init_weights(self.estimator, self.wl_estimator)

    def _init_weights(self, *modules):
        for m in modules:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def configure_optimizers(self):
        self.epoch_total_steps = len(self.train_dataset) // (self.hparams.batch_size * max(1, self.trainer.num_gpus))
        parameters = [
            {"params": self.estimator.parameters(), "lr": self.hparams.learning_rate},
            {"params": self.wl_estimator.parameters(), "lr": self.hparams.learning_rate},
            {"params": self.model.parameters(), "lr": self.hparams.encoder_learning_rate},
        ]
        if self.selected_layers == 'mix':
            parameters.append({"params": self.scalar_mix.parameters(), "lr": self.hparams.learning_rate})

        optimizer = AdamW(parameters, lr=self.hparams.learning_rate, correct_bias=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=self.hparams.metric_mode),
            "monitor": self.hparams.monitor
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mt_eos_ids: torch.Tensor,
        first_sentence_mask: torch.Tensor,
        first_piece_mask: torch.Tensor,
        bounds: torch.LongTensor,
        bounds_mask: torch.Tensor,
        return_attentions: bool = False,
    ):
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = model_output.hidden_states
        attentions = model_output.attentions

        if self.selected_layers == 'mix':
            hidden_states = self.scalar_mix(hidden_states, attention_mask)
        else:
            selected_layers = list(map(int, self.selected_layers.split(',')))
            hidden_states = torch.stack(hidden_states)[selected_layers].mean(dim=0)

        mt_summary = masked_average(hidden_states, first_sentence_mask.bool())
        src_summary = masked_average(hidden_states, (~first_sentence_mask.bool()) & attention_mask.bool())
        combined_summary = self.alpha_merge * mt_summary + (1 - self.alpha_merge) * src_summary
        mt_score = self.estimator(combined_summary)

        wl_hidden_states = BottleneckSummary.aggregate_word_pieces(hidden_states, bounds, self.aggregation)
        wl_scores = self.wl_estimator(wl_hidden_states)

        if return_attentions:
            return mt_score, wl_scores, attentions
        return mt_score, wl_scores

    def training_step(self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs) -> Dict[str, torch.Tensor]:
        batch_input, batch_target = batch
        predicted_scores, wl_scores = self.forward(**batch_input)

        # sentence level
        sl_loss_value = self.loss_fn(predicted_scores.view(-1), batch_target["score"].view(-1))
        if wl_scores.shape[1] != batch_target['word_tags'].shape[1]:
            print('bug...', wl_scores.shape, batch_target['word_tags'].shape)
            wl_loss_value = 0
        else:
            wl_loss_value = self.wl_loss(wl_scores.view(-1, 2), batch_target['word_tags'].view(-1))

        # word level
        tgt_weight = 1.0
        src_weight = 1.0
        fs_mask_word_tags = batch_target['fs_mask_word_tags'].view(-1)
        wl_mask = (fs_mask_word_tags == 1).float() * tgt_weight + (fs_mask_word_tags == 2).float() * src_weight
        wl_loss_value = (wl_loss_value * wl_mask).sum(-1).mean()

        # combine
        if self.word_level_lbda < 0:
            loss_value = wl_loss_value
        else:
            loss_value = sl_loss_value + self.word_level_lbda * wl_loss_value

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_value = loss_value.unsqueeze(0)

        if not self.hparams.use_adapters:
            if (
                    0.0 < self.hparams.nr_frozen_epochs < 1.0
                    and batch_nb > self.epoch_total_steps * self.hparams.nr_frozen_epochs
            ):
                self.unfreeze_encoder()

        self.log("train_loss", loss_value, on_step=True, on_epoch=True)
        return loss_value

    def validation_step(
            self,
            batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
            batch_nb: int,
            dataloader_idx: int,
            *args, **kwargs
    ) -> None:
        batch_input, batch_target = batch
        predicted_scores, wl_scores = self.forward(**batch_input)
        wl_mask = batch_target['fs_mask_word_tags'] != -100
        wl_labels_flat = wl_scores.argmax(dim=-1)[wl_mask]
        wl_probas_flat = wl_scores[:, :, 1][wl_mask].exp()
        word_tags_flat = batch_target['word_tags'][wl_mask]

        if dataloader_idx == 0:
            self.log("train_kendall", self.train_kendall(predicted_scores.view(-1), batch_target["score"]))
            self.log("train_pearson", self.train_pearson(predicted_scores.view(-1), batch_target["score"]))
            self.log("train_mcc", self.train_mcc(wl_labels_flat, word_tags_flat))
            self.log("train_auc", self.train_auc(wl_probas_flat, word_tags_flat.float()))
            self.log("train_ap", self.train_ap(wl_probas_flat, word_tags_flat.float()))

        if dataloader_idx == 1:
            self.log("kendall", self.dev_kendall(predicted_scores.view(-1), batch_target["score"]))
            self.log("pearson", self.dev_pearson(predicted_scores.view(-1), batch_target["score"]))
            self.log("mcc", self.dev_mcc(wl_labels_flat, word_tags_flat))
            self.log("auc", self.dev_auc(wl_probas_flat, word_tags_flat.float()))
            self.log("ap", self.dev_ap(wl_probas_flat, word_tags_flat.float()))

    def validation_epoch_end(self, *args, **kwargs) -> None:
        self.log("kendall", self.dev_kendall.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.dev_kendall.reset()
        self.log("pearson", self.dev_pearson.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.dev_pearson.reset()
        self.log("mcc", self.dev_mcc.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.dev_mcc.reset()
        self.log("auc", self.dev_auc.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.dev_auc.reset()
        self.log("ap", self.dev_ap.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.dev_ap.reset()

        self.log("train_pearson", self.train_pearson.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.train_pearson.reset()
        self.log("train_kendall", self.train_kendall.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.train_kendall.reset()
        self.log("train_mcc", self.train_mcc.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.train_mcc.reset()
        self.log("train_auc", self.train_auc.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.train_auc.reset()
        self.log("train_ap", self.train_ap.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.train_ap.reset()

    def prepare_sample(
            self, sample: List[Dict[str, Union[str, float]]], inference: bool = False, cuda: bool = False
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """
        Function that prepares a sample to input the model.

        :param sample: list of dictionaries.
        :param inference: If set to true prepares only the model inputs.

        :returns: Tuple with 2 dictionaries (model inputs and targets).
            If `inference=True` returns only the model inputs.
        """
        collated_sample = collate_tensors(sample)
        mt_inputs = self.tokenizer.batch_encode(collated_sample["src"], collated_sample["mt"])

        bounds, bounds_mask = BottleneckSummary.get_bounds_from_first_piece_mask(mt_inputs['first_piece_mask'])
        mt_inputs['bounds'] = bounds
        mt_inputs['bounds_mask'] = bounds_mask

        # might be useful later:
        # pe_inputs = self.tokenizer.batch_encode(collated_sample["src"], collated_sample["pe"])
        if cuda:
            mt_inputs = move_to_cuda(mt_inputs) if cuda and torch.cuda.is_available() else mt_inputs

        if inference:
            return mt_inputs

        tgt_tags = [torch.tensor(s['tgt_tags']) for s in sample]
        src_tags = [torch.tensor(s['src_tags']) for s in sample]
        word_tags = [torch.tensor(s['tgt_tags'] + s['src_tags']) for s in sample]
        fs_mask_word_tags = [torch.tensor([1]*len(s['tgt_tags']) + [2]*len(s['src_tags'])) for s in sample]

        targets = {
            "score": torch.tensor(collated_sample["score"], dtype=torch.float),
            "hter": torch.tensor(collated_sample["hter"], dtype=torch.float),
            # NLLLoss will ignore -100 labels directly (hardcoded)
            "tgt_tags": pad_sequence(tgt_tags, padding_value=-100, end=True),
            "src_tags": pad_sequence(src_tags, padding_value=-100, end=False),
            "word_tags": pad_sequence(word_tags, padding_value=-100, end=True),
            "fs_mask_word_tags": pad_sequence(fs_mask_word_tags, padding_value=-100, end=True)
        }

        if cuda:
            targets = move_to_cuda(targets) if cuda and torch.cuda.is_available() else targets

        return mt_inputs, targets
