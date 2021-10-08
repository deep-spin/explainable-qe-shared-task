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
        model = XLMRobertaBottlenecked.load_from_checkpoint(checkpoint, hparams={**hparams, **kwargs})
    else:
        raise Exception("hparams file not found.")

    model.eval()
    model.freeze()
    return model


class XLMRobertaBottlenecked(XLMRobertaQE):

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
        aggregation: str = 'none'
        selected_layers: str = 'mix'
        kv_rep: str = 'embeddings'
        bottleneck_alpha: float = 1.0
        classwise: bool = False
        bottleneck_reg: str = 'none'
        bottleneck_lbda: float = 1.0
        squared_attn: bool = False

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)
        self.aggregation = self.hparams.get('aggregation', 'none')
        self.selected_layers = self.hparams.get('selected_layers', 'mix')
        self.kv_rep = self.hparams.get('kv_rep', 'embeddings')
        self.bottleneck_alpha = self.hparams.get('bottleneck_alpha', 1.0)
        self.classwise = self.hparams.get('classwise', False)
        self.bottleneck_reg = self.hparams.get('bottleneck_reg', 'none')
        self.bottleneck_lbda = self.hparams.get('bottleneck_lbda', 1.0)
        self.squared_attn = self.hparams.get('squared_attn', False)
        self.bottleneck = BottleneckSummary(
            self.model.config.hidden_size,
            self.aggregation,
            self.kv_rep,
            self.bottleneck_alpha,
            self.classwise,
            alpha_merge=self.alpha_merge,
            squared_attn=self.squared_attn
        )
        self.train_mcc_mt = MCC()
        self.dev_mcc_mt = MCC()
        self.train_mcc_src = MCC()
        self.dev_mcc_src = MCC()
        self.train_auc_mt = AUC()
        self.dev_auc_mt = AUC()
        self.train_auc_src = AUC()
        self.dev_auc_src = AUC()
        self.train_ap_mt = AveragePrecision()
        self.dev_ap_mt = AveragePrecision()
        self.train_ap_src = AveragePrecision()
        self.dev_ap_src = AveragePrecision()

    def configure_optimizers(self):
        self.epoch_total_steps = len(self.train_dataset) // (self.hparams.batch_size * max(1, self.trainer.num_gpus))
        parameters = [
            {"params": self.estimator.parameters(), "lr": self.hparams.learning_rate},
            {"params": self.model.parameters(), "lr": self.hparams.encoder_learning_rate},
            {"params": self.bottleneck.parameters(), "lr": self.hparams.learning_rate},
        ]
        if self.selected_layers == 'mix':
            parameters.append({"params": self.scalar_mix.parameters(), "lr": self.hparams.learning_rate})

        optimizer = AdamW(
            parameters, lr=self.hparams.learning_rate, correct_bias=True
        )
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
        return_attentions: bool = False,
        return_bottleneck_probas: bool = False
    ):
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = model_output.hidden_states
        embeddings = model_output.hidden_states[0]
        attentions = model_output.attentions

        if self.selected_layers == 'mix':
            hidden_states = self.scalar_mix(hidden_states, attention_mask)
        else:
            selected_layers = list(map(int, self.selected_layers.split(',')))
            hidden_states = torch.stack(hidden_states)[selected_layers].mean(dim=0)

        bottleneck_summary, bottleneck_probas = self.bottleneck(
            hidden_states=hidden_states,
            embeddings=embeddings,
            attention_mask=attention_mask,
            first_sentence_mask=first_sentence_mask,
            first_piece_mask=first_piece_mask,
        )
        mt_score = self.estimator(bottleneck_summary)

        if return_bottleneck_probas:
            if return_attentions:
                return mt_score, bottleneck_probas, attentions
            return mt_score, bottleneck_probas
        if return_attentions:
            return mt_score, attentions
        return mt_score

    def training_step(self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs) -> Dict[str, torch.Tensor]:
        batch_input, batch_target = batch
        predicted_scores, bottleneck_probas = self.forward(**batch_input, return_bottleneck_probas=True)
        if self.num_labels == 1:
            predicted_scores = predicted_scores.view(-1)
            loss_value = self.loss_fn(predicted_scores, batch_target["score"])
        else:
            predicted_probas = torch.exp(predicted_scores)
            mse_predicted_scores = predicted_probas[:, 1].view(-1)
            mse_loss_value = self.mse_loss_fn(mse_predicted_scores, batch_target["score"] / 100)
            nll_loss_value = self.nll_loss_fn(predicted_scores, batch_target["score_bin"])
            loss_value = nll_loss_value + self.mse_lbda * mse_loss_value

        if self.bottleneck_reg != 'none':
            bn_loss_value = 0
            attn_mask = batch_input['attention_mask'].bool()
            fs_mask = batch_input['first_sentence_mask'].bool()
            fp_mask = batch_input['first_piece_mask'].bool()
            tgt_tags = batch_target['tgt_tags']
            src_tags = batch_target['src_tags']
            tgt_mask = tgt_tags != -100
            tgt_tags = tgt_tags.masked_fill(~tgt_mask, 0)
            src_mask = src_tags != -100
            src_tags = src_tags.masked_fill(~src_mask, 0)
            mt_probas = bottleneck_probas[0]
            src_probas = bottleneck_probas[1]
            if src_probas.shape[1] > src_tags.shape[1]:
                # due to padding of joint sequence, add more padding to src_tags at the end (yes, at the end):
                res_len = src_probas.shape[1] - src_tags.shape[1]
                zeros = torch.zeros(src_tags.shape[0], res_len, dtype=src_tags.dtype, device=src_tags.device)
                src_tags = torch.cat([src_tags, zeros], dim=-1)
                src_mask = torch.cat([src_mask, zeros.bool()], dim=-1)
            gold_mt_probas = (tgt_tags / tgt_tags.sum(-1).unsqueeze(-1))
            gold_src_probas = (src_tags / src_tags.sum(-1).unsqueeze(-1))

            if self.aggregation == 'none':
                bounds, bounds_mask = self.bottleneck.get_bounds_from_first_piece_mask(fp_mask)
                mt_probas = self.bottleneck.aggregate_word_pieces(mt_probas, bounds, method='sum')
                src_probas = self.bottleneck.aggregate_word_pieces(src_probas, bounds, method='sum')
                ar = torch.arange(bounds.shape[0], device=bounds.device).unsqueeze(-1)
                fs_mask = fs_mask[ar, bounds]
                fp_mask = fp_mask[ar, bounds]
                attn_mask = bounds_mask

            if self.bottleneck_reg == 'kl':
                log_mt_probas = (mt_probas + 1e-7).log()
                log_src_probas = (src_probas + 1e-7).log()
                tgt_loss = torch.nn.functional.kl_div(log_mt_probas, gold_mt_probas, reduction='none')
                tgt_loss = (tgt_loss * tgt_mask.float()).sum(-1).mean()
                src_loss = torch.nn.functional.kl_div(log_src_probas, gold_src_probas, reduction='none')
                src_loss = (src_loss * src_mask.float()).sum(-1).mean()
                bn_loss_value = tgt_loss + src_loss

            elif self.bottleneck_reg == 'dot_product':
                tgt_loss = -torch.log((mt_probas * tgt_tags.float()).sum(-1).clamp(min=1e-7)).mean()
                src_loss = -torch.log((src_probas * src_tags.float()).sum(-1).clamp(min=1e-7)).mean()
                bn_loss_value = tgt_loss + src_loss
                if torch.isnan(bn_loss_value):
                    import ipdb; ipdb.set_trace()

            elif self.bottleneck_reg == 'l2':
                tgt_loss = torch.norm(mt_probas - gold_mt_probas, p=2, dim=-1).mean()
                src_loss = torch.norm(src_probas - gold_src_probas, p=2, dim=-1).mean()
                bn_loss_value = tgt_loss + src_loss

            elif self.bottleneck_reg == 'fyl':
                from model.fyl_pytorch import SparsemaxLoss
                tgt_loss = SparsemaxLoss(weights='none')(mt_probas, gold_mt_probas)
                tgt_loss = (tgt_loss * tgt_mask.float()).sum(-1).mean()
                src_loss = SparsemaxLoss(weights='none')(src_probas, gold_src_probas)
                src_loss = (src_loss * src_mask.float()).sum(-1).mean()
                bn_loss_value = tgt_loss + src_loss

            elif self.bottleneck_reg == 'bce':
                tgt_loss = torch.nn.functional.binary_cross_entropy_with_logits(mt_probas, tgt_tags.float(), reduction='none')
                tgt_loss = (tgt_loss * tgt_mask.float()).sum(-1).mean()
                src_loss = torch.nn.functional.binary_cross_entropy_with_logits(src_probas, src_tags.float(), reduction='none')
                src_loss = (src_loss * src_mask.float()).sum(-1).mean()
                bn_loss_value = tgt_loss + src_loss

            loss_value += self.bottleneck_lbda * bn_loss_value

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

        # might be useful later:
        # pe_inputs = self.tokenizer.batch_encode(collated_sample["src"], collated_sample["pe"])
        if cuda:
            mt_inputs = move_to_cuda(mt_inputs) if cuda and torch.cuda.is_available() else mt_inputs

        if inference:
            return mt_inputs

        tgt_tags = [torch.tensor(s['tgt_tags']) for s in sample]
        src_tags = [torch.tensor(s['src_tags']) for s in sample]
        targets = {
            "score": torch.tensor(collated_sample["score"], dtype=torch.float),
            "hter": torch.tensor(collated_sample["hter"], dtype=torch.float),
            # NLLLoss will ignore -100 labels directly (hardcoded)
            "tgt_tags": pad_sequence(tgt_tags, padding_value=-100, end=True),
            "src_tags": pad_sequence(src_tags, padding_value=-100, end=False)
        }

        if self.num_labels > 1:
            targets['score_bin'] = (targets['score'] > 50).long()
            targets['hter_bin'] = (targets['hter'] > 0.5).long()

        if cuda:
            targets = move_to_cuda(targets) if cuda and torch.cuda.is_available() else targets

        return mt_inputs, targets

    def validation_step(
            self,
            batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
            batch_nb: int,
            dataloader_idx: int,
            *args, **kwargs
    ) -> None:
        batch_input, batch_target = batch
        predicted_scores, bottleneck_probas = self.forward(**batch_input, return_bottleneck_probas=True)
        tgt_tags = batch_target['tgt_tags']
        src_tags = batch_target['src_tags']
        mt_probas = bottleneck_probas[0]
        src_probas = bottleneck_probas[1]

        tgt_mask = tgt_tags != -100
        src_mask = src_tags != -100
        if src_probas.shape[1] > src_tags.shape[1]:
            # due to padding of joint sequence, add more padding to src_tags at the end (yes, at the end):
            res_len = src_probas.shape[1] - src_tags.shape[1]
            zeros = torch.zeros(src_tags.shape[0], res_len, dtype=src_tags.dtype, device=src_tags.device)
            src_tags = torch.cat([src_tags, zeros], dim=-1)
            src_mask = torch.cat([src_mask, zeros.bool()], dim=-1)

        tgt_tags_flat = tgt_tags[tgt_mask.bool()].long()
        src_tags_flat = src_tags[src_mask.bool()].long()
        mt_probas_flat = mt_probas[tgt_mask.bool()]
        src_probas_flat = src_probas[src_mask.bool()]
        mt_bin_flat = (mt_probas_flat > 0.5).long()
        src_bin_flat = (src_probas_flat > 0.5).long()

        if dataloader_idx == 0:
            if self.num_labels == 1:
                self.log("train_kendall", self.train_kendall(predicted_scores.view(-1), batch_target["score"]))
                self.log("train_pearson", self.train_pearson(predicted_scores.view(-1), batch_target["score"]))
            else:
                predicted_probas = torch.exp(predicted_scores)
                mse_predicted_scores = predicted_probas[:, 1].view(-1)
                nll_predicted_classes = predicted_probas.argmax(-1).view(-1)
                self.log("train_kendall", self.train_kendall(mse_predicted_scores, batch_target["score"] / 100))
                self.log("train_pearson", self.train_pearson(mse_predicted_scores, batch_target["score"] / 100))
                self.log("train_acc", self.train_acc(nll_predicted_classes, batch_target["score_bin"]))
            self.log("train_mcc_mt", self.train_mcc_mt(mt_bin_flat.view(-1), tgt_tags_flat))
            self.log("train_mcc_src", self.train_mcc_src(src_bin_flat.view(-1), src_tags_flat))
            self.log("train_auc_mt", self.train_auc_mt(mt_probas_flat.view(-1).float(), tgt_tags_flat.float()))
            self.log("train_auc_src", self.train_auc_src(src_probas_flat.view(-1).float(), src_tags_flat.float()))
            self.log("train_ap_mt", self.train_ap_mt(mt_probas_flat.view(-1).float(), tgt_tags_flat.float()))
            self.log("train_ap_src", self.train_ap_src(src_probas_flat.view(-1).float(), src_tags_flat.float()))

        if dataloader_idx == 1:
            if self.num_labels == 1:
                self.log("kendall", self.dev_kendall(predicted_scores.view(-1), batch_target["score"]))
                self.log("pearson", self.dev_pearson(predicted_scores.view(-1), batch_target["score"]))
            else:
                predicted_probas = torch.exp(predicted_scores)
                mse_predicted_scores = predicted_probas[:, 1].view(-1)
                nll_predicted_classes = predicted_probas.argmax(-1).view(-1)
                self.log("kendall", self.dev_kendall(mse_predicted_scores, batch_target["score"] / 100))
                self.log("pearson", self.dev_pearson(mse_predicted_scores, batch_target["score"] / 100))
                self.log("acc", self.dev_acc(nll_predicted_classes, batch_target["score_bin"]))
            self.log("mcc_mt", self.dev_mcc_mt(mt_bin_flat.view(-1), tgt_tags_flat))
            self.log("mcc_src", self.dev_mcc_src(src_bin_flat.view(-1), src_tags_flat))
            self.log("auc_mt", self.dev_auc_mt(mt_probas_flat.view(-1).float(), tgt_tags_flat.float()))
            self.log("auc_src", self.dev_auc_src(src_probas_flat.view(-1).float(), src_tags_flat.float()))
            self.log("ap_mt", self.dev_ap_mt(mt_probas_flat.view(-1).float(), tgt_tags_flat.float()))
            self.log("ap_src", self.dev_ap_src(src_probas_flat.view(-1).float(), src_tags_flat.float()))

    def validation_epoch_end(self, *args, **kwargs) -> None:
        self.log("kendall", self.dev_kendall.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.dev_kendall.reset()
        self.log("pearson", self.dev_pearson.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.dev_pearson.reset()
        self.log("mcc_mt", self.dev_mcc_mt.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.dev_mcc_mt.reset()
        self.log("mcc_src", self.dev_mcc_src.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.dev_mcc_src.reset()
        self.log("auc_mt", self.dev_auc_mt.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.dev_auc_mt.reset()
        self.log("auc_src", self.dev_auc_src.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.dev_auc_src.reset()
        self.log("ap_mt", self.dev_ap_mt.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.dev_ap_mt.reset()
        self.log("ap_src", self.dev_ap_src.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.dev_ap_src.reset()

        self.log("train_pearson", self.train_pearson.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.train_pearson.reset()
        self.log("train_kendall", self.train_kendall.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.train_kendall.reset()
        self.log("train_mcc_mt", self.train_mcc_mt.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.train_mcc_mt.reset()
        self.log("train_mcc_src", self.train_mcc_src.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.train_mcc_src.reset()
        self.log("train_auc_mt", self.train_auc_mt.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.train_auc_mt.reset()
        self.log("train_auc_src", self.train_auc_src.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.train_auc_src.reset()
        self.log("train_ap_mt", self.train_ap_mt.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.train_ap_mt.reset()
        self.log("train_ap_src", self.train_ap_src.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.train_ap_src.reset()

        if self.num_labels > 1:
            self.log("acc", self.dev_acc.compute(), on_epoch=True, prog_bar=True, logger=True)
            self.dev_acc.reset()
            self.log("train_acc", self.train_acc.compute(), on_epoch=True, prog_bar=True, logger=True)
            self.train_acc.reset()
