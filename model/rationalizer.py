# -*- coding: utf-8 -*-
import multiprocessing
import os
from argparse import Namespace
from typing import Dict, List, Optional, Tuple, Union

import click
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from model.bottleneck_layer import BottleneckSummary

from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchnlp.utils import collate_tensors
from transformers import AdamW, XLMRobertaModel, BertModel, AutoModel, BertConfig
from utils import Config, read_qe_files

from model.metrics import Pearson, Kendall, Accuracy, AUC, AveragePrecision, MCC
from model.tokenizer import Tokenizer
from model.scalar_mix import ScalarMixWithDropout
from model.utils import move_to_cpu, move_to_cuda, lengths_to_mask, masked_average, pad_sequence
from model.sparse_xlmr import SparseXLMRobertaModel
from model.xlm_roberta import XLMRobertaQE
from model.modules import SelfAdditiveScorer, RelaxedBernoulliGate

from tqdm import tqdm


def load_checkpoint(checkpoint: str, **kwargs):
    if not os.path.exists(checkpoint):
        raise Exception(f"{checkpoint} file not found!")

    hparam_yaml_file = "/".join(checkpoint.split("/")[:-1] + ["hparams.yaml"])

    if os.path.exists(hparam_yaml_file):
        with open(hparam_yaml_file) as yaml_file:
            hparams = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
        model = XLMRationalizer.load_from_checkpoint(checkpoint, hparams={**hparams, **kwargs})
    else:
        raise Exception("hparams file not found.")

    model.eval()
    model.freeze()
    return model


class XLMRationalizer(pl.LightningModule):

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
        z_reg: str = 'none'
        z_lbda: float = 1.0

    def __init__(self, hparams: Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)

        # backwards compatibility
        self.num_labels = self.hparams.get('num_labels', 1)
        self.attention_alpha = self.hparams.get('attention_alpha', 1.0)
        self.mse_lbda = self.hparams.get('mse_lbda', 1.0)
        self.output_norm = self.hparams.get('output_norm', False)
        self.norm_strategy = self.hparams.get('norm_strategy', 'weighted_norm')
        self.effective = self.hparams.get('effective', False)
        self.alpha_merge = self.hparams.get('alpha_merge', 1.0)
        self.return_hidden_states = self.hparams.get('return_hidden_states', False)
        self.z_reg = self.hparams.get('z_reg', 'none')
        self.z_lbda = self.hparams.get('z_lbda', 1.0)

        self.model = SparseXLMRobertaModel.from_pretrained(
            self.hparams.pretrained_model,
            output_hidden_states=True,
            output_attentions=True,
            alpha=self.attention_alpha,
            output_norm=self.output_norm,
            norm_strategy=self.norm_strategy,
            effective=self.effective
        )
        self.tokenizer = Tokenizer(self.hparams.pretrained_model)

        modules = [
            nn.Linear(self.model.config.hidden_size, self.hparams.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_size, self.num_labels),
        ]
        if self.num_labels > 1:
            modules.append(nn.LogSoftmax(dim=-1))
        self.estimator = nn.Sequential(*modules)

        self.scalar_mix = ScalarMixWithDropout(
            mixture_size=self.model.config.num_hidden_layers+1,
            dropout=self.hparams.dropout,
            do_layer_norm=True,
        )

        # Rationalizer Settings for Generator
        self.self_scorer = SelfAdditiveScorer(self.model.config.hidden_size, self.model.config.hidden_size)
        self.z_layer = RelaxedBernoulliGate(self.model.config.hidden_size)

        # Rationalizer Settings for Predictor
        # Last layers of BERT
        
        if self.num_labels == 1:
            self.loss_fn = nn.MSELoss()
        else:
            self.nll_loss_fn = nn.NLLLoss()
            self.mse_loss_fn = nn.MSELoss()

        if self.hparams.load_weights_from_checkpoint:
            self.load_weights(self.hparams.load_weights_from_checkpoint)

        self.train_pearson = Pearson()
        self.dev_pearson = Pearson()
        self.train_kendall = Kendall()
        self.dev_kendall = Kendall()
        self.train_acc = Accuracy()
        self.dev_acc = Accuracy()

        self._frozen = True
        self.freeze_encoder()
        self.freeze_scalar_mix()
        # self.freeze_estimator()
        self.model.eval()
        self.scalar_mix.eval()
        # self.estimator.eval()

        self.train_mcc = MCC()
        self.dev_mcc = MCC()
        self.train_auc = AUC()
        self.dev_auc = AUC()
        self.train_ap = AveragePrecision()
        self.dev_ap = AveragePrecision()

        self.mc_dropout = False
        self.epoch_nr = 0

    def freeze_scalar_mix(self):
        for param in self.scalar_mix.parameters():
            param.requires_grad = False

    def freeze_estimator(self):
        for param in self.estimator.parameters():
            param.requires_grad = False

    def load_weights(self, checkpoint: str) -> None:
        """Function that loads the weights from a given checkpoint file.
        Note:
            If the checkpoint model architecture is different then `self`, only
            the common parts will be loaded.

        :param checkpoint: Path to the checkpoint containing the weights to be loaded.
        """
        click.secho(f"Loading weights from {checkpoint}.", fg="red")
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        pretrained_dict = checkpoint["state_dict"]
        model_dict = self.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(model_dict)

    def configure_optimizers(self):
        self.epoch_total_steps = len(self.train_dataset) // (
            self.hparams.batch_size * max(1, self.trainer.num_gpus)
        )
        parameters = [
            {"params": self.estimator.parameters(), "lr": self.hparams.learning_rate},
            {"params": self.model.parameters(), "lr": self.hparams.encoder_learning_rate},
            {"params": self.scalar_mix.parameters(), "lr": self.hparams.learning_rate},
        ]

        optimizer = AdamW(
            parameters, lr=self.hparams.learning_rate, correct_bias=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=self.hparams.metric_mode),
            "monitor": self.hparams.monitor
        }

    def freeze_embeddings(self) -> None:
        if self.hparams.keep_embeddings_frozen:
            print ("Keeping Embeddings Frozen!")
            for param in self.model.embeddings.parameters():
                param.requires_grad = False
    
    def freeze_encoder(self) -> None:
        self._frozen = True
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        self._frozen = False
        for param in self.model.parameters():
            param.requires_grad = True

    def set_mc_dropout(self, value: bool):
        self.mc_dropout = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mt_eos_ids: torch.Tensor,
        first_sentence_mask: torch.Tensor,
        first_piece_mask: torch.Tensor,
        return_attentions: bool = False,
        **kwargs
    ):

        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # hidden_states = torch.stack(model_output.hidden_states)[[18, 19]].mean(dim=0)
        hidden_states = self.scalar_mix(model_output.hidden_states, attention_mask)
        attentions = model_output.attentions

        scores = hidden_states
        z_dist = self.z_layer(scores, attention_mask)
        z = z_dist.rsample()

        # masked_hidden_states = hidden_states * z
        mt_mask = first_sentence_mask.bool()
        src_mask = (~first_sentence_mask.bool()) & attention_mask.bool()

        mt_summary = masked_average(hidden_states, mt_mask.float() * z.squeeze(-1))
        src_summary = masked_average(hidden_states, src_mask.float() * z.squeeze(-1))
        combined_summary = self.alpha_merge * mt_summary + (1 - self.alpha_merge) * src_summary
        mt_score = self.estimator(combined_summary)

        if return_attentions:
            return mt_score, attentions, z
        return mt_score, z

    def training_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        batch_input, batch_target = batch
        predicted_scores, z = self.forward(**batch_input)

        bounds = batch_input['bounds'].long()
        bounds_mask = batch_input['bounds_mask'].bool()
        word_tags = batch_target['word_tags']
        word_mask = word_tags != -100
        word_tags = word_tags.masked_fill(~word_mask, 0)
        z = BottleneckSummary.aggregate_word_pieces(z, bounds, method='first')

        if self.num_labels == 1:
            predicted_scores = predicted_scores.view(-1)
            loss_value = self.loss_fn(predicted_scores, batch_target["score"])
        else:
            predicted_probas = torch.exp(predicted_scores)
            mse_predicted_scores = predicted_probas[:, 1].view(-1)
            mse_loss_value = self.mse_loss_fn(mse_predicted_scores, batch_target["score"] / 100)
            nll_loss_value = self.nll_loss_fn(predicted_scores, batch_target["score_bin"])
            loss_value = nll_loss_value + self.mse_lbda * mse_loss_value

        if self.z_reg == 'bce':
            y = word_tags.float()
            z = z.squeeze(-1)
            try:
                z_loss = -(y * torch.log(z.clamp(min=1e-7)) + (1 - y) * torch.log((1 - z).clamp(min=1e-7)))
                z_loss = (z_loss * bounds_mask.float()).sum(-1).mean()
            except RuntimeError:
                # I have no idea why, but there are TWO examples in the training data of si-en
                # that contains one more token when tokenized via RobertaTokenizer than word level tags
                # note that we are aggregating word pieces, and the logic for this aggregation seems correct to me
                print('bug...', y.shape, z.shape)
                z_loss = 0
            loss_value += self.z_lbda * z_loss

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
        predicted_scores, z = self.forward(**batch_input)

        bounds = batch_input['bounds'].long()
        bounds_mask = batch_input['bounds_mask'].bool()
        word_tags = batch_target['word_tags']
        word_mask = word_tags != -100
        word_tags = word_tags.masked_fill(~word_mask, 0)
        z = BottleneckSummary.aggregate_word_pieces(z, bounds, method='first')
        z_flat = z[word_mask].view(-1)
        w_flat = word_tags[word_mask].view(-1)

        if dataloader_idx == 0:
            if batch_target["score"].size()[0] > 1:
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
                self.log("train_mcc", self.train_mcc((z_flat > 0.5).long(), w_flat))
                self.log("train_auc", self.train_auc(z_flat.float(), w_flat.float()))
                self.log("train_ap", self.train_ap(z_flat.float(), w_flat.float()))

        if dataloader_idx == 1:
            if batch_target["score"].size()[0] > 1:
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
                self.log("mcc", self.dev_mcc((z_flat > 0.5).long(), w_flat))
                self.log("auc", self.dev_auc(z_flat.float(), w_flat.float()))
                self.log("ap", self.dev_ap(z_flat.float(), w_flat.float()))

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

        if self.num_labels > 1:
            self.log("acc", self.dev_acc.compute(), on_epoch=True, prog_bar=True, logger=True)
            self.dev_acc.reset()
            self.log("train_acc", self.train_acc.compute(), on_epoch=True, prog_bar=True, logger=True)
            self.train_acc.reset()

    def on_train_epoch_end(self, *args, **kwargs) -> None:
        """Hook used to unfreeze encoder during training."""
        self.epoch_nr += 1
        if not self.hparams.use_adapters:
            if self.epoch_nr >= self.hparams.nr_frozen_epochs and self._frozen:
                self.unfreeze_encoder()

    def predict(self, samples, show_progress=True,  cuda=True, batch_size=2):
        if self.training:
            self.eval()

        if cuda and torch.cuda.is_available():
            self.to("cuda")

        batch_size = self.hparams.batch_size if batch_size < 1 else batch_size
        with torch.no_grad():
            batches = [
                samples[i : i + batch_size] for i in range(0, len(samples), batch_size)
            ]
            model_inputs = []
            if show_progress:
                pbar = tqdm(
                    total=len(batches),
                    desc="Preparing batches...",
                    dynamic_ncols=True,
                    leave=None,
                )
            for batch in batches:
                batch = self.prepare_sample(batch, inference=True)
                model_inputs.append(batch)
                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

            if show_progress:
                pbar = tqdm(
                    total=len(batches),
                    desc="Scoring hypothesis...",
                    dynamic_ncols=True,
                    leave=None,
                )
            scores = []
            for model_input in model_inputs:
                if cuda and torch.cuda.is_available():
                    model_input = move_to_cuda(model_input)
                    model_out, _ = self.forward(**model_input)
                    model_out = move_to_cpu(model_out)
                else:
                    model_out, _ = self.forward(**model_input)

                model_scores = model_out.numpy().tolist()
                for i in range(len(model_scores)):
                    scores.append(model_scores[i])

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

        assert len(scores) == len(samples)
        for i in range(len(scores)):
            samples[i]["predicted_score"] = scores[i]
        return samples, scores

    def predict_instance(self, sample: Dict, cuda=True):
        if self.training:
            self.eval()
        with torch.no_grad():
            batch = self.prepare_sample([sample], inference=True)
            batch = move_to_cuda(batch) if cuda and torch.cuda.is_available() else batch
            model_out, _ = self.forward(**batch, return_attentions=True)
            return model_out

    # ------------------------------ DATA ------------------------------
    def read_csv(self, path: str) -> List[dict]:
        """Reads a comma separated value file.

        :param path: path to a csv file.

        :return: List of records as dictionaries
        """
        df = pd.read_csv(path)
        df = df[["src", "mt", "score"]]
        df["src"] = df["src"].astype(str)
        df["mt"] = df["mt"].astype(str)
        df["score"] = df["score"].astype(float)
        return df.to_dict("records")

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
        targets = {
            "score": torch.tensor(collated_sample["score"], dtype=torch.float),
            "hter": torch.tensor(collated_sample["hter"], dtype=torch.float),
            # NLLLoss will ignore -100 labels directly (hardcoded)
            "tgt_tags": pad_sequence(tgt_tags, padding_value=-100, end=True),
            "src_tags": pad_sequence(src_tags, padding_value=-100, end=False),
            "word_tags": pad_sequence(word_tags, padding_value=-100, end=True),
        }

        if self.num_labels > 1:
            targets['score_bin'] = (targets['score'] > 50).long()
            targets['hter_bin'] = (targets['hter'] > 0.5).long()

        if cuda:
            targets = move_to_cuda(targets) if cuda and torch.cuda.is_available() else targets

        return mt_inputs, targets

    def setup(self, stage) -> None:
        self.train_dataset = read_qe_files(self.hparams.train_path)
        self.val_reg_dataset = read_qe_files(self.hparams.val_path)
        
        # Always validate the model with 2k examples from training to control overfit.
        train_subset = np.random.choice(a=len(self.train_dataset), size=2000)
        self.train_subset = Subset(self.train_dataset, train_subset)

        if self.hparams.test_path:
            self.test_dataset = read_qe_files(self.hparams.test_path)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=4,
        )

    def val_dataloader(self) -> List[DataLoader]:
        return [
            DataLoader(
                dataset=self.train_subset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.prepare_sample,
                # num_workers=multiprocessing.cpu_count(),
                num_workers=4  # 0 so ipdb.set_trace() works
            ),
            DataLoader(
                dataset=self.val_reg_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.prepare_sample,
                # num_workers=multiprocessing.cpu_count(),
                num_workers=4  # 0 so ipdb.set_trace() works
            )
        ]

