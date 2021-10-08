# -*- coding: utf-8 -*-
import multiprocessing
import os
from argparse import Namespace
from typing import Dict, List, Tuple, Union

import click
import numpy as np
import pytorch_lightning as pl
import torch
import yaml

from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchnlp.utils import collate_tensors
from tqdm import tqdm
from transformers import AdamW, T5Model, ByT5Tokenizer, T5Config
from utils import Config, read_qe_files

from model.metrics import Pearson, Kendall, Accuracy, AUC, AveragePrecision
from model.scalar_mix import ScalarMixWithDropout
from model.utils import move_to_cpu, move_to_cuda, lengths_to_mask, masked_average, pad_sequence

from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors


def load_checkpoint(checkpoint: str, **kwargs):
    if not os.path.exists(checkpoint):
        raise Exception(f"{checkpoint} file not found!")

    hparam_yaml_file = "/".join(checkpoint.split("/")[:-1] + ["hparams.yaml"])

    if os.path.exists(hparam_yaml_file):
        with open(hparam_yaml_file) as yaml_file:
            hparams = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
        model = ByT5ModelQE.load_from_checkpoint(checkpoint, hparams={**hparams, **kwargs})
    else:
        raise Exception("hparams file not found.")

    model.eval()
    model.freeze()
    return model


class Tokenizer:

    def __init__(self, pretrained_model):
        self.tokenizer = ByT5Tokenizer.from_pretrained(pretrained_model)
        self.pad_index = self.tokenizer.pad_token_id
        self.eos_index = self.tokenizer.eos_token_id
        self.bos_index = self.tokenizer.eos_token_id
        self.stoi = self.tokenizer.get_vocab()
        self.itos = {v: k for k, v in self.stoi.items()}
        self.configs = T5Config.from_pretrained(pretrained_model)
        self.max_positions = 1000000

    def batch_encode(self, sources: list, hypothesis: list):
        encoded_batch = {
            "input_ids": [],
            "attention_mask": [],
            "first_piece_mask": [],
            "decoder_input_ids": [],
            "decoder_attention_mask": [],
            "decoder_first_piece_mask": [],
        }
        for mt, src in zip(hypothesis, sources):
            src_inputs = self.tokenizer(src)
            src_input_ids = src_inputs["input_ids"]
            src_attention_mask = src_inputs["attention_mask"]
            src_fp_mask = [int(s.startswith("▁")) for s in self.tokenizer.convert_ids_to_tokens(src_input_ids)]
            with self.tokenizer.as_target_tokenizer():
                mt_inputs = self.tokenizer(mt)
                mt_input_ids = mt_inputs["input_ids"]
                mt_attention_mask = mt_inputs["attention_mask"]
                mt_fp_mask = [int(s.startswith("▁")) for s in self.tokenizer.convert_ids_to_tokens(mt_input_ids)]
            encoded_batch["input_ids"].append(src_input_ids)
            encoded_batch["attention_mask"].append(src_attention_mask)
            encoded_batch["first_piece_mask"].append(src_fp_mask)
            encoded_batch["decoder_input_ids"].append(mt_input_ids)
            encoded_batch["decoder_attention_mask"].append(mt_attention_mask)
            encoded_batch["decoder_first_piece_mask"].append(mt_fp_mask)
        model_input = {}
        for k, v in encoded_batch.items():
            padded_input = stack_and_pad_tensors([torch.tensor(l) for l in v])
            model_input[k] = padded_input.tensor
        return model_input


class ByT5ModelQE(pl.LightningModule):

    class ModelConfig(Config):
        pretrained_model: str = "google/byt5-large"

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
        attention_alpha: float = 1.0
        output_norm: bool = False
        alpha_merge: float = 1.0
        selected_layers: str = 'mix'

    def __init__(self, hparams: Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)

        # backwards compatibility
        self.num_labels = 1
        self.attention_alpha = self.hparams.get('attention_alpha', 1.0)
        self.output_norm = self.hparams.get('output_norm', False)
        self.alpha_merge = self.hparams.get('alpha_merge', 1.0)
        self.selected_layers = self.hparams.get('selected_layers', 'mix')

        # byt5
        self.model = T5Model.from_pretrained(
            self.hparams.pretrained_model,
            output_hidden_states=True,
            output_attentions=True,
        )
        self.tokenizer = Tokenizer(self.hparams.pretrained_model)
        self.estimator = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.hparams.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_size, 1)
        )
        self.scalar_mix = ScalarMixWithDropout(
            mixture_size=self.model.config.num_hidden_layers+1,
            dropout=self.hparams.dropout,
            do_layer_norm=True,
        )
        self.loss_fn = nn.MSELoss()

        if self.hparams.load_weights_from_checkpoint:
            self.load_weights(self.hparams.load_weights_from_checkpoint)

        self.train_pearson = Pearson()
        self.dev_pearson = Pearson()
        self.train_kendall = Kendall()
        self.dev_kendall = Kendall()

        if self.hparams.use_adapters:
            click.secho("Adding Adaptors.", fg="red")
            self.model.add_adapter("regression")
            self.model.train_adapter("regression")

        else:
            if self.hparams.nr_frozen_epochs > 0:
                self._frozen = True
                self.freeze_encoder()
            else:
                self._frozen = False

            if self.hparams.keep_embeddings_frozen:
                self.freeze_embeddings()
            
        self.epoch_nr = 0

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

    def freeze_embeddings(self) -> None:
        if self.hparams.keep_embeddings_frozen:
            print ("Keeping Embeddings Frozen!")
            for param in self.model.shared.parameters():
                param.requires_grad = False
    
    def freeze_encoder(self) -> None:
        self._frozen = True
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        self._frozen = False
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                first_piece_mask: torch.Tensor,
                decoder_input_ids,
                decoder_attention_mask,
                decoder_first_piece_mask,
                return_hidden_states: bool = False,
                return_attentions: bool = False):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask
        )
        if self.selected_layers == 'mix':
            hidden_states = self.scalar_mix(output.decoder_hidden_states, decoder_attention_mask)
        else:
            selected_layers = list(map(int, self.selected_layers.split(',')))
            hidden_states = torch.stack(output.decoder_hidden_states)[selected_layers].mean(dim=0)
        # hidden_states = output.last_hidden_state  # last decoder hidden state
        mt_summary = hidden_states.mean(1)
        mt_score = self.estimator(mt_summary)

        if return_hidden_states:
            return mt_score, output.encoder_hidden_states, output.decoder_hidden_states, hidden_states
        if return_attentions:
            return mt_score, output.encoder_attentions, output.cross_attentions, output.decoder_attentions
        return mt_score

    def training_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        batch_input, batch_target = batch
        predicted_scores = self.forward(**batch_input)
        predicted_scores = predicted_scores.view(-1)
        loss_value = self.loss_fn(predicted_scores, batch_target["score"])
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
        predicted_scores = self.forward(**batch_input)
        if dataloader_idx == 0:
            self.log("train_kendall", self.train_kendall(predicted_scores.view(-1), batch_target["score"]))
            self.log("train_pearson", self.train_pearson(predicted_scores.view(-1), batch_target["score"]))
        if dataloader_idx == 1:
            self.log("kendall", self.dev_kendall(predicted_scores.view(-1), batch_target["score"]))
            self.log("pearson", self.dev_pearson(predicted_scores.view(-1), batch_target["score"]))
            
    def validation_epoch_end(self, *args, **kwargs) -> None:
        self.log("kendall", self.dev_kendall.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.dev_kendall.reset()
        self.log("pearson", self.dev_pearson.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.dev_pearson.reset()
        self.log("train_pearson", self.train_pearson.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.train_pearson.reset()
        self.log("train_kendall", self.train_kendall.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.train_kendall.reset()

    def on_train_epoch_end(self, *args, **kwargs) -> None:
        """Hook used to unfreeze encoder during training."""
        self.epoch_nr += 1
        if not self.hparams.use_adapters:
            if self.epoch_nr >= self.hparams.nr_frozen_epochs and self._frozen:
                self.unfreeze_encoder()

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

        # bounds, bounds_mask = BottleneckSummary.get_bounds_from_first_piece_mask(mt_inputs['first_piece_mask'])
        # mt_inputs['bounds'] = bounds
        # mt_inputs['bounds_mask'] = bounds_mask

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
                    model_out = self.forward(**model_input)
                    model_out = move_to_cpu(model_out)
                else:
                    model_out = self.forward(**model_input)

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
