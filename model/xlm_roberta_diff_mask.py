# -*- coding: utf-8 -*-
import os
from argparse import Namespace
from typing import Dict, List, Optional, Tuple, Union

from model.lookahead import LookaheadRMSprop
from model.metrics import Pearson
from torchmetrics import PearsonCorrcoef

import click
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from model.xlm_roberta import load_checkpoint as load_checkpoint_xlmr_qe
from model.bottleneck_layer import BottleneckSummary
from model.gates import DiffMaskGateInput, DiffMaskGateHidden
from model.getter_setter import bert_getter, bert_setter
from model.utils import move_to_cuda, move_to_cpu, pad_sequence, accuracy_precision_recall_f1

from torch.utils.data import DataLoader, RandomSampler, Subset
from torchnlp.utils import collate_tensors
from transformers import AdamW, get_constant_schedule_with_warmup, get_constant_schedule

from model.utils import move_to_cpu, move_to_cuda, masked_average

from tqdm import tqdm
from utils import Config, read_qe_files


def load_checkpoint(checkpoint: str, **kwargs):
    if not os.path.exists(checkpoint):
        raise Exception(f"{checkpoint} file not found!")

    hparam_yaml_file = "/".join(checkpoint.split("/")[:-1] + ["hparams.yaml"])

    if os.path.exists(hparam_yaml_file):
        with open(hparam_yaml_file) as yaml_file:
            hparams = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
        model = XLMRobertaDiffMask.load_from_checkpoint(checkpoint, hparams={**hparams, **kwargs})
    else:
        raise Exception("hparams file not found.")

    model.eval()
    model.freeze()
    return model


class XLMRobertaDiffMask(pl.LightningModule):

    class ModelConfig(Config):
        pretrained_model: str = "xlm-roberta-large"
        monitor: str = "pearson"
        metric_mode: str = "max"

        # Data configs
        train_path: str = None
        val_path: str = None
        test_path: str = None
        load_weights_from_checkpoint: Union[str, bool] = False

        # Training details
        batch_size: int = 4

        # diff mask
        layer_pred: int = -1
        gate: str = "hidden"
        stop_train: bool = False
        learning_rate: float = 3.0e-5
        learning_rate_alpha: float = 3e-1
        learning_rate_placeholder: float = 1e-3
        eps: float = 0.1
        eps_valid: float = 0.8
        acc_valid: float = 0.75
        placeholder: bool = True

    def __init__(self, hparams: Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)
        print('Loading checkpoint from:', self.hparams.load_weights_from_checkpoint)
        self.qe_model = load_checkpoint_xlmr_qe(self.hparams.load_weights_from_checkpoint)
        self.tokenizer = self.qe_model.tokenizer
        self.epoch_nr = 0

        # freeze base model and set it to eval mode
        self.freeze_qe_model()

        # diff mask
        self.learning_rate = self.hparams.get('learning_rate', 3.0e-4)
        self.layer_pred = self.hparams.get('layer_pred', -1)
        self.gate_type = self.hparams.get('gate', 'hidden')
        self.stop_train = self.hparams.get('stop_train', False)
        self.learning_rate_alpha = self.hparams.get('learning_rate_alpha', 3e-1)
        self.learning_rate_placeholder = self.hparams.get('learning_rate_placeholder', 1e-3)
        self.eps = self.hparams.get('eps', 0.1)
        self.eps_valid = self.hparams.get('eps_valid', 0.8)
        self.acc_valid = self.hparams.get('acc_valid', 0.75)
        self.placeholder = self.hparams.get('placeholder', True)
        self.val_pearson = Pearson()

        gate_cls = DiffMaskGateInput if self.gate_type == "input" else DiffMaskGateHidden
        if self.layer_pred == 0 or self.gate_type == "input":
            init_vector = self.qe_model.model.embeddings.word_embeddings.weight[self.tokenizer.tokenizer.mask_token_id]
        else:
            init_vector = None
        self.gate = gate_cls(
            hidden_size=self.qe_model.model.config.hidden_size,
            hidden_attention=self.qe_model.model.config.hidden_size // 4,
            num_hidden_layers=self.qe_model.model.config.num_hidden_layers + 2,
            max_position_embeddings=1,
            gate_bias=True,  # whether to add bias to linear layers
            placeholder=self.placeholder,  # whether to create a learnable "baseline" vector or to use a zero vector instead
            init_vector=init_vector
        )
        self.alpha = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.ones(()))
                for _ in range(self.qe_model.model.config.num_hidden_layers + 2)
            ]
        )
        self.register_buffer("running_acc", torch.ones((self.qe_model.model.config.num_hidden_layers + 2,)))
        self.register_buffer("running_l0", torch.ones((self.qe_model.model.config.num_hidden_layers + 2,)))
        self.register_buffer("running_steps", torch.zeros((self.qe_model.model.config.num_hidden_layers + 2,)))

    def freeze_qe_model(self) -> None:
        for param in self.qe_model.parameters():
            param.requires_grad = False

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                mt_eos_ids: torch.Tensor,
                first_sentence_mask: torch.Tensor,
                first_piece_mask: torch.Tensor,
                return_attentions: bool = False,
                return_hidden_states: bool = False
    ):
        self.qe_model.eval()
        with torch.no_grad():
            # original_batch_size = self.qe_model.hparams.batch_size
            # current_batch_size = self.hparams.batch_size
            # results = []
            # for i in range(0, current_batch_size, original_batch_size):
            #     res = self.qe_model(
            #         input_ids=input_ids[i:i+original_batch_size],
            #         attention_mask=attention_mask[i:i+original_batch_size],
            #         mt_eos_ids=mt_eos_ids[i:i+original_batch_size],
            #         first_sentence_mask=first_sentence_mask[i:i+original_batch_size],
            #         first_piece_mask=first_piece_mask[i:i+original_batch_size],
            #         return_attentions=False,
            #         return_hidden_states=False,
            #     )
            #     results.append(res)
            # return torch.cat(results, dim=0)
            return self.qe_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mt_eos_ids=mt_eos_ids,
                first_sentence_mask=first_sentence_mask,
                first_piece_mask=first_piece_mask,
                return_attentions=return_attentions,
                return_hidden_states=return_hidden_states,
            )

    def forward_explainer(
            self, inputs_dict, labels=None, layer_pred=None, attribution=False, return_logits=False
    ):
        input_ids = inputs_dict['input_ids']
        mask = inputs_dict['attention_mask']
        logits_orig, hidden_states = bert_getter(self.qe_model.model, inputs_dict, self.forward)

        if layer_pred is None:
            if self.stop_train:
                stop_train = (
                    lambda i: self.running_acc[i] > 0.75
                              and self.running_l0[i] < 0.1
                              and self.running_steps[i] > 100
                )
                p = np.array(
                    [0.1 if stop_train(i) else 1 for i in range(len(hidden_states))]
                )
                layer_pred = torch.tensor(
                    np.random.choice(range(len(hidden_states)), (), p=p / p.sum()),
                    device=input_ids.device,
                )
            else:
                layer_pred = torch.randint(len(hidden_states), ()).item()

        if "hidden" in self.hparams.gate:
            layer_drop = layer_pred
        else:
            layer_drop = 0

        (new_hidden_state, gates, expected_L0, gates_full, expected_L0_full,) = self.gate(
            hidden_states=hidden_states,
            mask=mask,
            layer_pred=None if attribution else layer_pred,
        )

        if attribution and not return_logits:
            return expected_L0_full
        else:
            new_hidden_states = (
                    [None] * layer_drop
                    + [new_hidden_state]
                    + [None] * (len(hidden_states) - layer_drop - 1)
            )
            logits, _ = bert_setter(self.qe_model.model, inputs_dict, new_hidden_states, self.forward)

        if attribution and return_logits:
            return logits, logits_orig, expected_L0_full

        return (
            logits,
            logits_orig,
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
            layer_drop,
            layer_pred,
        )

    def training_step(
            self, batch: Tuple[torch.Tensor], batch_nb: int, optimizer_idx=None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        batch_input, batch_target = batch
        mask = batch_input['attention_mask'].long()
        (
            logits,
            logits_orig,
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
            layer_drop,
            layer_pred,
        ) = self.forward_explainer(batch_input)
        squared_diff = (logits - logits_orig) ** 2
        loss_c = (squared_diff * mask.float()).sum(-1) / mask.sum(-1).float() - self.eps
        loss_g = (expected_L0 * mask.float()).sum(-1) / mask.sum(-1).float()
        loss = 1000.0 * self.alpha[layer_pred] * loss_c + loss_g
        loss_value = loss.mean()

        # acc, _, _, _ = accuracy_precision_recall_f1(
        #     logits.argmax(-1), logits_orig.argmax(-1), average=True
        # )
        # pretend that pearson is acc :-p
        pearson_fn = PearsonCorrcoef()
        acc = pearson_fn(logits.view(-1).cpu(), logits_orig.view(-1).cpu()).cuda()
        l0 = (expected_L0.exp() * mask).sum(-1) / mask.sum(-1)

        # outputs_dict = {
        #     "loss": loss_value,
        #     "loss_c": loss_c.mean(-1),
        #     "loss_g": loss_g.mean(-1),
        #     "alpha": self.alpha[layer_pred].mean(-1),
        #     "acc": acc,
        #     "l0": l0.mean(-1),
        #     "layer_pred": layer_pred,
        #     "r_acc": self.running_acc[layer_pred],
        #     "r_l0": self.running_l0[layer_pred],
        #     "r_steps": self.running_steps[layer_pred],
        # }
        # for k, v in outputs_dict.items():
        #     self.log(k, v.item(), on_step=True, on_epoch=True)
        self.log("loss", loss_value, on_step=True, on_epoch=True)
        self.log("loss_c", loss_c.mean(-1), on_step=True, on_epoch=True)
        self.log("loss_g", loss_g.mean(-1), on_step=True, on_epoch=True)
        self.log("l0", l0.mean(-1), on_step=True, on_epoch=True)
        self.log("acc", acc, on_step=True, on_epoch=True)

        self.running_acc[layer_pred] = self.running_acc[layer_pred] * 0.9 + acc * 0.1
        self.running_l0[layer_pred] = self.running_l0[layer_pred] * 0.9 + l0.mean(-1) * 0.1
        self.running_steps[layer_pred] += 1

        return loss_value

    def validation_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_nb: int,
        dataloader_idx: int,
        *args, **kwargs
    ):
        self.eval()
        batch_input, batch_target = batch
        mask = batch_input['attention_mask'].long()
        (
            logits,
            logits_orig,
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
            layer_drop,
            layer_pred,
        ) = self.forward_explainer(batch_input)
        squared_diff = (logits - logits_orig) ** 2
        loss_c = (squared_diff * mask.float()).sum(-1) / mask.sum(-1).float() - self.eps
        loss_g = (expected_L0 * mask.float()).sum(-1) / mask.sum(-1).float()
        loss = 1000.0 * self.alpha[layer_pred] * loss_c + loss_g
        loss_value = loss.mean(-1)
        acc = self.val_pearson(logits.view(-1), logits_orig.view(-1))
        self.log("val_pearson", acc)

        l0 = (expected_L0.exp() * mask).sum(-1) / mask.sum(-1)
        outputs_dict = {
            "val_loss": loss_value,
            "val_loss_c": loss_c.mean(-1),
            "val_loss_g": loss_g.mean(-1),
            "val_alpha": self.alpha[layer_pred].mean(-1),
            "val_acc": acc,
            "val_l0": l0.mean(-1),
            "val_layer_pred": layer_pred,
            "val_r_acc": self.running_acc[layer_pred],
            "val_r_l0": self.running_l0[layer_pred],
            "val_r_steps": self.running_steps[layer_pred],
        }
        return outputs_dict

    def validation_epoch_end(self, outputs):
        outputs_val = outputs[1]
        outputs_dict = {
            k: [e[k] for e in outputs_val if k in e]
            for k in ("val_loss_c", "val_loss_g", "val_acc", "val_l0")
        }
        outputs_dict = {k: sum(v) / len(v) for k, v in outputs_dict.items()}
        outputs_dict["val_loss_c"] += self.eps
        val_pearson = self.val_pearson.compute()
        outputs_dict = {
            "val_loss": outputs_dict["val_l0"]
            if outputs_dict["val_loss_c"] <= self.eps_valid and val_pearson >= self.acc_valid
            else torch.full_like(outputs_dict["val_l0"], float("inf")),
            **outputs_dict,
        }


        # print(
        #     "Epoch {}: Validation accuracy = {:.2f}, gates at zero = {:.2%}, constraint = {:.5f}".format(
        #         self.epoch_nr,
        #         outputs_dict["val_acc"],
        #         1 - outputs_dict["val_l0"],
        #         outputs_dict["val_loss_c"],
        #     )
        # )
        self.log("val_pearson", val_pearson, on_epoch=True, prog_bar=True, logger=True)
        self.val_pearson.reset()
        for k, v in outputs_dict.items():
            self.log(k, v, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        self.epoch_total_steps = len(self.train_dataset) // (self.hparams.batch_size * max(1, self.trainer.num_gpus))
        optimizers = [
            AdamW(
                params=[
                    {
                        "params": self.gate.g_hat.parameters(),
                        "lr": self.learning_rate,
                    },
                    {
                        "params": self.gate.placeholder.parameters()
                        if isinstance(self.gate.placeholder, torch.nn.ParameterList)
                        else [self.gate.placeholder],
                        "lr": self.learning_rate_placeholder,
                    },
                ],
                # centered=True,
            ),
            AdamW(
                params=[self.alpha]
                if isinstance(self.alpha, torch.Tensor)
                else self.alpha.parameters(),
                lr=self.learning_rate_alpha,
            ),
        ]

        schedulers = [
            {
                "scheduler": get_constant_schedule_with_warmup(optimizers[0], 12 * 100),
                "interval": "step",
            },
            get_constant_schedule(optimizers[1]),
        ]
        return optimizers, schedulers

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, **kwargs):
        if optimizer_idx == 0:
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    p.grad = None

        elif optimizer_idx == 1:
            for i in range(len(self.alpha)):
                if self.alpha[i].grad:
                    self.alpha[i].grad *= -1

            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    p.grad = None

            for i in range(len(self.alpha)):
                self.alpha[i].data = torch.where(
                    self.alpha[i].data < 0,
                    torch.full_like(self.alpha[i].data, 0),
                    self.alpha[i].data,
                    )
                self.alpha[i].data = torch.where(
                    self.alpha[i].data > 200,
                    torch.full_like(self.alpha[i].data, 200),
                    self.alpha[i].data,
                    )

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
            num_workers=1,
        )

    def val_dataloader(self) -> List[DataLoader]:
        return [
            DataLoader(
                dataset=self.train_subset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.prepare_sample,
                # num_workers=multiprocessing.cpu_count(),
                num_workers=1  # 0 so ipdb.set_trace() works
            ),
            DataLoader(
                dataset=self.val_reg_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.prepare_sample,
                # num_workers=multiprocessing.cpu_count(),
                num_workers=1  # 0 so ipdb.set_trace() works
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
