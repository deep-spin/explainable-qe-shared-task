# -*- coding: utf-8 -*-
r"""
Command Line Interface
=======================
   Commands:
   - train: for Training a new model.
   - interact: Model interactive mode where we can "talk" with a trained model.
   - test: Tests the model ability to rank candidate answers and generate text.
"""
import math

import click
import yaml
import optuna
from functools import partial

from pytorch_lightning import seed_everything
from trainer import TrainerConfig, build_trainer


@click.group()
def cli():
    pass


@cli.command(name="train")
@click.option(
    "--config",
    "-f",
    type=click.Path(exists=True),
    required=True,
    help="Path to the configure YAML file",
)
@click.option(
    "--model_type",
    required=True,
    help="Type of model: XLMRobertaQE, XLMRationalizer, XLMRobertaBottlenecked",
    default='XLMRobertaQE'
)
def train(config: str, model_type: str) -> None:
    yaml_file = yaml.load(open(config).read(), Loader=yaml.FullLoader)
    # Build Trainer
    train_configs = TrainerConfig(yaml_file)
    seed_everything(train_configs.seed)
    trainer = build_trainer(train_configs.namespace())

    # Build Model
    if model_type == "XLMRationalizer":
        from model.rationalizer import XLMRationalizer
        model_config = XLMRationalizer.ModelConfig(yaml_file)
        model = XLMRationalizer(model_config.namespace())
        trainer.fit(model)
    elif model_type == "XLMRobertaBottlenecked":
        from model.xlm_roberta_bottlenecked import XLMRobertaBottlenecked
        model_config = XLMRobertaBottlenecked.ModelConfig(yaml_file)
        model = XLMRobertaBottlenecked(model_config.namespace())
        trainer.fit(model)
    elif model_type == "XLMRobertaWithWordLevel":
        from model.xlm_roberta_word_level import XLMRobertaWithWordLevel
        model_config = XLMRobertaWithWordLevel.ModelConfig(yaml_file)
        model = XLMRobertaWithWordLevel(model_config.namespace())
        trainer.fit(model)
    elif model_type == "MBart":
        from model.mbart50 import MBartModelQE
        model_config = MBartModelQE.ModelConfig(yaml_file)
        model = MBartModelQE(model_config.namespace())
        trainer.fit(model)
    elif model_type == "ByT5":
        from model.byt5 import ByT5ModelQE
        model_config = ByT5ModelQE.ModelConfig(yaml_file)
        model = ByT5ModelQE(model_config.namespace())
        trainer.fit(model)
    elif model_type == "RemBERT":
        from model.rembert import RemBERTModelQE
        model_config = RemBERTModelQE.ModelConfig(yaml_file)
        model = RemBERTModelQE(model_config.namespace())
        trainer.fit(model)
    elif model_type == "RemBERTWithWordLevel":
        from model.rembert_word_level import RemBERTModelQE
        model_config = RemBERTModelQE.ModelConfig(yaml_file)
        model = RemBERTModelQE(model_config.namespace())
        trainer.fit(model)
    elif model_type == "XLMRComp":
        from model.xlm_roberta_comp import XLMRobertaDiffMask
        model_config = XLMRobertaDiffMask.ModelConfig(yaml_file)
        model = XLMRobertaDiffMask(model_config.namespace())
        trainer.fit(model)
    elif model_type == "XLMRDiffMask":
        from model.xlm_roberta_diff_mask import XLMRobertaDiffMask
        model_config = XLMRobertaDiffMask.ModelConfig(yaml_file)
        model = XLMRobertaDiffMask(model_config.namespace())
        trainer.fit(model)
    else:
        from model.xlm_roberta import XLMRobertaQE
        model_config = XLMRobertaQE.ModelConfig(yaml_file)
        model = XLMRobertaQE(model_config.namespace())
        trainer.fit(model)


@cli.command(name="search")
@click.option(
    "--config",
    "-f",
    type=click.Path(exists=True),
    required=True,
    help="Path to the configure YAML file",
)
@click.option(
    "--n_trials",
    type=int,
    default=15,
    help="Number of search trials",
)
def search(config: str, n_trials: int) -> None:
    
    def objective(trial, train_config, model_config):
        model_config.learning_rate = trial.suggest_loguniform(
            "learning_rate", 1e-6, 1e-4
        )
        seed_everything(train_config.seed)
        trainer = build_trainer(train_config.namespace())
        model = XLMRobertaQE(model_config.namespace())
        try:
            trainer.fit(model)
        except RuntimeError:
            click.secho("CUDA OUT OF MEMORY, SKIPPING TRIAL", fg="red")
            return -1

        best_score = trainer.callbacks[0].best_score.item()
        return -1 if math.isnan(best_score) else best_score
    
    yaml_file = yaml.load(open(config).read(), Loader=yaml.FullLoader)
    train_config = TrainerConfig(yaml_file)
    model_config = XLMRobertaQE.ModelConfig(yaml_file)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    try:
        study.optimize(
            partial(objective, train_config=train_config, model_config=model_config),
            n_trials=n_trials,
        )

    except KeyboardInterrupt:
        click.secho("Early stopping search caused by ctrl-C", fg="red")

    except Exception as e:
        click.secho(
            f"Error occured during search: {e}; current best params are {study.best_params}",
            fg="red",
        )

    try:
        click.secho(
            "Number of finished trials: {}".format(len(study.trials)), fg="yellow"
        )
        click.secho("Best trial:", fg="yellow")
        trial = study.best_trial
        click.secho("  Value: {}".format(trial.value), fg="yellow")
        click.secho("  Params: ", fg="yellow")
        for key, value in trial.params.items():
            click.secho("    {}: {}".format(key, value), fg="blue")

    except Exception as e:
        click.secho(f"Logging at end of search failed: {e}", fg="red")

    click.secho(f"Saving Optuna plots for this search to experiments/", fg="yellow")
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html("experiments/optimization_history.html")

    except Exception as e:
        click.secho(f"Failed to create plot: {e}", fg="red")

    try:
        fig = optuna.visualization.plot_parallel_coordinate(
            study, params=list(trial.params.keys())
        )
        fig.write_html("experiments/parallel_coordinate.html")

    except Exception as e:
        click.secho(f"Failed to create plot: {e}", fg="red")

    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html("experiments/param_importances.html")

    except Exception as e:
        click.secho(f"Failed to create plot: {e}", fg="red")



if __name__ == "__main__":
    cli()