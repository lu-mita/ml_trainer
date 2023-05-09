# coding: utf-8
from typing import List, Union
import torch
import optuna
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import EarlyStopping
from ignite.metrics import RunningAverage, Precision, Loss, Recall, Fbeta, ClassificationReport
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .utils import threshold_output_transform
from .integration import CustomPruningHandler


def get_early_stopping_handler(trainer, patience: int = 3):
    def score_function(engine: Engine):
        return -engine.state.metrics['f1']
    
    early_stopping_handler = EarlyStopping(
        patience=patience,
        score_function=score_function,
        trainer=trainer
    )
    return early_stopping_handler


def create_supervised_trainer(model, optimizer: torch.optim.Optimizer):
    def process_function(engine: Engine, batch):
        model.train()
        optimizer.zero_grad()
        output = model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()
        return loss.item()
    
    return Engine(process_function)


def create_supervised_evaluator(model):
    def eval_function(engine: Engine, batch):
        model.eval()
        with torch.no_grad():
            output = model(**batch)
        return output.logits, batch['labels']
    
    return Engine(eval_function)


def create_supervised_train_evaluator(model):
    def eval_function(engine: Engine, batch):
        model.eval()
        with torch.no_grad():
            output = model(**batch)
        return output.logits, batch['labels']
    
    return Engine(eval_function)


def get_train_metrics():
    train_metrics = {'loss': RunningAverage(output_transform=lambda x: x)}
    return train_metrics


def get_eval_metrics(labels):
    eval_metrics = {
        'ce': Loss(cross_entropy),
        'f1': Fbeta(
                beta=1.0,
                precision=Precision(average=False, is_multilabel=True, output_transform=threshold_output_transform),
                recall=Recall(average=False, is_multilabel=True, output_transform=threshold_output_transform),
            ),
        'report': ClassificationReport(beta=1, output_transform=threshold_output_transform, is_multilabel=True, labels=labels, output_dict=True)
    }
    return eval_metrics


def train(
        model: torch.nn.Module,
        train_dl: torch.utils.data.DataLoader,
        val_dl: torch.utils.data.DataLoader,
        learning_rate: float = 5e-5,
        max_epochs: int = 10,
        patience: int = 3,
        integrations: dict = {},
        device: torch.device = torch.device('cpu'),
        **kargs
    ):
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        labels: list[str] = kargs.get('labels')

        # integrations
        trial: optuna.trial.Trial | None = integrations.get('trial', None)
        wandb = integrations.get('wandb', None)
        logs = {}

        trainer = create_supervised_trainer(model, optimizer)
        evaluator = create_supervised_evaluator(model)
        train_evaluator = create_supervised_train_evaluator(model)

        for name, metric in get_train_metrics().items():
            metric.attach(trainer, name)

        for name, metric in get_eval_metrics(labels).items():
            metric.attach(train_evaluator, name)
            metric.attach(evaluator, name)

        if trial:
            # Register a pruning handler to the evaluator.
            pruning_handler = CustomPruningHandler(trial, "f1", trainer, wandb=wandb)
            evaluator.add_event_handler(Events.COMPLETED, pruning_handler)
        else:
            early_stopping_handler = get_early_stopping_handler(trainer, patience=patience)
            evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)
        
        progress_bar = ProgressBar(persist=True, bar_format="")
        progress_bar.attach(trainer)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine: Engine):
            train_evaluator.run(train_dl)
            metrics = train_evaluator.state.metrics
            message = \
                f"Training Results - Epoch {engine.state.epoch} - " \
                f"train_loss: {metrics['ce']:.2f} - " \
                f"train_f1: {metrics['f1']:.2f}"
            logs[engine.state.epoch] = {
                'train_loss': metrics['ce'],
                'train_f1': metrics['f1'],
                **transform_report(metrics['report'], prefix="train")
            }
            progress_bar.log_message(message)


        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_dl)
            metrics = evaluator.state.metrics
            message = \
                f"Validation Results - Epoch {engine.state.epoch} - " \
                f"validation_loss: {metrics['ce']:.2f} - " \
                f"validation_f1: {metrics['f1']:.2f}"
            logs[engine.state.epoch].update({
                'validation_loss': metrics['ce'],
                'f1': metrics['f1'],
                **transform_report(metrics['report'], prefix="validation")
            })
            progress_bar.log_message(message)
            if wandb:
                wandb.log(data={**logs[engine.state.epoch]}, step=engine.state.epoch)

        trainer.run(train_dl, max_epochs=max_epochs)

        if wandb:
            wandb.run.summary["f1-score"] = evaluator.state.metrics['report']['macro avg']['f1-score']
            wandb.run.summary["state"] = "completed"
        return evaluator.state.metrics['report']['macro avg']['f1-score']


def transform_report(report, prefix: str = ""):
    if len(prefix) > 0:
        prefix = prefix + "_"
    metrics_to_log = {f"{prefix}{label.lower()}_{metric}": report[label][metric] for label in report.keys() for metric in report[label] if label != 'macro avg'}
    metrics_to_log['f1-score'] = report['macro avg']['f1-score']
    return metrics_to_log


def model_init(model_ref: str = "dbmdz/bert-base-italian-xxl-cased", num_labels: int = 10, hidden_dropout_prob: float = 0.1, **kwargs):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        output_attentions=False,
        output_hidden_states=False,
        hidden_dropout_prob=hidden_dropout_prob,
        ignore_mismatched_sizes=True
    )
    return model
