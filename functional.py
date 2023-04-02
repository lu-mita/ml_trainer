# coding: utf-8
import os

import pandas as pd
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from ignite.metrics import RunningAverage, Precision, Loss, Recall, Fbeta, ClassificationReport
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .utils import set_seed
from .utils import threshold_output_transform
from .utils import TextualDataset


def get_train_metrics():
    train_metrics = {'loss': RunningAverage(output_transform=lambda x: x)}
    return train_metrics


def get_eval_metrics():
    eval_metrics = {
        'ce': Loss(cross_entropy),
        'f1': Fbeta(
                beta=1.0,
                precision=Precision(average=False, is_multilabel=True, output_transform=threshold_output_transform),
                recall=Recall(average=False, is_multilabel=True, output_transform=threshold_output_transform)
            ),
    }
    return eval_metrics



def get_loop_handlers(model, optimizer: torch.optim.Optimizer, patience: int):

    def process_function(engine: Engine, batch):
        model.train()
        optimizer.zero_grad()
        output = model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()
        return loss.item()

    def eval_function(engine: Engine, batch):
        model.eval()
        with torch.no_grad():
            output = model(**batch)
        return output.logits, batch['labels']

    def score_function(engine: Engine):
        return -engine.state.metrics['f1']

    trainer = Engine(process_function)
    train_evaluator = Engine(eval_function)
    validation_evaluator = Engine(eval_function)

    early_stopping_handler = EarlyStopping(
        patience=patience,
        score_function=score_function,
        trainer=trainer,
    )
    validation_evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)

    return trainer, train_evaluator, validation_evaluator




def train_model(model, train_dl, val_dl, optimizer, scheduler, max_epochs: int = 5, patience: int = 3):
    trainer, train_evaluator, validation_evaluator = get_loop_handlers(model, optimizer, patience)
    for name, metric in get_train_metrics().items():
        metric.attach(trainer, name)

    for name, metric in get_eval_metrics().items():
        metric.attach(train_evaluator, name)
        metric.attach(validation_evaluator, name)
    
    logs = {}
    progress_bar = ProgressBar(persist=True, bar_format="")
    progress_bar.attach(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine: Engine):
        train_evaluator.run(train_dl)
        metrics = train_evaluator.state.metrics
        logs[engine.state.epoch] = {'train_loss': metrics['ce'], 'train_f1': metrics['f1']}
        message = f"Training Results - Epoch {engine.state.epoch} - " \
            f"train_loss: {metrics['ce']:.2f} - " \
            f"train_f1: {metrics['f1']:.2f}"
        progress_bar.log_message(message)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        validation_evaluator.run(val_dl)
        metrics = validation_evaluator.state.metrics
        logs[engine.state.epoch] = {
            **logs[engine.state.epoch],
            'validation_loss': metrics['ce'],
            'validation_f1': metrics['f1']
        }
        message = f"Validation Results - Epoch {engine.state.epoch} - " \
            f"validation_loss: {metrics['ce']:.2f} - " \
            f"validation_f1: {metrics['f1']:.2f}"
        progress_bar.log_message(message)
   
    trainer.run(train_dl, max_epochs=max_epochs)
    return logs



def model_init(model_name_or_path: str, num_labels: int = 10, hidden_dropout_prob = 0.1, **kwargs):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        output_attentions=False,
        output_hidden_states=False,
        hidden_dropout_prob=hidden_dropout_prob,
        ignore_mismatched_sizes=True,
        **kwargs
    )
    return model



def validate_model(model, val_dl, labels):

    def eval_function(engine: Engine, batch):
        model.eval()
        with torch.no_grad():
            output = model(**batch)
        return output.logits, batch['labels']

    evaluator = Engine(eval_function)
    report = ClassificationReport(beta=1, output_transform=threshold_output_transform, is_multilabel=True, labels=labels, output_dict=True)
    report.attach(evaluator, 'report')
    evaluator.run(val_dl)
    # evaluator.state.metrics['report']['macro avg']['f1-score']
    return evaluator.state.metrics['report']['macro avg']['f1-score'], evaluator.state.metrics['report']



def objective(
        model_name_or_path,
        train_dl: DataLoader,
        val_dl: DataLoader,
        labels: list = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Love', 'Neutral', 'Sadness', 'Surprise', 'Trust'],
        device: torch.device = torch.device('cpu'),
        max_epochs: int = 5,
        patience: int = 3,
        num_labels: int = 10,
        hidden_dropout_prob: float = 0.1,
        learning_rate: float = 5e-5,
        **kwargs
    ):

    num_labels = len(labels)
    model = model_init(model_name_or_path, num_labels, hidden_dropout_prob=hidden_dropout_prob)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    train_logs = train_model(model, train_dl, val_dl, optimizer, None, max_epochs=max_epochs, patience=patience)
    f1, report = validate_model(model, val_dl, labels)
    return model, f1, train_logs, report

