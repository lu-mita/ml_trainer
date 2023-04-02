import hashlib
import json
import os
import random
from typing import Callable, Sequence, Union
from typing import Dict, Any
import math
import numpy as np
import pandas as pd
import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce
from torch import tensor
from torch.utils.data import Dataset


# __all__ = ["MultiLabelConfusionMatrix", "set_seed", "CamembertTextualDataset", "BertTextualDataset", "thresholded_output_transform"]

def threshold_output_transform(output, threshold=0.5):
    """
    In binary and multilabel cases, the elements of y and y_pred should have 0 or 1 values.
    """
    logits, labels = output
    predictions = torch.sigmoid(logits)
    predictions = torch.where(predictions > threshold, 1, 0)
    return predictions, labels


class TextualDataset(Dataset):
    def __init__(self, src, device: torch.device = torch.device('cpu')):
        if isinstance(src, str):
            self.data = []
            with open(src, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))
        else:
            self.data = src
        self.device = device

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        type_map = {
            'input_ids': torch.long,
            'attention_mask': torch.float,
            'token_type_ids': torch.long,
            'labels': torch.float
        }
        row = self.data[item]
        row = {k: tensor(v, dtype=type_map[k]).to(self.device) for k, v in row.items()}
        return row



def compute_metrics_per_label(confusion_matrix: torch.Tensor):
    tn, fp, fn, tp = confusion_matrix.numpy().ravel()
    metrics = {}
    metrics["True positives"] = tp
    metrics["True negatives"] = tn
    metrics["False positives"] = fp
    metrics["False negatives"] = fn
    try:
        metrics["Precision"] = tp / (tp + fp)
    except ZeroDivisionError | TypeError | RuntimeError:
        metrics["Precision"] = None
    try:
        metrics["Recall"] = tp / (tp + fn)
    except ZeroDivisionError | TypeError | RuntimeError:
        metrics["Recall"] = None
    try:
        metrics["F1 score"] = 2 * tp / (2 * tp + fp + fn)
    except ZeroDivisionError | TypeError | RuntimeError:
        metrics["F1 score"] = None
    try:
        metrics["Accuracy"] = (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError | TypeError | RuntimeError:
        metrics["Accuracy"] = None
    try:
        metrics["Specificity"] = tn / (tn + fp)
    except ZeroDivisionError | TypeError | RuntimeError:
        metrics["Specificity"] = None
    try:
        metrics["Sensitivity"] = tp / (tp + fn)
    except ZeroDivisionError | TypeError | RuntimeError:
        metrics["Sensitivity"] = None
    try:
        metrics["MCC"] = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    except ZeroDivisionError | TypeError | RuntimeError:
        metrics["MCC"] = None
    try:
        metrics["FPR"] = fp / (fp + tn)
    except ZeroDivisionError | TypeError | RuntimeError:
        metrics["FPR"] = None
    try:
        metrics["FNR"] = fn / (tp + fn)
    except ZeroDivisionError | TypeError | RuntimeError:
        metrics["FNR"] = None
    try:
        metrics["FDR"] = fp / (tp + fp)
    except ZeroDivisionError | TypeError | RuntimeError:
        metrics["FDR"] = None
    try:
        metrics["FOR"] = fn / (tn + fn)
    except ZeroDivisionError | TypeError | RuntimeError:
        metrics["FOR"] = None
    try:
        metrics["NPV"] = tn / (tn + fp)
    except ZeroDivisionError | TypeError | RuntimeError:
        metrics["NPV"] = None
    try:
        metrics["PLR"] = tp / (fn + tp)
    except ZeroDivisionError | TypeError | RuntimeError:
        metrics["PLR"] = None
    try:
        metrics["NLR"] = fn / (tp + fn)
    except ZeroDivisionError | TypeError | RuntimeError:
        metrics["NLR"] = None
    return metrics
"""
    return {
        "True positives": tp,
        "True negatives": tn,
        "False positives": fp,
        "False negatives": fn,
        "Precision": tp / (tp + fp) if tp + fp > 0 else 0,
        "Recall": tp / (tp + fn) if tp + fn > 0 else 0,
        "F1 score": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0,
        "Accuracy": (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0,
        "Specificity": tn / (tn + fp) if tn + fp > 0 else 0,
        "Sensitivity": tp / (tp + fn) if tp + fn > 0 else 0,
        "MCC": (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
        "FPR": fp / (fp + tn),
        "FNR": fn / (tp + fn),
        "FDR": fp / (tp + fp),
        "FOR": fn / (tn + fn),
        "NPV": tn / (tn + fp),
        "PLR": tp / (fn + tp),
        "NLR": fn / (tp + fn)
    }
"""


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class MultiLabelConfusionMatrix(Metric):
    """Calculates a confusion matrix for multi-labelled, multi-class data."""
    def __init__(
            self,
            num_classes: int,
            output_transform: Callable = lambda x: x,
            device: Union[str, torch.device] = torch.device("cpu"),
            normalized: bool = False,
    ):
        if num_classes <= 1:
            raise ValueError("Argument num_classes needs to be > 1")

        self.num_classes = num_classes
        self._num_examples = 0
        self.normalized = normalized
        super(MultiLabelConfusionMatrix, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self.confusion_matrix = torch.zeros(self.num_classes, 2, 2, dtype=torch.int64, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_input(output)
        y_pred, y = output[0].detach(), output[1].detach()
        y_pred, y = y_pred.to(torch.int64), y.to(torch.int64)
        self._num_examples += y.shape[0]
        y_reshaped = y.transpose(0, 1).reshape(self.num_classes, -1)
        y_pred_reshaped = y_pred.transpose(0, 1).reshape(self.num_classes, -1)

        y_total = y_reshaped.sum(dim=1)
        y_pred_total = y_pred_reshaped.sum(dim=1)

        tp = (y_reshaped * y_pred_reshaped).sum(dim=1)
        fp = y_pred_total - tp
        fn = y_total - tp
        tn = y_reshaped.shape[1] - tp - fp - fn

        self.confusion_matrix += torch.stack([tn, fp, fn, tp], dim=1).reshape(-1, 2, 2).to(self._device)

    @sync_all_reduce("confusion_matrix", "_num_examples")
    def compute(self) -> torch.Tensor:
        if self._num_examples == 0:
            raise NotComputableError("Confusion matrix must have at least one example before it can be computed.")

        if self.normalized:
            conf = self.confusion_matrix.to(dtype=torch.float64)
            sums = conf.sum(dim=(1, 2))
            return conf / sums[:, None, None]

        return self.confusion_matrix

    def _check_input(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()
        y_pred, y = y_pred.to(torch.int64), y.to(torch.int64)

        if y_pred.ndimension() < 2:
            raise ValueError(
                f"y_pred must at least have shape (batch_size, num_classes (currently set to {self.num_classes}), ...)"
            )

        if y.ndimension() < 2:
            raise ValueError(
                f"y must at least have shape (batch_size, num_classes (currently set to {self.num_classes}), ...)"
            )

        if y_pred.shape[0] != y.shape[0]:
            raise ValueError(f"y_pred and y have different batch size: {y_pred.shape[0]} vs {y.shape[0]}")

        if y_pred.shape[1] != self.num_classes:
            raise ValueError(f"y_pred does not have correct number of classes: {y_pred.shape[1]} vs {self.num_classes}")

        if y.shape[1] != self.num_classes:
            raise ValueError(f"y does not have correct number of classes: {y.shape[1]} vs {self.num_classes}")

        if y.shape != y_pred.shape:
            raise ValueError("y and y_pred shapes must match.")

        valid_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if y_pred.dtype not in valid_types:
            raise ValueError(f"y_pred must be of any type: {valid_types}")

        if y.dtype not in valid_types:
            raise ValueError(f"y must be of any type: {valid_types}")

        if not torch.equal(y_pred, y_pred ** 2):
            raise ValueError("y_pred must be a binary tensor")

        if not torch.equal(y, y ** 2):
            raise ValueError("y must be a binary tensor")


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """SHA256 hash of a dictionary."""
    dhash = hashlib.sha256()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()
