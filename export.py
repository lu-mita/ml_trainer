import os
import pandas as pd

from .utils import compute_metrics_per_label

def export_metrics_mlcm(mlcm, save_path: str):
    """
    Export metrics from a MultiLabelConfusionMatrix object to a csv file and excel file.
    """
    os.makedirs(save_path, exist_ok=True)
    labels = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Love', 'Neutral', 'Sadness', 'Surprise', 'Trust']
    df_metrics = pd.DataFrame([{"label": label, **compute_metrics_per_label(i)} for i, label in zip(mlcm, labels)])
    df_metrics.to_csv(os.path.join(save_path, "confusion_matrices.csv"))
    df_metrics.to_excel(os.path.join(save_path, "confusion_matrices.xlsx"))