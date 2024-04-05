from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss


@dataclass(frozen=True, kw_only=True)
class SingleOutcomeEvalMetrics:
    roc_auc: float
    log_loss: float


@dataclass(frozen=True, kw_only=True)
class EvalMetrics:
    epoch: int
    metrics: dict[str, SingleOutcomeEvalMetrics]


def evaluate_model(model, x: pd.DataFrame, y: pd.DataFrame, epoch):
    y_gt = y.to_dict(orient="list")
    yhat = model.forward(x)
    yhat = yhat.detach().numpy()
    yhat = {name: yhat[:, i] for i, name in enumerate(model.outputs)}

    all_na = []
    for k in y_gt.keys():
        na_idx = np.isnan(y_gt[k])
        y_gt[k] = np.array(y_gt[k])[~na_idx]
        yhat[k] = yhat[k][~na_idx]
        if na_idx.all():
            all_na.append(k)
    for k in all_na:
        del y_gt[k]
        del yhat[k]

    return EvalMetrics(
        epoch=epoch,
        metrics={
            k: SingleOutcomeEvalMetrics(
                roc_auc=roc_auc_score(y_gt[k], yhat[k]),
                log_loss=log_loss(y_gt[k], yhat[k]),
            )
            for k in y_gt.keys()
        },
    )
