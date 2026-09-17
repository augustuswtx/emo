"""Pure numerical helpers for baseline-defined modality-utility audits."""

import numpy as np


def thirds_by_baseline_utility(utility, indices):
    """Return stable equal-count strata without using P4 predictions."""
    utility = np.asarray(utility, dtype=np.float64).reshape(-1)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if utility.size < 3 or utility.size != indices.size:
        raise ValueError('at least three aligned samples are required')
    if not np.isfinite(utility).all() or np.unique(indices).size != indices.size:
        raise ValueError('utility must be finite and sample indices unique')
    order = np.lexsort((indices, utility))
    groups = np.empty(utility.size, dtype=np.int8)
    chunks = np.array_split(order, 3)
    for group, members in enumerate(chunks):
        groups[members] = group
    boundary_ties = [
        bool(utility[left[-1]] == utility[right[0]])
        for left, right in zip(chunks[:-1], chunks[1:])
    ]
    return groups, boundary_ties


def regression_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if labels.size != predictions.size or labels.size < 2:
        raise ValueError('at least two aligned labels and predictions are required')
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError('labels and predictions must be finite')
    error = predictions - labels
    corr = (
        float(np.corrcoef(labels, predictions)[0, 1])
        if np.std(labels) > 0 and np.std(predictions) > 0 else None
    )
    return {
        'MAE': float(np.mean(np.abs(error))),
        'Corr': corr,
        'Loss': float(np.mean(error ** 2)),
    }
