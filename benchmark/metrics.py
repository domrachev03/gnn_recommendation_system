import numpy as np


def dcg_k(score_label, k):
    dcg, i = 0., 0
    for s in score_label:
        if i < k:
            dcg += (2**s[1]-1) / np.log2(2+i)
            i += 1
    return dcg


def ndcg_k(preds, labels, k):
    score_label = np.stack([preds, labels], axis=1).tolist()
    score_label = sorted(score_label, key=lambda d: d[0], reverse=True)
    score_label_ = sorted(score_label, key=lambda d: d[1], reverse=True)
    norm, i = 0., 0
    for s in score_label_:
        if i < k:
            norm += (2**s[1]-1) / np.log2(2+i)
            i += 1
    dcg = dcg_k(score_label, k)
    return dcg / norm


def ndcg(preds, labels):
    ndcg_sum, num = 0, 0
    preds, labels = preds.T, labels.T
    n_users = labels.shape[0]

    for i in range(n_users):
        preds_i = preds[i][np.where(labels[i])]
        labels_i = labels[i][np.where(labels[i])]

        if labels_i.shape[0] < 2:
            continue

        ndcg_sum += ndcg_k(preds_i, labels_i, labels_i.shape[0])  # user-wise calculation
        num += 1

    return ndcg_sum / num


def rmse(preds: np.ndarray, labels: np.ndarray, mask: np.ndarray | None = None):
    '''Root Mean Square Error metric with optional boolean mask

    Parameters:
        preds: predictions of the model
        labels: true labels
        mask (optional): boolean mask whether include the value in the evaluation or not
    '''

    if mask is not None:
        return np.sqrt(
            (mask * (np.clip(preds, 1., 5.) - labels) ** 2).sum() / mask.sum()
        )
    else:
        n_elems = preds.size
        return np.sqrt(
            ((np.clip(preds, 1., 5.) - labels) ** 2).sum() / n_elems
        )


def mae(preds: np.ndarray, labels: np.ndarray, mask: np.ndarray | None = None):
    '''Mean Absolute Error metric with optional boolean mask

    Parameters:
        preds: predictions of the model
        labels: true labels
        mask (optional): boolean mask whether include the value in the evaluation or not
    '''
    if mask is not None:
        return np.sqrt(
            (mask * np.abs(np.clip(preds, 1., 5.) - labels)).sum() / mask.sum()
        )
    else:
        n_elems = preds.size
        return np.sqrt(
            np.abs(np.clip(preds, 1., 5.) - labels).sum() / n_elems
        )


def get_metrics_names():
    '''Retrieving information about metrics: the names and if the bigger the better'''

    return (('rmse', False), ('mae', False), ('ndcg', True))


def get_metrics(preds: np.ndarray, labels: np.ndarray, mask: np.ndarray | None = None):
    '''Combine data from all metrics in the dictionary

    Parameters:
        preds: predictions of the model
        labels: true labels
        mask (optional): boolean mask whether include the value in the evaluation or not
    '''

    return {
        'rmse': rmse(preds, labels, mask),
        'mae': mae(preds, labels, mask),
        'ndcg': ndcg(preds, labels)
    }