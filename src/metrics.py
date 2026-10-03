"""Evaluation metrics and statistical tests."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, f1_score, roc_auc_score)


def classification_metrics(y, score, pred, prob=None) -> dict:
    """score: any monotone risk score (for AUROC/AUPRC); prob: calibrated probability (for Brier)."""
    out = {
        "auroc": roc_auc_score(y, score),
        "auprc": average_precision_score(y, score),
        "f1": f1_score(y, pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "accuracy": float(np.mean(np.asarray(y) == np.asarray(pred))),
    }
    out["brier"] = brier_score_loss(y, prob) if prob is not None else np.nan
    return out


def bootstrap_ci(y, score, pred, n_boot=1000, seed=42, metric="auroc"):
    rng = np.random.default_rng(seed)
    y, score, pred = map(np.asarray, (y, score, pred))
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(roc_auc_score(y[i], score[i]) if metric == "auroc" else f1_score(y[i], pred[i], zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def kappa(a, b) -> float:
    return float(cohen_kappa_score(a, b))


def mcnemar(y, pred_a, pred_b) -> float:
    """Exact/chi2 McNemar p-value comparing the error patterns of two classifiers."""
    y, pa, pb = map(np.asarray, (y, pred_a, pred_b))
    a_right, b_right = pa == y, pb == y
    n01 = int(np.sum(a_right & ~b_right))
    n10 = int(np.sum(~a_right & b_right))
    if n01 + n10 == 0:
        return 1.0
    if n01 + n10 < 25:
        return float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(stats.chi2.sf(chi2, 1))


# ---- Fast DeLong (Sun & Xu, 2014) for comparing two correlated AUCs ----
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong_test(y, score_a, score_b) -> tuple[float, float, float]:
    """Returns (auc_a, auc_b, two-sided p-value)."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y)  # positives first
    preds = np.vstack([np.asarray(score_a, float), np.asarray(score_b, float)])[:, order]
    m = int(y.sum())
    n = len(y) - m
    tx = np.array([_midrank(p[:m]) for p in preds])
    ty = np.array([_midrank(p[m:]) for p in preds])
    tz = np.array([_midrank(p) for p in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(2 * stats.norm.sf(abs(z)))
