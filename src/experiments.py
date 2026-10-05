"""Experiments E1-E5. Every function returns a tidy DataFrame (one row per model/setting)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (KFold, StratifiedGroupKFold, StratifiedKFold, cross_val_predict)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import metrics as M
from . import models as Z

warnings.filterwarnings("ignore", category=UserWarning)


def _oof_proba(model, X, y, cv, groups=None):
    return cross_val_predict(clone(model), X, y, cv=cv, groups=groups, method="predict_proba", n_jobs=1)[:, 1]


# ------------------------------------------------------------------ E1 / E2 / E3
def cross_sectional(df: pd.DataFrame, features: list[str], target: str, cfg, label: str,
                    tune=False, fast=False) -> pd.DataFrame:
    """Repeated stratified CV for every classifier + the WHO rule (no training).

    target: 'status_raw' (original label, as used by prior studies) or 'stunted_who'.
    """
    d = df.dropna(subset=features + [target]).reset_index(drop=True)
    X, y = d[features].values, d[target].astype(int).values
    rule_pred = d["stunted_who"].values
    rule_score = -d["haz"].values  # lower HAZ = higher risk
    rows = []

    r = M.classification_metrics(y, rule_score, rule_pred)
    lo, hi = M.bootstrap_ci(y, rule_score, rule_pred, cfg.N_BOOTSTRAP, cfg.SEED)
    rows.append({"experiment": label, "features": "+".join(features), "target": target, "n": len(d),
                 "prevalence": y.mean(), "model": "WHO rule (no training)", **r,
                 "auroc_ci_low": lo, "auroc_ci_high": hi, "kappa_vs_rule": 1.0,
                 "delong_p_vs_rule": np.nan, "mcnemar_p_vs_rule": np.nan})

    for name, model in Z.classifiers(cfg.SEED, tune, fast).items():
        reps, kap = [], []
        first = None
        for rep in range(cfg.N_REPEATS):
            cv = StratifiedKFold(cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED + rep)
            prob = _oof_proba(model, X, y, cv)
            pred = (prob >= 0.5).astype(int)
            reps.append(M.classification_metrics(y, prob, pred, prob))
            kap.append(M.kappa(pred, rule_pred))
            if first is None:
                first = (prob, pred)
        mean = pd.DataFrame(reps).mean().to_dict()
        prob, pred = first
        lo, hi = M.bootstrap_ci(y, prob, pred, cfg.N_BOOTSTRAP, cfg.SEED)
        _, _, p_delong = M.delong_test(y, prob, rule_score)
        rows.append({"experiment": label, "features": "+".join(features), "target": target, "n": len(d),
                     "prevalence": y.mean(), "model": name, **mean,
                     "auroc_sd_repeats": pd.DataFrame(reps)["auroc"].std(),
                     "auroc_ci_low": lo, "auroc_ci_high": hi, "kappa_vs_rule": float(np.mean(kap)),
                     "delong_p_vs_rule": p_delong, "mcnemar_p_vs_rule": M.mcnemar(y, pred, rule_pred)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E4
def prospective(pairs: pd.DataFrame, cfg, fast=False) -> pd.DataFrame:
    """pairs: one row per matched child with *_feb and *_aug columns."""
    feats = [c for c in ["age_month_feb", "sex", "height_feb", "weight_feb", "haz_feb", "waz_feb"]
             if c in pairs and pairs[c].notna().all()]
    rows = []
    # --- regression: August HAZ
    d = pairs.dropna(subset=feats + ["haz_aug"]).reset_index(drop=True)
    X, y = d[feats].values, d["haz_aug"].values
    base = {"Persistence (HAZ_Aug = HAZ_Feb)": d["haz_feb"].values}
    for name, pred in base.items():
        rows.append({"experiment": "E4", "task": "regression: HAZ August", "n": len(d), "model": name,
                     "mae": mean_absolute_error(y, pred), "rmse": mean_squared_error(y, pred) ** 0.5,
                     "r2": r2_score(y, pred)})
    models = {"Linear on HAZ_Feb only (regression to the mean)": (Z.regressors(cfg.SEED, fast)["Ridge"], ["haz_feb"])}
    models.update({k: (v, feats) for k, v in Z.regressors(cfg.SEED, fast).items()})
    for name, (model, cols) in models.items():
        Xm = d[cols].values
        maes, rmses, r2s = [], [], []
        for rep in range(cfg.N_REPEATS):
            cv = KFold(cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED + rep)
            pred = cross_val_predict(clone(model), Xm, y, cv=cv)
            maes.append(mean_absolute_error(y, pred)); rmses.append(mean_squared_error(y, pred) ** 0.5)
            r2s.append(r2_score(y, pred))
        rows.append({"experiment": "E4", "task": "regression: HAZ August", "n": len(d), "model": name,
                     "mae": np.mean(maes), "rmse": np.mean(rmses), "r2": np.mean(r2s)})

    # --- classification: stunted in August (all children, and incident = not stunted in Feb)
    for subset, mask in [("all children", np.ones(len(d), bool)), ("incident (not stunted in Feb)", d["stunted_feb"].values == 0)]:
        s = d[mask].reset_index(drop=True)
        ys = s["stunted_aug"].astype(int).values
        minority = int(min(ys.sum(), len(ys) - ys.sum()))
        if minority < 5:
            rows.append({"experiment": "E4", "task": f"classification: {subset}", "n": len(s),
                         "model": "SKIPPED (fewer than 5 cases in a class)"})
            continue
        k = min(cfg.N_SPLITS, minority)
        score_p, pred_p = -s["haz_feb"].values, s["stunted_feb"].astype(int).values
        rows.append({"experiment": "E4", "task": f"classification: {subset}", "n": len(s), "events": int(ys.sum()),
                     "model": "Persistence (status_Aug = status_Feb)", **M.classification_metrics(ys, score_p, pred_p)})
        for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
            model = Z.classifiers(cfg.SEED, fast=fast)[name]
            reps = []
            for rep in range(cfg.N_REPEATS):
                cv = StratifiedKFold(k, shuffle=True, random_state=cfg.SEED + rep)
                prob = _oof_proba(model, s[feats].values, ys, cv)
                reps.append(M.classification_metrics(ys, prob, (prob >= 0.5).astype(int), prob))
            rows.append({"experiment": "E4", "task": f"classification: {subset}", "n": len(s), "events": int(ys.sum()),
                         "model": name, **pd.DataFrame(reps).mean().to_dict()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E5
def _e5_models(seed, fast):
    n = 100 if fast else 300
    return {"KNN (k=3)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
            "Random Forest": RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=seed)}


def group_leakage(df: pd.DataFrame, features: list[str], target: str, cfg, fast=False) -> pd.DataFrame:
    """Random row-level split vs split by child (same child never in train and test)."""
    d = df.dropna(subset=features + [target, "child_id"]).reset_index(drop=True)
    if d["child_id"].duplicated().sum() == 0:
        return pd.DataFrame([{"experiment": "E5a", "note": "No child has repeated records; group leakage cannot occur."}])
    X, y, g = d[features].values, d[target].astype(int).values, d["child_id"].values
    rows = []
    for name, model in _e5_models(cfg.SEED, fast).items():
        for split_name, cv, groups in [
            ("random row split", StratifiedKFold(cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED), None),
            ("child-level split", StratifiedGroupKFold(cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED), g)]:
            prob = _oof_proba(model, X, y, cv, groups)
            pred = (prob >= 0.5).astype(int)
            rows.append({"experiment": "E5a", "features": "+".join(features), "model": name, "split": split_name,
                         "n_records": len(d), "n_children": d["child_id"].nunique(),
                         **M.classification_metrics(y, prob, pred, prob)})
    return pd.DataFrame(rows)


def smote_leakage(df: pd.DataFrame, features: list[str], target: str, cfg, fast=False) -> pd.DataFrame:
    """SMOTE applied to the full data before CV (leaky) vs SMOTE inside training folds only."""
    d = df.dropna(subset=features + [target]).reset_index(drop=True)
    X, y = d[features].values, d[target].astype(int).values
    rows = []
    for name, model in _e5_models(cfg.SEED, fast).items():
        cv = StratifiedKFold(cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED)
        # leaky: oversample everything, then split (synthetic neighbours of test points end up in training)
        Xr, yr = SMOTE(random_state=cfg.SEED).fit_resample(X, y)
        prob = _oof_proba(model, Xr, yr, cv)
        rows.append({"experiment": "E5b", "features": "+".join(features), "model": name,
                     "pipeline": "SMOTE before split (leaky)", "n_evaluated": len(yr),
                     **M.classification_metrics(yr, prob, (prob >= 0.5).astype(int), prob)})
        # correct: oversample only the training part of each fold, evaluate on real data
        pipe = ImbPipeline([("smote", SMOTE(random_state=cfg.SEED)), ("clf", clone(model))])
        prob = _oof_proba(pipe, X, y, cv)
        rows.append({"experiment": "E5b", "features": "+".join(features), "model": name,
                     "pipeline": "SMOTE inside training folds", "n_evaluated": len(y),
                     **M.classification_metrics(y, prob, (prob >= 0.5).astype(int), prob)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Overfitting check
def overfitting_check(df: pd.DataFrame, features: list[str], target: str, cfg, label: str) -> pd.DataFrame:
    """Compare training-fold and held-out-fold performance for each model.

    A large train-test gap indicates overfitting. All scores reported elsewhere in the paper
    are held-out (out-of-fold) scores, so overfitting cannot inflate them; this check shows
    how large the gap is. The WHO rule has no fitted parameters and therefore no gap.
    """
    from sklearn.model_selection import cross_validate
    d = df.dropna(subset=features + [target]).reset_index(drop=True)
    X, y = d[features].values, d[target].astype(int).values
    cv = StratifiedKFold(cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED)
    rows = [{"experiment": label, "features": "+".join(features), "model": "WHO rule (no training)",
             "train_auroc": np.nan, "test_auroc": M.roc_auc_score(y, -d["haz"].values),
             "train_f1": np.nan, "test_f1": M.f1_score(y, d["stunted_who"].values), "n": len(d)}]
    for name, model in Z.overfit_models(cfg.SEED).items():
        r = cross_validate(clone(model), X, y, cv=cv, scoring=["roc_auc", "f1"], return_train_score=True, n_jobs=1)
        rows.append({"experiment": label, "features": "+".join(features), "model": name,
                     "train_auroc": r["train_roc_auc"].mean(), "test_auroc": r["test_roc_auc"].mean(),
                     "train_f1": r["train_f1"].mean(), "test_f1": r["test_f1"].mean(),
                     "test_auroc_sd": r["test_roc_auc"].std(), "n": len(d)})
    out = pd.DataFrame(rows)
    out["auroc_gap"] = out["train_auroc"] - out["test_auroc"]
    out["f1_gap"] = out["train_f1"] - out["test_f1"]
    return out
