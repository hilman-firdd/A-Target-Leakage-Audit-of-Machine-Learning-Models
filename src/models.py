"""Model zoo. Defaults mirror the configurations commonly used in the replicated studies.

With tune=True, RF and XGBoost are wrapped in an inner 3-fold grid search, so the outer
cross-validation becomes nested cross-validation.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor


def classifiers(seed: int = 42, tune: bool = False, fast: bool = False) -> dict:
    n_trees = 100 if fast else 300
    rf = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, random_state=seed)
    xgb = XGBClassifier(n_estimators=n_trees, max_depth=6, learning_rate=0.1, subsample=0.9,
                        eval_metric="logloss", n_jobs=-1, random_state=seed, verbosity=0)
    if tune:
        inner = StratifiedKFold(3, shuffle=True, random_state=seed)
        rf = GridSearchCV(rf, {"max_depth": [None, 10], "min_samples_leaf": [1, 5]},
                          scoring="roc_auc", cv=inner, n_jobs=1)
        xgb = GridSearchCV(xgb, {"max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
                           scoring="roc_auc", cv=inner, n_jobs=1)
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Random Forest": rf,
        "XGBoost": xgb,
    }


def regressors(seed: int = 42, fast: bool = False) -> dict:
    n_trees = 100 if fast else 300
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=5, n_jobs=-1, random_state=seed),
        "XGBoost": XGBRegressor(n_estimators=n_trees, max_depth=3, learning_rate=0.05, n_jobs=-1,
                                random_state=seed, verbosity=0),
    }


def overfit_models(seed: int = 42) -> dict:
    """Models compared in the overfitting check (train vs. out-of-fold performance).

    'XGBoost (regularized)' uses shallow trees, row/column subsampling, a minimum child
    weight and an L2 penalty: the standard remedies for gradient-boosting overfitting.
    """
    return {
        "Decision Tree (unpruned)": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
        "XGBoost (default)": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
                                           eval_metric="logloss", n_jobs=-1, random_state=seed, verbosity=0),
        "XGBoost (regularized)": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                               colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0,
                                               eval_metric="logloss", n_jobs=-1, random_state=seed, verbosity=0),
    }
