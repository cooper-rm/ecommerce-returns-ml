import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score


def oof(model, X, y, cv=5, seed=42, groups=None):
    """Out-of-fold probabilities. Pass ``groups`` (e.g. customer id) to keep a customer's orders in one fold."""
    folds = GroupKFold(cv) if groups is not None else StratifiedKFold(cv, shuffle=True, random_state=seed)
    return cross_val_predict(model, X, y, cv=folds, groups=groups, method="predict_proba")[:, 1]


def time_split(df, date_col, cutoff):
    """Boolean train/valid masks split on ``cutoff``: train is strictly before, valid is on or after."""
    d = pd.to_datetime(df[date_col])
    return d < pd.Timestamp(cutoff), d >= pd.Timestamp(cutoff)


def cv_auc(model, X, y, cv=5, groups=None):
    a = roc_auc_score(y, oof(model, X, y, cv=cv, groups=groups))
    print(f"OOF AUC: {a:.4f}")
    return a
