import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from data_prep import CATEGORICAL_FEATURES, SEED, TARGET, split_train_test, time_cv_splits
from features import feature_list, load_features
from train_lgbm import N_ROUNDS, PARAMS, metrics

CAT_PARAMS = dict(iterations=1500, learning_rate=0.06, depth=7, l2_leaf_reg=5, verbose=0, thread_count=-1)
W_CAT = 0.3
LGB_SEEDS = [SEED, SEED + 1, SEED + 2]
CAT_SEEDS = [SEED]


def fit_lgb(X_tr, y_tr, X_pred, seeds):
    X_tr, X_pred = X_tr.copy(), X_pred.copy()
    for c in CATEGORICAL_FEATURES:
        dtype = pd.CategoricalDtype(sorted(set(X_tr[c].dropna()) | set(X_pred[c].dropna())))
        X_tr[c], X_pred[c] = X_tr[c].astype(dtype), X_pred[c].astype(dtype)
    return np.mean([lgb.train({**PARAMS, "seed": s}, lgb.Dataset(X_tr, y_tr), N_ROUNDS).predict(X_pred)
                    for s in seeds], axis=0)


def fit_cat(X_tr, y_tr, X_pred, seeds):
    X_tr, X_pred = X_tr.copy(), X_pred.copy()
    for c in CATEGORICAL_FEATURES:
        X_tr[c], X_pred[c] = X_tr[c].fillna("Missing").astype(str), X_pred[c].fillna("Missing").astype(str)
    return np.mean([CatBoostClassifier(**CAT_PARAMS, cat_features=CATEGORICAL_FEATURES, random_seed=s)
                    .fit(X_tr, y_tr).predict_proba(X_pred)[:, 1] for s in seeds], axis=0)


def fit_blend(X_tr, y_tr, X_pred, lgb_seeds, cat_seeds):
    p_lgb = fit_lgb(X_tr, y_tr, X_pred, lgb_seeds)
    p_cat = fit_cat(X_tr, y_tr, X_pred, cat_seeds)
    return (1 - W_CAT) * p_lgb + W_CAT * p_cat, p_lgb, p_cat


def main():
    df = load_features()
    feats = feature_list(df)
    train, _, test = split_train_test(df)
    X, y = train[feats], train[TARGET]
    Xt, yt = test[feats], test[TARGET]
    print(f"Ознак: {len(feats)} | train: {len(train):,} | test: {len(test):,}")

    oof_p, oof_y = [], []
    for k, (tr, va) in enumerate(time_cv_splits(train["Date"]), 1):
        p, p_lgb, p_cat = fit_blend(X.iloc[tr], y.iloc[tr], X.iloc[va], [SEED], [SEED])
        yv = y.iloc[va]
        print(f"fold {k}: blend AUC={roc_auc_score(yv, p):.4f}  lgb={roc_auc_score(yv, p_lgb):.4f}  "
              f"cat={roc_auc_score(yv, p_cat):.4f}  PR-AUC={average_precision_score(yv, p):.4f}", flush=True)
        oof_p.append(p)
        oof_y.append(yv.to_numpy())
    oof_p, oof_y = np.concatenate(oof_p), np.concatenate(oof_y)

    grid = np.arange(0.15, 0.61, 0.01)
    threshold = float(grid[np.argmax([f1_score(oof_y, oof_p > t) for t in grid])])
    print(f"Поріг з CV: {threshold:.2f}")

    p, p_lgb, p_cat = fit_blend(X, y, Xt, LGB_SEEDS, CAT_SEEDS)
    res = metrics(yt, p, threshold)
    base_f1 = f1_score(yt, test["RainToday"].fillna(0))
    print("TEST blend: " + "  ".join(f"{k}={v:.4f}" for k, v in res.items()) + f"  | baseline F1={base_f1:.3f}")
    print(f"TEST тільки LightGBM: ROC-AUC={roc_auc_score(yt, p_lgb):.4f}  тільки CatBoost: ROC-AUC={roc_auc_score(yt, p_cat):.4f}")

    with open("results_blend.json", "w", encoding="utf-8") as f:
        json.dump({**res, "threshold": threshold, "baseline_f1": base_f1, "n_features": len(feats)}, f, indent=2)
    pd.DataFrame({"Date": test["Date"], "Location": test["Location"], "y_true": yt, "proba": p,
                  "proba_lgb": p_lgb, "proba_cat": p_cat, "pred": (p > threshold).astype(int)}
                 ).to_csv("predictions_blend.csv", index=False)
    print("Збережено: results_blend.json, predictions_blend.csv")


if __name__ == "__main__":
    main()
