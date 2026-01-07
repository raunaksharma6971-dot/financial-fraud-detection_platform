import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA_PATH = Path("data/creditcard.csv")
MODEL_PATH = Path("models/xgb_model.joblib")
METRICS_PATH = Path("models/metrics.json")


def find_threshold_for_precision(y_true, y_proba, min_precision=0.90):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    best = None
    for i in range(1, len(thresholds)):
        p = float(precision[i])
        r = float(recall[i])
        t = float(thresholds[i - 1])
        if p >= min_precision:
            if best is None or r > best["recall"]:
                best = {"threshold": t, "precision": p, "recall": r}
    return best


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Missing data/creditcard.csv. Upload it into the data/ folder.")

    df = pd.read_csv(DATA_PATH)

    # Sanity check: correct dataset signature
    must_have = {"Time", "Amount", "Class"}
    if not must_have.issubset(df.columns):
        raise ValueError(f"Wrong dataset. Expected at least columns {must_have}.")

    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])

    # Stratified split (fraud is rare)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = float(neg / max(pos, 1))

    model = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1,
        objective="binary:logistic",
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    pr_auc = float(average_precision_score(y_test, y_proba))
    roc_auc = float(roc_auc_score(y_test, y_proba))

    # Choose a practical threshold: maximize recall while keeping precision high
    best = find_threshold_for_precision(y_test.values, y_proba, min_precision=0.90)

    # Fallback: best F1 threshold if precision constraint can't be met
    if best is None:
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        f1 = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        best_idx = int(np.nanargmax(f1))
        best = {
            "threshold": float(thresholds[best_idx]),
            "precision": float(precision[best_idx + 1]),
            "recall": float(recall[best_idx + 1]),
        }

    y_pred = (y_proba >= best["threshold"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Save artifacts
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metrics = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "fraud_count": int((y == 1).sum()),
        "nonfraud_count": int((y == 0).sum()),
        "test_pr_auc": pr_auc,
        "test_roc_auc": roc_auc,
        "chosen_threshold": best["threshold"],
        "precision_at_threshold": best["precision"],
        "recall_at_threshold": best["recall"],
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "scale_pos_weight": scale_pos_weight,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Training complete")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model:   {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
