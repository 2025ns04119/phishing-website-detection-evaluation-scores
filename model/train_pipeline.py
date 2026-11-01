"""Train the 5 classifiers on the phishing-website data and save them.

Run from anywhere:  python model/train_pipeline.py
Self-check only:    python model/train_pipeline.py --test
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
TARGET = "class"          # 1 = phishing, 0 = legitimate
SEED = 2025


def data_path(name):
    """Training data lives in model/ (keeps the repo root as the assignment requires)."""
    inside = MODEL_DIR / name
    return inside if inside.exists() else ROOT / name


def build_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=SEED),
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "Naive Bayes": GaussianNB(),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=SEED)),
    }


def model_path(name):
    return MODEL_DIR / (name.replace(" ", "_").lower() + ".pkl")


def evaluate(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def train_and_save():
    train = pd.read_csv(data_path("train_data.csv"))
    test = pd.read_csv(ROOT / "test_data.csv")
    X_tr, y_tr = train.drop(columns=TARGET), train[TARGET]
    X_te, y_te = test.drop(columns=TARGET), test[TARGET]

    rows = []
    for name, model in build_models().items():
        model.fit(X_tr, y_tr)
        scores = evaluate(y_te, model.predict(X_te), model.predict_proba(X_te)[:, 1])
        rows.append({"Model": name, **scores})
        joblib.dump(model, model_path(name))
    results = pd.DataFrame(rows).round(4)
    results.to_csv(MODEL_DIR / "metrics_summary.csv", index=False)
    return results


def self_test():
    """Simple checks on the training data, saved models and metrics (test data is not touched)."""
    train = pd.read_csv(data_path("train_data.csv"))
    assert train.shape[1] - 1 >= 12, "need at least 12 features"
    assert len(train) >= 500, "need at least 500 rows"
    assert set(train[TARGET].unique()) == {0, 1}, "target must be 0/1"
    assert train.isna().sum().sum() == 0, "nulls found"
    assert not train.duplicated().any(), "duplicate rows in training data"

    results = pd.read_csv(MODEL_DIR / "metrics_summary.csv")
    assert len(results) == 5, "expected 5 models"
    vals = results.drop(columns="Model")
    assert ((vals >= -1) & (vals <= 1)).all().all(), "metric out of range"
    for name in build_models():
        assert model_path(name).exists(), f"missing saved model: {name}"
    print("train_pipeline self-test: all checks passed")


if __name__ == "__main__":
    if "--test" in sys.argv:
        self_test()
    else:
        print(train_and_save().to_string(index=False))
        self_test()
