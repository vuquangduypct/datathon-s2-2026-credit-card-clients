"""Analyze credit-card customers and predict default probability."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "default"
ID_COLUMN = "client_id"
CATEGORICAL_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE"] + [
    f"PAY_{period}" for period in [0, 2, 3, 4, 5, 6]
]
# Best Latin-hypercube settings from credit_default_analysis.ipynb
NEURAL_NET_PARAMS = {
    "hidden_layer_sizes": (11, 11),
    "alpha": 0.005173693460665387,
    "learning_rate_init": 0.003764072612068487,
    "activation": "relu",
    "solver": "adam",
    "max_iter": 300,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
    "random_state": 42,
}


def add_customer_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable balance, utilization, and repayment indicators."""
    result = data.copy()
    bill_columns = [f"BILL_AMT{month}" for month in range(1, 7)]
    payment_columns = [f"PAY_AMT{month}" for month in range(1, 7)]
    status_columns = ["PAY_0"] + [f"PAY_{month}" for month in range(2, 7)]

    result["average_bill_balance"] = result[bill_columns].mean(axis=1)
    result["average_payment"] = result[payment_columns].mean(axis=1)
    result["total_bill_balance"] = result[bill_columns].sum(axis=1)
    result["total_payment"] = result[payment_columns].sum(axis=1)
    result["payment_to_bill_ratio"] = result["total_payment"] / (
        result["total_bill_balance"].abs() + 1
    )
    result["credit_utilization"] = result["average_bill_balance"] / (
        result["LIMIT_BAL"] + 1
    )
    result["maximum_delinquency"] = result[status_columns].max(axis=1)
    result["delinquent_months"] = (result[status_columns] > 0).sum(axis=1)
    return result


def make_pipeline(model: object, feature_columns: list[str]) -> Pipeline:
    categorical = [column for column in CATEGORICAL_COLUMNS if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    transformer = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )
    return Pipeline([("preprocess", transformer), ("model", model)])


def evaluate_model(name: str, model: Pipeline, x_valid: pd.DataFrame, y_valid: pd.Series) -> None:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_valid)[:, 1]
    else:
        probabilities = np.clip(model.predict(x_valid), 0, 1)
    predictions = (probabilities >= 0.5).astype(int)
    print(f"\n{name}")
    print(f"  ROC-AUC:             {roc_auc_score(y_valid, probabilities):.4f}")
    print(f"  Average precision:   {average_precision_score(y_valid, probabilities):.4f}")
    print(f"  Log loss:            {log_loss(y_valid, probabilities, labels=[0, 1]):.4f}")
    print(f"  Brier score:         {brier_score_loss(y_valid, probabilities):.4f}")
    print(f"  Accuracy at 0.5:     {accuracy_score(y_valid, predictions):.4f}")
    print(f"  Mean absolute error: {mean_absolute_error(y_valid, probabilities):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing train.csv and test.csv")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = add_customer_metrics(pd.read_csv(args.data_dir / "train.csv"))
    test = add_customer_metrics(pd.read_csv(args.data_dir / "test.csv"))

    print(f"Training customers: {len(train):,}")
    print(f"Default rate:       {train[TARGET].mean():.2%}")
    print("\nCustomer status summary:")
    print(
        train.groupby(TARGET)[
            ["average_bill_balance", "average_payment", "credit_utilization", "delinquent_months"]
        ]
        .mean()
        .round(2)
        .to_string()
    )

    feature_columns = [
        column for column in train.columns if column not in {TARGET, ID_COLUMN}
    ]
    x = train[feature_columns]
    y = train[TARGET]
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    linear_model = make_pipeline(LinearRegression(), feature_columns)
    logistic_model = make_pipeline(LogisticRegression(max_iter=2000), feature_columns)
    neural_net_model = make_pipeline(MLPClassifier(**NEURAL_NET_PARAMS), feature_columns)
    linear_model.fit(x_train, y_train)
    logistic_model.fit(x_train, y_train)
    neural_net_model.fit(x_train, y_train)
    evaluate_model("Linear probability regression", linear_model, x_valid, y_valid)
    evaluate_model("Logistic regression comparison", logistic_model, x_valid, y_valid)
    evaluate_model("Neural network (LHS-tuned)", neural_net_model, x_valid, y_valid)

    linear_model.fit(x, y)
    probabilities = np.clip(linear_model.predict(test[feature_columns]), 0, 1)
    submission = pd.DataFrame(
        {ID_COLUMN: test[ID_COLUMN], "default_probability": probabilities}
    )
    submission.to_csv(args.output, index=False)
    print(f"\nWrote {len(submission):,} predictions to {args.output}")


if __name__ == "__main__":
    main()