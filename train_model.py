import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
MODEL_DIR = BASE_DIR / "models"

COUNTRY_MAPPING = {
    "Viet Nam": "Vietnam",
    "AmericanSamoa": "United States",
    "Hong Kong": "China",
}

RELATION_MAPPING = {
    "Relative": "others",
    "Parent": "others",
    "?": "others",
    "Others": "others",
    "Health care professional": "others",
}


def cap_outliers(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    median = data[column].median()
    data[column] = data[column].apply(
        lambda value: median if value < lower_bound or value > upper_bound else value
    )
    return data


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(value) for value in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    dataset = pd.read_csv(DATA_PATH)
    dataset["age"] = dataset["age"].astype(int)
    dataset = dataset.drop(columns=["ID", "age_desc"])
    dataset["contry_of_res"] = dataset["contry_of_res"].replace(COUNTRY_MAPPING)
    dataset["ethnicity"] = dataset["ethnicity"].replace({"?": "others", "Others": "others"})
    dataset["relation"] = dataset["relation"].replace(RELATION_MAPPING)
    dataset = cap_outliers(dataset, "age")
    dataset = cap_outliers(dataset, "result")

    country_options = sorted(dataset["contry_of_res"].unique().tolist())
    ethnicity_options = sorted(dataset["ethnicity"].unique().tolist())
    relation_options = sorted(dataset["relation"].unique().tolist())
    gender_options = sorted(dataset["gender"].unique().tolist())

    object_columns = dataset.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}
    for column in object_columns:
        encoder = LabelEncoder()
        dataset[column] = encoder.fit_transform(dataset[column])
        encoders[column] = encoder

    features = [column for column in dataset.columns if column != "Class/ASD"]
    X = dataset[features]
    y = dataset["Class/ASD"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    smote = SMOTE(random_state=123)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    param_grids = {
        "DecisionTree": {
            "criterion": ["entropy", "gini"],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_leaf": [1, 2, 3, 4],
            "min_samples_split": [2, 4, 6],
        },
        "RandomForest": {
            "n_estimators": [50, 100, 200, 400],
            "max_depth": [4, 7, 10, 20],
            "min_samples_leaf": [1, 2, 3, 4],
            "min_samples_split": [2, 4, 6],
            "bootstrap": [True, False],
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200, 400],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.1, 0.01, 0.2, 0.3],
            "subsample": [0.2, 0.55, 0.75, 1.0],
            "colsample_bytree": [0.5, 0.6, 0.7],
        },
    }

    estimators = {
        "DecisionTree": DecisionTreeClassifier(random_state=123),
        "RandomForest": RandomForestClassifier(random_state=123),
        "XGBoost": XGBClassifier(random_state=123, eval_metric="logloss"),
    }

    report = {}
    best_name = None
    best_model = None
    best_cv = -1.0

    for name, estimator in estimators.items():
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grids[name],
            n_iter=10,
            cv=5,
            scoring="accuracy",
            random_state=32,
            n_jobs=-1,
        )
        search.fit(X_train_balanced, y_train_balanced)
        tuned = search.best_estimator_
        cv_accuracy = float(search.best_score_)
        predictions = tuned.predict(X_test)
        test_accuracy = float(accuracy_score(y_test, predictions))
        report[name] = {
            "cv_accuracy": round(cv_accuracy, 4),
            "test_accuracy": round(test_accuracy, 4),
            "best_params": search.best_params_,
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                target_names=["No ASD", "ASD"],
                zero_division=0,
            ),
        }
        print(f"{name}: cv={cv_accuracy:.4f} test={test_accuracy:.4f}")
        if cv_accuracy > best_cv:
            best_name = name
            best_model = tuned
            best_cv = cv_accuracy

    with open(MODEL_DIR / "best_model.pkl", "wb") as handle:
        pickle.dump(best_model, handle)
    with open(MODEL_DIR / "encoders.pkl", "wb") as handle:
        pickle.dump(encoders, handle)

    meta = {
        "features": features,
        "target": "Class/ASD",
        "best_model": best_name,
        "best_cv_accuracy": round(best_cv, 4),
        "options": {
            "gender": gender_options,
            "ethnicity": ethnicity_options,
            "jaundice": ["no", "yes"],
            "austim": ["no", "yes"],
            "contry_of_res": country_options,
            "used_app_before": ["no", "yes"],
            "relation": relation_options,
        },
        "stats": {
            "rows": int(dataset.shape[0]),
            "asd_positive": int(y.sum()),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "balanced_rows": int(len(X_train_balanced)),
        },
    }
    with open(MODEL_DIR / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(make_json_safe(report), handle, indent=2)

    print(f"\nBest model: {best_name} (cv accuracy {best_cv:.4f})")
    print(f"Artifacts saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
