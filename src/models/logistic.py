from pathlib import Path
import numpy as np
import optuna
import yaml
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

from src.utils.vectorizer import get_vectorizer


def train_logistic(cfg, df):
    X = df[cfg.dataset.text_columns] if isinstance(cfg.dataset.text_columns, str) else df[cfg.dataset.text_columns].astype(str).agg(" ".join, axis=1)
    y = df[cfg.dataset.label_column]
    skf = StratifiedKFold(n_splits=cfg.run.folds, shuffle=True, random_state=cfg.run.seed)

    def objective(trial):
        max_features = trial.suggest_int("max_features", 5000, 20000, step=5000)
        C = trial.suggest_float("C", 0.01, 10.0, log=True)
        penalty = trial.suggest_categorical("penalty", ["l2"])
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
        max_iter = trial.suggest_int("max_iter", 100, 1000, step=100)
        vectorizer = trial.suggest_categorical("vectorizer", ["tfidf", "count"])

        auc_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            vec = get_vectorizer(vectorizer, max_features)
            X_train_vec = vec.fit_transform(X_train)
            X_val_vec = vec.transform(X_val)

            model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter)
            model.fit(X_train_vec, y_train)

            y_proba = model.predict_proba(X_val_vec)[:, 1]
            auc = roc_auc_score(y_val, y_proba)

            # Optional: print both AUC and F1
            y_pred = model.predict(X_val_vec)
            f1 = f1_score(y_val, y_pred)
            print(f"Fold {fold+1} AUC: {auc:.4f} | F1: {f1:.4f}")

            auc_scores.append(auc)

        return np.mean(auc_scores)

    # Run Optuna tuning
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params
    print("\n✅ Best hyperparameters:")
    print(best_params)
    print(f"🎯 Best CV AUC: {study.best_value:.4f}")

    # Save best hyperparameters to config/model/logistic.yaml
    model_yaml_path = Path("config/model/logistic.yaml")
    model_config = {
        "name": "logistic_regression",
        "type": "sklearn",
        "classifier": "logistic",
        **best_params
    }
    model_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with model_yaml_path.open("w") as f:
        yaml.dump(model_config, f)

    # Save best CV AUC score to model_comparison.csv
    comparison_path = Path(cfg.paths.output_dir) / "model_comparison.csv"
    cv_row = pd.DataFrame([{
        "model": "logistic",
        "accuracy": None,
        "f1_score": None,
        "cv_auc": study.best_value
    }])
    if comparison_path.exists():
        existing = pd.read_csv(comparison_path)
        existing = existing[existing["model"] != "logistic"]
        updated = pd.concat([existing, cv_row], ignore_index=True)
    else:
        updated = cv_row
    updated.to_csv(comparison_path, index=False)
    print(f"📄 CV AUC score logged to {comparison_path}")

    # Train final model on full data with best parameters
    vec = get_vectorizer(best_params['vectorizer'], best_params['max_features'])
    X_vec = vec.fit_transform(X)
    model = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'],
                               solver=best_params['solver'], max_iter=best_params['max_iter'])
    model.fit(X_vec, y)

    dump(vec, f"{cfg.paths.output_dir}/logistic_vectorizer.joblib")
    dump(model, f"{cfg.paths.output_dir}/logistic_model.joblib")
    print("✅ Final model and vectorizer saved.")
