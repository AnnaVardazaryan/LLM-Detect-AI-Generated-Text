import numpy as np
import optuna
import yaml
import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

from src.utils.vectorizer import get_vectorizer

def train_naive_bayes(cfg, df):
    X = df[cfg.dataset.text_columns] if isinstance(cfg.dataset.text_columns, str) else df[cfg.dataset.text_columns].astype(str).agg(" ".join, axis=1)
    y = df[cfg.dataset.label_column]
    skf = StratifiedKFold(n_splits=cfg.run.folds, shuffle=True, random_state=cfg.run.seed)

    def objective(trial):
        max_features = trial.suggest_int("max_features", 5000, 20000, step=5000)
        alpha = trial.suggest_float("alpha", 1e-3, 1.0, log=True)
        fit_prior = trial.suggest_categorical("fit_prior", [True, False])
        vectorizer_type = trial.suggest_categorical("vectorizer", ["tfidf", "count"])

        auc_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            vec = get_vectorizer(vectorizer_type, max_features)
            X_train_vec = vec.fit_transform(X_train)
            X_val_vec = vec.transform(X_val)

            model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            model.fit(X_train_vec, y_train)

            y_proba = model.predict_proba(X_val_vec)[:, 1]
            auc = roc_auc_score(y_val, y_proba)

            # Optional: print F1 as well
            preds = model.predict(X_val_vec)
            f1 = f1_score(y_val, preds)
            print(f"Fold {fold+1} AUC: {auc:.4f} | F1: {f1:.4f}")

            auc_scores.append(auc)

        return np.mean(auc_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params
    print("\n✅ Best hyperparameters:")
    print(best_params)
    print(f"🎯 Best CV AUC: {study.best_value:.4f}")

    model_yaml_path = Path("config/model/naive_bayes.yaml")
    model_config = {
        "name": "naive_bayes",
        "type": "sklearn",
        "classifier": "naive_bayes",
        **best_params
    }
    model_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with model_yaml_path.open("w") as f:
        yaml.dump(model_config, f)

    comparison_path = Path(cfg.paths.output_dir) / "model_comparison.csv"
    cv_row = pd.DataFrame([{
        "model": "naive_bayes",
        "accuracy": None,
        "f1_score": None,
        "cv_auc": study.best_value
    }])
    if comparison_path.exists():
        existing = pd.read_csv(comparison_path)
        existing = existing[existing["model"] != "naive_bayes"]
        updated = pd.concat([existing, cv_row], ignore_index=True)
    else:
        updated = cv_row
    updated.to_csv(comparison_path, index=False)
    print(f"📄 CV AUC score logged to {comparison_path}")

    vec = get_vectorizer(best_params['vectorizer'], best_params['max_features'])
    X_vec = vec.fit_transform(X)
    model = MultinomialNB(alpha=best_params['alpha'], fit_prior=best_params['fit_prior'])
    model.fit(X_vec, y)

    dump(vec, f"{cfg.paths.output_dir}/nb_vectorizer.joblib")
    dump(model, f"{cfg.paths.output_dir}/nb_model.joblib")
    print("✅ Final model and vectorizer saved.")
