from pathlib import Path
import numpy as np
import optuna
import yaml
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.utils.vectorizer import get_vectorizer
import seaborn as sns
import matplotlib.pyplot as plt

def train_logistic(cfg, df):
    X = df[cfg.dataset.text_columns] if isinstance(cfg.dataset.text_columns, str) else df[cfg.dataset.text_columns].astype(str).agg(" ".join, axis=1)
    y = df[cfg.dataset.label_column]
    skf = StratifiedKFold(n_splits=cfg.run.folds, shuffle=True, random_state=cfg.run.seed)

    val_probas_all = []
    val_labels_all = []

    def objective(trial):
        max_features = trial.suggest_int("max_features", 5000, 20000, step=5000)
        C = trial.suggest_float("C", 0.01, 10.0, log=True)
        penalty = trial.suggest_categorical("penalty", ["l2"])
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
        max_iter = trial.suggest_int("max_iter", 100, 1000, step=100)
        vectorizer = trial.suggest_categorical("vectorizer", ["tfidf", "count"])

        auc_scores = []
        acc_scores = []
        f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            vec = get_vectorizer(vectorizer, max_features)
            X_train_vec = vec.fit_transform(X_train)
            X_val_vec = vec.transform(X_val)

            model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter)
            model.fit(X_train_vec, y_train)

            y_proba = model.predict_proba(X_val_vec)[:, 1]
            y_pred = model.predict(X_val_vec)

            val_probas_all.extend(y_proba)
            val_labels_all.extend(y_val.tolist())

            auc = roc_auc_score(y_val, y_proba)
            f1 = f1_score(y_val, y_pred)
            acc = accuracy_score(y_val, y_pred)

            print(f"Fold {fold+1} AUC: {auc:.4f} | F1: {f1:.4f} | Accuracy: {acc:.4f}")

            auc_scores.append(auc)
            acc_scores.append(acc)
            f1_scores.append(f1)

        train_logistic.cv_auc = np.mean(auc_scores)
        train_logistic.cv_accuracy = np.mean(acc_scores)
        train_logistic.cv_f1 = np.mean(f1_scores)

        return train_logistic.cv_auc  # Optuna optimizes AUC

    # Run Optuna tuning
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)

    best_params = study.best_params
    print("\n✅ Best hyperparameters:")
    print(best_params)
    print(f"🎯 Best CV AUC: {train_logistic.cv_auc:.4f}")
    print(f"📊 Mean CV Accuracy: {train_logistic.cv_accuracy:.4f}")
    print(f"📊 Mean CV F1 Score: {train_logistic.cv_f1:.4f}")

    # Save best hyperparameters
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

    # Save CV metrics
    comparison_path = Path(cfg.paths.output_dir) / "model_comparison.csv"
    cv_row = pd.DataFrame([{
        "model": "logistic",
        "cv_auc": train_logistic.cv_auc,
        "cv_accuracy": train_logistic.cv_accuracy,
        "cv_f1": train_logistic.cv_f1,
        "test_accuracy": None,
        "test_f1": None,
        "test_auc": None
    }])
    if comparison_path.exists():
        existing = pd.read_csv(comparison_path)
        existing = existing[existing["model"] != "logistic"]
        updated = pd.concat([existing, cv_row], ignore_index=True)
    else:
        updated = cv_row
    updated.to_csv(comparison_path, index=False)
    print(f"📄 CV scores logged to {comparison_path}")

    # Save validation predictions
    val_df = pd.DataFrame({
        "true_label": val_labels_all,
        "predicted_proba": val_probas_all
    })

    val_dir = Path(cfg.paths.output_dir) / "plots" / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    val_csv_path = val_dir / "logistic_val_preds.csv"
    val_df.to_csv(val_csv_path, index=False)
    print(f"📄 Saved validation predictions to {val_csv_path}")

    # Save boxplot
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=val_df, x="true_label", y="predicted_proba", palette="Set2")
    plt.title("Logistic Regression")
    plt.xlabel("True Label")
    plt.ylabel("Predicted Probability")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    boxplot_path = val_dir / "boxplot_logistic_val.png"
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    print(f"🖼️ Saved validation boxplot to {boxplot_path}")

    # Train final model
    vec = get_vectorizer(best_params['vectorizer'], best_params['max_features'])
    X_vec = vec.fit_transform(X)
    model = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'],
                               solver=best_params['solver'], max_iter=best_params['max_iter'])
    model.fit(X_vec, y)

    dump(vec, f"{cfg.paths.output_dir}/logistic_vectorizer.joblib")
    dump(model, f"{cfg.paths.output_dir}/logistic_model.joblib")
    print("✅ Final model and vectorizer saved.")
