import numpy as np
import optuna
import yaml
import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

from src.utils.vectorizer import get_vectorizer


def train_naive_bayes(cfg, df):
    import seaborn as sns
    import matplotlib.pyplot as plt

    X = df[cfg.dataset.text_columns] if isinstance(cfg.dataset.text_columns, str) else df[cfg.dataset.text_columns].astype(str).agg(" ".join, axis=1)
    y = df[cfg.dataset.label_column]
    skf = StratifiedKFold(n_splits=cfg.run.folds, shuffle=True, random_state=cfg.run.seed)

    val_probas_all = []
    val_labels_all = []

    def objective(trial):
        max_features = trial.suggest_int("max_features", 5000, 20000, step=5000)
        alpha = trial.suggest_float("alpha", 1e-3, 1.0, log=True)
        fit_prior = trial.suggest_categorical("fit_prior", [True, False])
        vectorizer_type = trial.suggest_categorical("vectorizer", ["tfidf", "count"])

        auc_scores = []
        acc_scores = []
        f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            vec = get_vectorizer(vectorizer_type, max_features)
            X_train_vec = vec.fit_transform(X_train)
            X_val_vec = vec.transform(X_val)

            model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            model.fit(X_train_vec, y_train)

            y_proba = model.predict_proba(X_val_vec)[:, 1]
            preds = model.predict(X_val_vec)

            val_probas_all.extend(y_proba)
            val_labels_all.extend(y_val.tolist())

            auc = roc_auc_score(y_val, y_proba)
            f1 = f1_score(y_val, preds)
            acc = accuracy_score(y_val, preds)

            print(f"Fold {fold+1} AUC: {auc:.4f} | F1: {f1:.4f} | Accuracy: {acc:.4f}")

            auc_scores.append(auc)
            acc_scores.append(acc)
            f1_scores.append(f1)

        # Store for logging after tuning
        train_naive_bayes.cv_auc = np.mean(auc_scores)
        train_naive_bayes.cv_accuracy = np.mean(acc_scores)
        train_naive_bayes.cv_f1 = np.mean(f1_scores)

        return train_naive_bayes.cv_auc  # Optuna still optimizes AUC

    # Run Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params
    print("\n✅ Best hyperparameters:")
    print(best_params)
    print(f"🎯 Best CV AUC: {train_naive_bayes.cv_auc:.4f}")
    print(f"📊 Mean CV Accuracy: {train_naive_bayes.cv_accuracy:.4f}")
    print(f"📊 Mean CV F1 Score: {train_naive_bayes.cv_f1:.4f}")

    # Save best params
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

    # Save comparison metrics
    comparison_path = Path(cfg.paths.output_dir) / "model_comparison.csv"
    cv_row = pd.DataFrame([{
        "model": "naive_bayes",
        "cv_auc": train_naive_bayes.cv_auc,
        "cv_accuracy": train_naive_bayes.cv_accuracy,
        "cv_f1": train_naive_bayes.cv_f1,
        "test_accuracy": None,
        "test_f1": None,
        "test_auc": None
    }])
    if comparison_path.exists():
        existing = pd.read_csv(comparison_path)
        existing = existing[existing["model"] != "naive_bayes"]
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

    val_csv_path = val_dir / "naive_bayes_val_preds.csv"
    val_df.to_csv(val_csv_path, index=False)
    print(f"📄 Saved validation predictions to {val_csv_path}")

    # Save boxplot
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=val_df, x="true_label", y="predicted_proba", palette="Set2")
    plt.title("Naive Bayes")
    plt.xlabel("True Label")
    plt.ylabel("Predicted Probability")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    boxplot_path = val_dir / "boxplot_naive_bayes_val.png"
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    print(f"🖼️ Saved validation boxplot to {boxplot_path}")

    # Final model training
    vec = get_vectorizer(best_params['vectorizer'], best_params['max_features'])
    X_vec = vec.fit_transform(X)
    model = MultinomialNB(alpha=best_params['alpha'], fit_prior=best_params['fit_prior'])
    model.fit(X_vec, y)

    dump(vec, f"{cfg.paths.output_dir}/nb_vectorizer.joblib")
    dump(model, f"{cfg.paths.output_dir}/nb_model.joblib")
    print("✅ Final model and vectorizer saved.")
