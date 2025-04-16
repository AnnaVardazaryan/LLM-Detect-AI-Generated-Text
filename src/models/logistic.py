import optuna
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump

def get_vectorizer(name, max_features):
    if name == "tfidf":
        return TfidfVectorizer(max_features=max_features)
    elif name == "count":
        return CountVectorizer(max_features=max_features)
    # elif name == "hashing":
    #     return HashingVectorizer(n_features=max_features, alternate_sign=False)
    else:
        raise ValueError(f"Unsupported vectorizer type: {name}")

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
        vectorizer_type = trial.suggest_categorical("vectorizer", ["tfidf", "count"])

        f1_scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            vec = get_vectorizer(vectorizer_type, max_features)
            X_train_vec = vec.fit_transform(X_train)
            X_val_vec = vec.transform(X_val)

            model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter)
            model.fit(X_train_vec, y_train)
            preds = model.predict(X_val_vec)

            f1 = f1_score(y_val, preds)
            f1_scores.append(f1)

        return np.mean(f1_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("\n Best hyperparameters:")
    print(study.best_params)
    print(f"Best CV F1 Score: {study.best_value:.4f}")

    with open(f"{cfg.paths.output_dir}/logistic_best_params.txt", "w") as f:
        f.write(f"Best Params: {study.best_params}\n")
        f.write(f"Best F1 Score: {study.best_value:.4f}\n")


    best_params = study.best_params
    vec = get_vectorizer(best_params['vectorizer'], best_params['max_features'])
    X_vec = vec.fit_transform(X)
    model = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], solver=best_params['solver'], max_iter=best_params['max_iter'])
    model.fit(X_vec, y)

    dump(vec, f"{cfg.paths.output_dir}/logistic_vectorizer.joblib")
    dump(model, f"{cfg.paths.output_dir}/logistic_model.joblib")