from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np


def train_naive_bayes(cfg, df):
    X = df[cfg.dataset.text_column]
    y = df[cfg.dataset.label_column]
    skf = StratifiedKFold(n_splits=cfg.run.folds, shuffle=True, random_state=cfg.run.seed)

    acc_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        vec = TfidfVectorizer(max_features=cfg.model.max_features)
        X_train_vec = vec.fit_transform(X_train)
        X_val_vec = vec.transform(X_val)

        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_val_vec)

        acc = accuracy_score(y_val, preds)
        acc_scores.append(acc)
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")

    print(f"Mean CV Accuracy: {np.mean(acc_scores):.4f}")