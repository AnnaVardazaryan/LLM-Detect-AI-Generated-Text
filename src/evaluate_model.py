import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from joblib import load
from pathlib import Path
from sklearn.metrics import confusion_matrix


# Load test data
test_df = pd.read_csv("data/texts_labled.csv")
X_test = test_df["text"].astype(str)
y_test = test_df["generated"]  # Ground truth

# Define models to evaluate
models = {
    "logistic": {
        "model_path": "outputs/logistic_model.joblib",
        "vectorizer_path": "outputs/logistic_vectorizer.joblib"
    },
    "naive_bayes": {
        "model_path": "outputs/nb_model.joblib",
        "vectorizer_path": "outputs/nb_vectorizer.joblib"
    }
}

# Load or initialize results DataFrame
comparison_path = Path("outputs/model_comparison2.csv")
if comparison_path.exists():
    results_df = pd.read_csv(comparison_path)
else:
    results_df = pd.DataFrame(columns=["model", "accuracy", "f1_score", "cv_auc", "test_auc"])

# Evaluate each model
for name, paths in models.items():
    model = load(paths["model_path"])
    vectorizer = load(paths["vectorizer_path"])

    X_vec = vectorizer.transform(X_test)
    preds = model.predict(X_vec)
    probas = model.predict_proba(X_vec)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probas)

    # Update row or append new
    if name in results_df["model"].values:
        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        # Compute confusion matrix
        cm = confusion_matrix(y_test, preds)

        # Print as array
        print(f"\n📊 Confusion Matrix for {name}:\n{cm}")

        # Optional: show visual plot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", values_format='d')
        plt.title(f"Confusion Matrix: {name}")
        plt.show()
        results_df.loc[results_df["model"] == name, "accuracy"] = acc
        results_df.loc[results_df["model"] == name, "f1_score"] = f1
        results_df.loc[results_df["model"] == name, "test_auc"] = auc
    else:
        results_df = pd.concat([
            results_df,
            pd.DataFrame([{
                "model": name,
                "accuracy": acc,
                "f1_score": f1,
                "cv_auc": None,
                "test_auc": auc
            }])
        ], ignore_index=True)

# Save updated results
results_df.to_csv(comparison_path, index=False)
print(f"✅ Updated model comparison saved to {comparison_path}")
print(results_df)


