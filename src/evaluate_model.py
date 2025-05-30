import hydra
from omegaconf import DictConfig
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from joblib import load
from pathlib import Path


@hydra.main(config_path="../config", config_name="config.yaml")
def main(cfg: DictConfig):
    # 📂 Load test data and columns from config
    test_df = pd.read_csv(cfg.paths.test_data)
    X_test = test_df[cfg.dataset.text_columns]
    y_test = test_df[cfg.dataset.label_column]

    # 🧠 Define models to evaluate
    models = {
        "Logistic Regression": {
            "model_path": "outputs/logistic_model.joblib",
            "vectorizer_path": "outputs/logistic_vectorizer.joblib",
            "threshold": 0.5
        },
        "Naive Bayes": {
            "model_path": "outputs/nb_model.joblib",
            "vectorizer_path": "outputs/nb_vectorizer.joblib",
            "threshold": 0.5
        }
    }

    # 📄 Where to save results
    comparison_path = Path(cfg.paths.output_dir) / "model_comparison_test.csv"
    plot_dir = Path(cfg.paths.output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 📊 Initialize results table
    results_df = pd.DataFrame(columns=["model", "accuracy", "f1_score", "auc"])

    # 🔁 Evaluate each model
    for name, paths in models.items():
        model = load(paths["model_path"])
        vectorizer = load(paths["vectorizer_path"])
        X_vec = vectorizer.transform(test_df["text"].astype(str))  # TF-IDF input must be raw text

        probas = model.predict_proba(X_vec)[:, 1]
        threshold = paths.get("threshold", 0.5)
        preds = (probas >= threshold).astype(int)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probas)

        # 🧮 Save metrics
        results_df = pd.concat([
            results_df,
            pd.DataFrame([{
                "model": name,
                "accuracy": acc,
                "f1_score": f1,
                "auc": auc
            }])
        ], ignore_index=True)

        # 📉 Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", values_format='d')
        plt.title(f"Confusion Matrix: {name}")
        plt.savefig(plot_dir / f"confusion_matrix_{name.replace(' ', '_')}.png", dpi=300)
        plt.close()

        # 📦 Boxplot
        proba_df = pd.DataFrame({
            "predicted_proba": probas,
            "true_label": y_test
        })

        plt.figure(figsize=(6, 4))
        sns.boxplot(data=proba_df, x="true_label", y="predicted_proba", palette="Set2")
        plt.title(name)
        plt.xlabel("True Label")
        plt.ylabel("Predicted Probability")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(plot_dir / f"boxplot_{name.replace(' ', '_')}.png", dpi=300)
        plt.close()

    # 💾 Save updated results
    results_df.to_csv(comparison_path, index=False)
    print(f"\n✅ Updated model comparison saved to {comparison_path}")
    print(results_df)
    print(f"\n🖼️ Plots saved to {plot_dir.resolve()}")


if __name__ == "__main__":
    main()
