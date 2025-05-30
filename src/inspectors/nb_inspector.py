import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from joblib import load
from .base import BaseInspector

class NaiveBayesInspector(BaseInspector):
    def analyze(self, top_k=20):
        model = load(self.model_path)
        vectorizer = load(self.vectorizer_path)

        feature_names = vectorizer.get_feature_names_out()
        log_probs = model.feature_log_prob_

        df = pd.DataFrame({
            "feature": feature_names,
            "log_prob_human": log_probs[0],
            "log_prob_llm": log_probs[1]
        })
        df["log_odds"] = df["log_prob_llm"] - df["log_prob_human"]
        top_pos = df.sort_values("log_odds", ascending=False).head(top_k)
        top_neg = df.sort_values("log_odds", ascending=False).tail(top_k)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12, 6))
        sns.barplot(data=pd.concat([top_pos, top_neg]), x="log_odds", y="feature", palette="coolwarm")
        plt.title("Top Log-Odds Features (Naive Bayes)")
        plt.tight_layout()
        path = Path(self.output_dir) / "important_features_naive_bayes.png"
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"✅ Saved to {path}")
