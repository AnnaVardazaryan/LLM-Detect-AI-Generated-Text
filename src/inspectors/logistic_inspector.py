import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from joblib import load
from .base import BaseInspector

class LogisticInspector(BaseInspector):
    def analyze(self, top_k=20):
        # 🔁 Load model and vectorizer
        model = load(self.model_path)
        vectorizer = load(self.vectorizer_path)

        # 🔤 Get feature names and coefficients
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[0]

        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coef": coefs
        })

        # 📊 Top positive (LLM) and negative (Human) features
        top_pos = coef_df.sort_values("coef", ascending=False).head(top_k)
        top_neg = coef_df.sort_values("coef", ascending=True).head(top_k)

        print("\nTop Positive (LLM) Features:")
        print(top_pos)
        print("\nTop Negative (Human) Features:")
        print(top_neg)

        # 🟦🟥 Add direction label
        top_pos["direction"] = "LLM-associated"
        top_neg["direction"] = "Human-associated"

        coef_plot_df = pd.concat([top_pos, top_neg]).sort_values("coef", ascending=False)


        # 📁 Output path
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(self.output_dir) / "important_features_logistic.png"

        # 🖼️ Plot
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=coef_plot_df,
            x="coef",
            y="feature",
            hue="direction",
            dodge=False,
            palette={
                "LLM-associated": "steelblue",
                "Human-associated": "indianred"
            }
        )
        plt.axvline(0, color="gray", linestyle="--")
        plt.title("Top Influential Features (Logistic Regression)")
        plt.xlabel("Coefficient")
        plt.ylabel("Feature (word)")
        plt.legend(title="Direction of Influence", loc="lower right")

        # 📘 Explanation note
        plt.figtext(
            0.01, -0.15,
            "🔵 Positive (blue) → words that push model toward 'LLM-generated'\n"
            "🔴 Negative (red) → words that push model toward 'Human-written'\n\n"
            "Only the top 20 features from each side are shown.\n"
            "If the model overfits training style, it may classify all new data as 'human'.",
            ha="left", va="bottom", fontsize=9
        )

        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()

        print(f"✅ Saved plot to {path.resolve()}")
