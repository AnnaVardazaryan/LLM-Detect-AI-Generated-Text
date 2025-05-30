import hydra
from omegaconf import DictConfig
from pathlib import Path

from src.inspectors.logistic_inspector import LogisticInspector
from src.inspectors.nb_inspector import NaiveBayesInspector

@hydra.main(config_path="../config", config_name="config.yaml")
def main(cfg: DictConfig):
    model_type = cfg.model.name  # e.g., "logistic" or "naive_bayes"
    model_path = Path(cfg.paths.output_dir) / f"{model_type}_model.joblib"
    vectorizer_path = Path(cfg.paths.output_dir) / f"{model_type}_vectorizer.joblib"

    if model_type == "logistic":
        inspector = LogisticInspector(model_path, vectorizer_path)
    elif model_type == "naive_bayes":
        inspector = NaiveBayesInspector(model_path, vectorizer_path)
    else:
        raise ValueError(f"Unsupported model: {model_type}")

    inspector.analyze()


if __name__ == "__main__":
    main()
