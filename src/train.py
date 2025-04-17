from src.models.logistic import train_logistic
from src.models.naive_bayes import train_naive_bayes

def train_model(cfg):
    import pandas as pd

    df = pd.read_csv(cfg.paths.train_data)

    model_type = cfg.model.classifier

    if model_type == "logistic":
        train_logistic(cfg, df)
    elif model_type == "naive_bayes":
        train_naive_bayes(cfg, df)
    else:
        raise ValueError(f"Unknown model: {model_type}")
