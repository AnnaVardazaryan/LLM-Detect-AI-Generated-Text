from src.utils.preprocess import load_data
from src.models.logistic import train_logistic
from src.models.naive_bayes import train_naive_bayes

def train_model(cfg):
    df = load_data(cfg.dataset.train_path)

    if cfg.model.classifier == "logistic":
        train_logistic(cfg, df)
    elif cfg.model.classifier == "naive_bayes":
        train_naive_bayes(cfg, df)
    else:
        raise ValueError("Unsupported model")
