import hydra
from omegaconf import DictConfig
from src.train import train_model

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    train_model(cfg)

if __name__ == "__main__":
    main()