import argparse
from typing import Optional

import yaml

from discriminator import Discriminator
from gan_trainer import GanTrainer
from generator import Generator
from preprocess import preprocess_from_path


def train(data_dir: str, horizontal_flip_p: float, random_apply_p: float, image_size: int, batch_size: int, shuffle: bool, channels: int, nz: int,
          epochs: int, real_label: float, fake_label: float, output_dir: str, learning_rate_d: float, learning_rate_g: float, beta1: float, 
          experiment_name: str, run_name: str, run_id: Optional[str], should_save_all: bool):
    train_loader = preprocess_from_path(data_dir=data_dir, horizontal_flip_p=horizontal_flip_p, random_apply_p=random_apply_p, image_size=image_size, batch_size=batch_size, shuffle=shuffle)
    
    discriminator = Discriminator(channels=channels)
    generator = Generator(nz=nz, channels=channels)
    
    gan_trainer = GanTrainer(discriminator=discriminator, generator=generator)
    gan_trainer.set_training_configs(train_loader=train_loader, epochs=epochs, real_label=real_label, fake_label=fake_label, output_dir=output_dir,
                                     learning_rate_d=learning_rate_d, learning_rate_g=learning_rate_g, beta1=beta1,
                                     experiment_name=experiment_name, run_name=run_name, run_id=run_id)

    gan_trainer.train(should_save_all=should_save_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate RNN for air quality uci dataset.")
    parser.add_argument("-c", "--config_yaml_path", required=False, type=str, default="./config.yaml")
    args = parser.parse_args()

    # Load configs
    with open(args.config_yaml_path, "r") as yaml_f:
        config = yaml.safe_load(yaml_f)
    config_mlflow = config["mlflow"]
    config_dataset = config["dataset"]
    config_model = config["model"]
    config_train = config["train"]

    train(**config_mlflow, **config_dataset, **config_model, **config_train)
