import os
import time
from tkinter import N
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import torch
from torch import nn, optim
from tqdm import tqdm

from discriminator import Discriminator
from generator import Generator


class GanTrainer():
    """Gan Trainer class"""

    def __init__(self, discriminator: Discriminator, generator: Generator) -> None:
        self.generator = generator
        self.discriminator = discriminator

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.run_id = None
        self.final_discriminator_path = None
        self.final_generator_path = None
        self.final_image_path = None

    def set_training_configs(self, train_loader, epochs: int, real_label: float, fake_label: float, output_dir: str, learning_rate_d: float, learning_rate_g: float, beta1: float,
                             experiment_name: str, run_name: str, run_id: Optional[str]):
        self.epochs = epochs
        self.epoch_times = []
        self.train_loader = train_loader
        self.real_label = real_label
        self.fake_label = fake_label

        # Set paths to output weights
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir
        self.discriminator_path = os.path.join(self.output_dir, "discriminator")
        self.generator_path = os.path.join(self.output_dir, "generator")

        # Set optimizers
        self.beta1 = beta1
        self.learning_rate_d = learning_rate_d
        self.learning_rate_g = learning_rate_g
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate_d, betas=(beta1, 0.999))
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.learning_rate_g, betas=(beta1, 0.999))

        self.criterion = nn.BCELoss()
        self.loss_graphs = []

        # Run mlflow
        mlflow.set_experiment(experiment_name)
        if run_id is not None:
            mlflow.start_run(run_id=run_id)
        else:
            mlflow.start_run(run_name=run_name)
        self.run_id = mlflow.active_run().info.run_id

        # Register information
        # mlflow.pytorch.autolog
        mlflow.log_param("epochs", self.epochs)
        mlflow.log_param("real_label", self.real_label)
        mlflow.log_param("fake_label", self.fake_label)
        mlflow.log_param("learning_rate_d", self.learning_rate_d)
        mlflow.log_param("learning_rate_g", self.learning_rate_g)
        mlflow.log_param("beta1", self.beta1)

    def train(self, should_save_all: bool = False):
        if self.run_id is None:
            msg = "set_training_configs function must be run before running train function."
            raise ValueError(msg)

        try:
            self.current_itteration = 0
            for epoch in range(self.epochs):
                d_losses, g_losses, epoch_time = self.__train_one_epoch(epoch)
                self.__save_loss_graph(g_losses, d_losses, epoch)

                if epoch % 10 == 0:
                    self.__show_generated_images(n_images=5, epoch=epoch)

                if should_save_all:
                    self.final_discriminator_path = self.discriminator_path + f"_{epoch}.pt"
                    self.final_generator_path = self.generator_path + f"_{epoch}.pt"
                else:
                    self.final_discriminator_path = self.discriminator_path + ".pt"
                    self.final_generator_path = self.generator_path + ".pt"
                torch.save(self.discriminator.state_dict(), self.final_discriminator_path)
                torch.save(self.generator.state_dict(), self.final_generator_path)
                
                self.epoch_times.append(epoch_time)
        except KeyboardInterrupt:
            msg = f"run_id-{self.run_id}: An error occurs on epoch {epoch}"
            print(msg)
        finally:
            if self.final_discriminator_path is not None:
                mlflow.log_artifact(self.final_discriminator_path)
            if self.final_generator_path is not None:
                mlflow.log_artifact(self.final_generator_path)
            if self.final_image_path is not None:
                mlflow.log_artifact(self.final_image_path)

    def __train_one_epoch(self, epoch: int):
        g_losses = []
        d_losses = []
        
        start = time.time()
        for ii, (real_images, _) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            self.discriminator.zero_grad()
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)
            labels = torch.full((batch_size, 1), self.real_label, device=self.device)

            output = self.discriminator(real_images)
            errD_real = self.criterion(output, labels)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, self.generator.nz, 1, 1, device=self.device)
            fake = self.generator(noise)
            labels.fill_(self.fake_label)
            output = self.discriminator(fake.detach())
            errD_fake = self.criterion(output, labels)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            self.optimizer_d.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.generator.zero_grad()
            labels.fill_(self.real_label)  # fake labels are real for generator cost
            output = self.discriminator(fake)
            errG = self.criterion(output, labels)
            errG.backward()
            D_G_z2 = output.mean().item()
            self.optimizer_g.step()

            mlflow.log_metric("g_loss", errG.item(), step=ii + epoch*len(self.train_loader))
            mlflow.log_metric("d_loss", errD.item(), step=ii + epoch*len(self.train_loader))

            if (ii+1) % (len(self.train_loader) // 2) == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch + 1, self.epochs, ii+1, len(self.train_loader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        epoch_time = time.time()- start

        return d_losses, g_losses, epoch_time

    def __save_loss_graph (self, g_losses, d_losses, epoch) -> None:
        fig = plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss - EPOCH "+ str(epoch))
        plt.plot(g_losses, label="G")
        plt.plot(d_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        # plt.show()

        self.loss_graphs.append(fig)

        plt.close()

    def __show_generated_images(self, n_images: int, epoch: int) -> None:
        sample = []
        for _ in range(n_images):
            noise = torch.randn(1, self.generator.nz, 1, 1, device=self.device)
            gen_image = self.generator(noise).to("cpu").clone().detach().squeeze(0)
            gen_image = gen_image.numpy().transpose(1, 2, 0)
            sample.append(gen_image)

        figure, axes = plt.subplots(1, len(sample), figsize = (64,64))
        for index, axis in enumerate(axes):
            axis.axis('off')
            image_array = sample[index]
            axis.imshow(image_array)

        image_path = os.path.join(self.output_dir, f"image_{epoch}")
        figure.savefig(image_path)
        self.final_image_path = image_path

        plt.close()
