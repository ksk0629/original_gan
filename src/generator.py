import torch
import torch.nn as nn

import util


class Generator(nn.Module):
    """Generator class"""

    def __init__(self, nz: int = 128, channels: int =3) -> None:
        super(Generator, self).__init__()

        self.nz = nz
        self.channels = channels

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0):
            block = [
                nn.ConvTranspose2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(n_output),
                nn.ReLU(inplace=True),
            ]
            return block

        self.model = nn.Sequential(
            *convlayer(self.nz, 1024, 4, 1, 0), # Fully connected layer via convolution.
            *convlayer(1024, 512, 4, 2, 1),
            *convlayer(512, 256, 4, 2, 1),
            *convlayer(256, 128, 4, 2, 1),
            *convlayer(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, self.channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(-1, self.nz, 1, 1)
        img = self.model(z)

        return img

    def save_model(self, model_path: str) -> None:
        """Save the weights of this model.

        :param str model_path: path to the weights of model file
        """
        torch.save(self.state_dict(), model_path)

    @classmethod
    def create_initial_model(cls: object) -> object:
        """Create a model whose weights are initialized.

        :return Generator generator: a Generator object
        """
        generator = cls()
        generator.apply(util.weights_init)
        
        return generator

    @classmethod
    def load_model(cls: object, model_path: str) -> object:
        """Load weights of a Generator model.

        :param str model_path: path to the weights of model file
        :return Generator generator: a loaded Generator object
        """
        generator = cls()
        generator.load_state_dict(torch.load(model_path))
        
        return generator
