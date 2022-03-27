import torch
import torch.nn as nn

import util


class Discriminator(nn.Module):
    """Discriminator class"""

    def __init__(self, channels: int = 3):
        super(Discriminator, self).__init__()

        self.channels = channels

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0, bn=False):
            block = [nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)]
            if bn:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.model = nn.Sequential(
            *convlayer(self.channels, 32, 4, 2, 1),
            *convlayer(32, 64, 4, 2, 1),
            *convlayer(64, 128, 4, 2, 1, bn=True),
            *convlayer(128, 256, 4, 2, 1, bn=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),  # FC with Conv.
        )

    def forward(self, imgs):
        logits = self.model(imgs)
        out = torch.sigmoid(logits)

        return out.view(-1, 1)

    def save_model(self, model_path: str) -> None:
        """Save the weights of this model.

        :param str model_path: path to the weights of model file
        """
        torch.save(self.state_dict(), model_path)

    @classmethod
    def create_initial_model(cls: object) -> object:
        """Create a model whose weights are initialized.

        :return Discriminator discriminator: a Discriminator object
        """
        discriminator = cls()
        discriminator.apply(util.weights_init)
        
        return discriminator

    @classmethod
    def load_model(cls: object, model_path: str) -> object:
        """Load weights of a Discriminator model.

        :param str model_path: path to the weights of model file
        :return Discriminator discriminator: a loaded Discriminator object
        """
        discriminator = cls()
        discriminator.load_state_dict(torch.load(model_path))
        
        return discriminator
