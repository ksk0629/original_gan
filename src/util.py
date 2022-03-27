import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn


def weights_init(net: nn.Module) -> None:
    """Takes as input a neural network net that will initialize all its weights.

    :param torch.nn net: a neural network, which is Generator or Discriminator
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

