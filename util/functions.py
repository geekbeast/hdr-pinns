import numpy as np
import torch
from torch import nn

mse = nn.MSELoss()


def esin(x: float):
    inv_x_values = 1 / x
    return np.sin(inv_x_values) * np.exp(inv_x_values)


def torch_sumAsin(input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    total = None
    for i in range(0, output.shape[1], 2):
        pair = torch.exp(output[:, i:i + 1]) * torch.sin(output[:, i + 1:i + 2] * input)
        if total is None:
            total = pair
        else:
            total = total + pair
    return total


def torch_sumAsin_loss(input: torch.Tensor, output, target):
    loss = torch.mean((torch_sumAsin(input, output) - target) ** 2)
    return loss


def torch_esin3d(output: torch.Tensor) -> torch.Tensor:
    assert (output.shape[1] == 3)
    one_over = (output[:, 0:1]) ** -1
    # one_over = output ** -1
    # val = torch.exp(output[:, 0:1] ) * torch.sin(output[:, 1:2])
    # val = output[:, 0:1] + output[:, 1:2]
    loss = output[:, 1:2] * torch.sin(one_over) * torch.cost(output[:, 2:3])
    return loss


def torch_esin3d_loss(output, target):
    # loss = mse(torch_esin3d(output), target)
    loss = torch.mean((torch_esin3d(output) - target) ** 2)
    # loss = torch.mean(torch.log( .001+ (torch_esin3d(output) - target) ** 2 ))
    # loss = torch.log(torch.mean((torch_esin3d(output) - target) ** 2))
    # return torch.clamp(loss, max= 10)
    return loss


class MultiSin:
    """

    """

    def __init__(self, amplitudes: list[float], frequencies: list[float]):
        """
        Instantiates a callable class that computes sum of sin functions with the specified `amplitudes`
        and `frequencies`
        :param amplitudes:
        :param frequencies:
        """
        assert (len(amplitudes) == len(frequencies), "The number of frequencies must match the number of amplitudes")
        self.amplitudes = amplitudes
        self.frequencies = frequencies

    def __call__(self, x):
        return sum([self.amplitudes[i] * np.sin(self.frequencies[i] * x) for i in range(len(self.amplitudes))])


class TwoSin(MultiSin):
    """
    """

    def __init__(self, a, b):
        """
        Instantiates a function that is a sum of two equal amplitude sin wave with
        frequencies a and b
        :param a: The frequency of one of the sin functions
        :param b: The frequency of the other sin function.
        """
        MultiSin.__init__(self, amplitudes=[1.0, 1.0], frequencies=[a, b])
