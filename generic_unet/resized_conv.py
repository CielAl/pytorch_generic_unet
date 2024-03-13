from typing import Callable

from torch import nn
from torch.nn import functional as F


class ResizeConv(nn.Module):
    def __init__(self,
                 new_size,
                 kernel_size: int,
                 in_channels: int,
                 out_channels: int,
                 activation: Callable = nn.ReLU):
        super().__init__()
        self.new_size = new_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            activation()
        )

    def forward(self, x):
        p = F.interpolate(input=x, size=self.new_size, mode='nearest')
        return self.conv(p)
