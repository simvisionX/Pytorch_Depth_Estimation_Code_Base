import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn


def convbn(in_channel, out_channel, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation),
        nn.BatchNorm2d(out_channel))


def convbn_3d(in_channel, out_channel, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            padding=pad,
            stride=stride),
        nn.BatchNorm3d(out_channel))

class DispNet(nn.Module):
    pass
