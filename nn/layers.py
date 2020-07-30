from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np


def groupwise_correlation(fea1, fea2, num_groups):
    # turn into full correlation volume when num_groups equal to 1
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C / num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    if num_groups == 1:
        cost = cost.squeeze(dim=1)
        assert cost.shape == (B, H, W)
    return cost


def correlate_volume(fea1, fea2, max_disp, num_groups):
    B, C, H, W = fea1.shape
    if num_groups != 1:
        cost_volume = fea1.new_zeros([B, num_groups, max_disp, H, W])
        for i in range(max_disp):
            if i == 0:
                cost_volume[:, :, i, :, :] = groupwise_correlation(fea1, fea2, num_groups)
            else:
                cost_volume[:, :, i, :, i:] = groupwise_correlation(fea1[:,:,:,i:], fea2[:,:,:,:-i], num_groups)
    else:
        cost_volume = fea1.new_zeros([B, max_disp, H, W])
        for i in max_disp:
            if i == 0:
                cost_volume[:, i, :, :] = groupwise_correlation(fea1, fea2, num_groups)
            else:
                cost_volume[:, i, :, i:] = groupwise_correlation(fea1[:,:,:,i:], fea2[:,:,:,:-i], num_groups)
    cost_volume = cost_volume.contiguous()
    return cost_volume


def concat_volume(fea1, fea2, max_disp):
    B, C, H, W = fea1.shape
    cost_volume = fea1.new_zeros([B, 2 * C, max_disp, H, W])
    for i in range(max_disp):
        if i == 0:
            cost_volume[:, :C, i, :, :] = fea1
            cost_volume[:, C:, i, :, :] = fea2
        else:
            cost_volume[:, :C, i, :, i:] = fea1[:, :, :, i:]
            cost_volume[:, C:, i, :, i:] = fea2[:, :, :, :-i]
    cost_volume = cost_volume.contiguous()
    return cost_volume


