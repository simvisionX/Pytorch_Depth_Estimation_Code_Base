import argparse
import os
import logging
import time
import cv2 as cv
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data
import torch.nn.functional as F

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

try:
    import nvidia.dali.plugin.torch
    dali_found = True
except:
    dali_found = False

def parse_args():
    parser = argparse.ArgumentParser(description='Train StereoNet')
    parser.add_argument('--max_disp', type=int, default=192, help='maximum disparity')
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[1.0, 1.0, 1.0, 1.0, 1.0])
    parser.add_argument('--datapath', default='/home/cy', help='datapath')
    parser.add_argument('--epoch', type=int, default=15, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--iter_size', type=int, default=1, metavar='IS', help='iter size')
    parser.add_argument('--save_path', type=str, default='/home/cy', help='path od saving checkpoints and log')
    parser.add_argument('--resume', type=str, default=None, help='resume path')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', '--wd', type=float, default=2e-4, metavar='W', help='default weight decay')
    parser.add_argument('--step_size', type=int, default=1, metavar='SS', help='learning rate step size')
    parser.add_argument('--gamma', '--gm', type=float, default=0.6, help='learning rate decay parameter: Gamma')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequence')
    parser.add_argument('--stages', type=int, default=4, help='the stage num of refinement')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')

    args = parser.parse_args()
    return args

def validate():
    pass



def train():
    pass

if __name__ == '__main__':
    args = parse_args()








