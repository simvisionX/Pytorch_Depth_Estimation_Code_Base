import argparse
import os
import logging
import time
import cv2 as cv
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dataloader
from dataloader import transforms
from utils import utils
from model_zoo import DispNet
from loss import SupervisedLoss


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
    parser.add_argument('--img_height', type=int, default=288, help='Image height for training')
    parser.add_argument('--img_width', type=int, default=512, help='Image width for training')
    parser.add_argument('--max_disp', type=int, default=192, help='maximum disparity')
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[1.0, 1.0, 1.0, 1.0, 1.0])
    parser.add_argument('--dir_path', default='/home/cy', help='datapath')
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
    parser.add_argument('--dataset_name', type=str, default='SceneFLow', help='depict a dataset to use')

    args = parser.parse_args()
    return args

def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    logger.info('learning rate is changed to %.2f' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def validate():
    pass



def train(model, train_loader, device, logger, args):
    model.train()

    Supervised_loss = SupervisedLoss(args.loss_weight, device)

    for epoch in range(args.epoch):
        adjust_learning_rate(optimizer, epoch)
        loss_value = 0.
        num_batch = 0
        with tqdm(total=len(train_loader), desc="Epoch %2d/%d" % (epoch, args.epoch)) as pbar:
            for i, sample in enumerate(train_loader):
                left = sample['left'].to(device)
                right = sample['right'].to(device)
                gt_disp = sample['disp'].to(device)
                mask = (gt_disp > 0) & (gt_disp < args.max_disp)

                optimizer.zero_grad()

                pred_disps = model(torch.cat([left, right], 1))
                loss = Supervised_loss(pred_disps, gt_disp, mask)

                loss.backward()
                optimizer.step()

                num_batch = num_batch + 1
                loss_value = loss_value + loss.item()

                pbar.set_postfix({'loss': '%。6f' % (loss_value / num_batch)})
                pbar.update(1)

        model_path = os.path.join(args.save_path, "{:s}/weights_{:02d}".format(repr(model), epoch))
        os.makedirs(model_path, exist_ok=True)
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(model_path, "test1.pth"))
        torch.save(optimizer.state_dict(), os.path.join(model_path, "test1_adam.pth"))



if __name__ == '__main__':
    args = parse_args()
    logger = utils.get_logger()

    utils.check_path(args.checkpoint_dir)
    utils.save_args(args)

    filename = 'command_test.txt' if args.mode == 'test' else 'command_train.txt'
    utils.save_command(args.checkpoint_dir, filename)



    torch.manual_seed(233)
    torch.cuda.manual_seed(233)
    np.random.seed(233)

    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # learning_rate = 0.001
    # Train Dataloader
    train_transform_list = [transforms.RandomCrop(args.img_height, args.img_width),
                            transforms.RandomColor(),
                            transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]

    train_transform = transforms.Compose(train_transform_list)

    train_data = dataloader.StereoDataset(data_dir=args.dir_path,
                                          dataset_name=args.dataset_name,
                                          mode='train' if args.mode != 'train_all' else 'train_all',
                                          load_pseudo_gt=args.load_pseudo_gt,
                                          transform=train_transform)

    logger.info('=> {} training samples found in the training set'.format(len(train_data)))

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # Validation Dataloader
    val_transform_list = [transforms.RandomCrop(args.val_img_height, args.val_img_width, validate=True),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]
    val_transform = transforms.Compose(val_transform_list)
    val_data = dataloader.StereoDataset(data_dir=args.data_dir,
                                        dataset_name=args.dataset_name,
                                        mode=args.mode,
                                        transform=val_transform)
    val_loader = DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)


    # Model
    model = DispNet().to(device)
    optimizer = optim.Adam(model.parameters(), args.learning_rate, betas=(0.9, 0.999))

    if args.pretrained_model is not None:
        model_dict = model.state_dict()
        pretrain_dict = torch.load(args.pretrained_model + ".pth")
        pretrain_dict = {k:v for k, v in pretrain_dict.items() if k in model_dict}

        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)

        try:
            # loading adam state
            optimizer_dict = torch.load(args.pretrained_model + "_adam.pth")
            optimizer.load_state_dict(optimizer_dict)
        except:
            logger.info("Cannot find Adam weights '{}', so Adam is randomly initialized").format(args.pretrain_model + "_adam.pth")


    if torch.cuda.device_count() > 1:
        logger.info('=> %d GPUs' % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    train(args)






















    
