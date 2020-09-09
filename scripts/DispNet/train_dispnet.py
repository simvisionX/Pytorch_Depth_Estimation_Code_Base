import argparse
import os
import logging
import time
import cv2 as cv
import numpy as np
from tqdm import tqdm
import time

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
from utils.utils import *


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
    parser.add_argument('--img_height', type=int, default=384, help='Image height for training')
    parser.add_argument('--img_width', type=int, default=768, help='Image width for training')
    parser.add_argument('--val_img_height', type=int, default=384, help='Image height for validation')
    parser.add_argument('--val_img_width', type=int, default=768, help='Image width for validation')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Default learning rate')
    parser.add_argument('--loss', type=str, help='indicates the loss scheme', default='./loss_configs/DispNetS_SceneFlow.json')
    parser.add_argument('--checkpoint_dir', type=str, help='Dir path to save weights', default='/home/cy')
    parser.add_argument('--dir_path', help='source datapath', default='/home/cy')
    parser.add_argument('--dataset_name', type=str, default='SceneFLow', help='depict a dataset to use')
    parser.add_argument('--mode', type=str, default='train', help='depict a training mode')
    parser.add_argument('--load_pseudo_gt', type=bool, default=False, help='whether to use pseudo labels')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='batch size for validation')
    parser.add_argument('--num_workers', type=int, default=1, help='dataloader workers')
    parser.add_argument('--pretrained_model', type=str, default='/home/cy', help='path to pretrained model')
    parser.add_argument('--kitti_finetune', type=bool, default=False, help='whether to finetune model on kitti')

    args = parser.parse_args()
    return args

def validate():
    pass




def adjust_learning_rate(optimizer, lr, epoch):

    cur_lr = lr / (2 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr



def train(model, train_loader, val_loader, device, logger, args):

    # load the training loss scheme
    loss_json = load_loss_scheme(args.loss)
    train_round = loss_json["round"]
    loss_scale = loss_json["loss_scale"]
    loss_weights = loss_json["loss_weights"]
    epoches = loss_json["epoches"]
    logger.info(loss_weights)

    # TODO:不同round间的学习率如何调整
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

    model.train()

    # if args.kitti_finetune:
    #     Supervised_loss = SupervisedLoss(scales=7, weights=loss_weights, loss='SmoothL1', downsample=True, mask=True)
    # else:
    #     Supervised_loss = SupervisedLoss(scales=7, weights=loss_weights, loss='L1', downsample=True, mask=False)

    for r in range(0, train_round):
        Supervised_loss = SupervisedLoss(loss_scale, 1, loss_weights[r], loss='L1', sparse=False)
        end_epoch = epoches[r]

        logger.info('round %d: %s' % (r, str(loss_weights[r])))
        logger.info('num of epoches: %d' % end_epoch)
        logger.info('\t'.join(['epoch', 'time_stamp', 'train_loss', 'train_EPE', 'EPE', 'lr']))
        for epoch in range(end_epoch):
            cur_lr = adjust_learning_rate(optimizer, epoch)
            logger.info("learning rate of epoch %d: %f." % (epoch, cur_lr))
            # loss_value = 0.
            num_batch = 0
            num_batches_per_epoch = len(train_loader)
            # with tqdm(total=len(train_loader), desc="Epoch %2d/%d" % (epoch, args.epoch)) as pbar:
            for idx_batch, sample in enumerate(train_loader):
                batch_time = time.time()
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
                # loss_value = loss_value + loss.item()
                batch_time = time.time() - batch_time

                if idx_batch % 10 == 0:
                    logger.info('Epoch: [{0}][{1}/{2}]\t'
                                'Time {3}\t'
                                'Loss {4}\t'.format(
                        epoch, idx_batch, num_batches_per_epoch, batch_time,
                        loss.item()))

                    # pbar.set_postfix({'loss': '%。6f' % (loss_value / num_batch)})
                    # pbar.update(1)

            model_path = os.path.join(args.save_path, "{:s}/weights_{:02d}".format(repr(model), epoch))
            os.makedirs(model_path, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(model_path, "test1.pth"))
            torch.save(optimizer.state_dict(), os.path.join(model_path, "test1_adam.pth"))



if __name__ == '__main__':
    args = parse_args()

    # set up logger
    logger = utils.get_logger()

    utils.check_path(args.checkpoint_dir)
    utils.save_args(args)

    filename = 'command_test.txt' if args.mode == 'test' else 'command_train.txt'
    utils.save_command(args.checkpoint_dir, filename)


    #  fix the random seed
    torch.manual_seed(233)
    torch.cuda.manual_seed(233)
    np.random.seed(233)

    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

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
                                        mode='val',
                                        transform=val_transform)
    val_loader = DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)


    # Model
    model = DispNet().to(device)
    # optimizer = optim.Adam(model.parameters(), args.learning_rate, betas=(0.9, 0.999))

    if torch.cuda.device_count() > 1:
        logger.info('=> %d GPUs' % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    train(model, train_loader, val_loader, device, logger, args)






















    
