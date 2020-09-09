import torch
import torch.nn as nn
import torch.nn.functional as F


def EPE(input_flow, target_flow):
    target_valid = target_flow < 192
    return F.smooth_l1_loss(input_flow[target_valid], target_flow[target_valid], size_average=True)

class MultiScaleLoss(nn.Module):

    def __init__(self, scales, weights=None, loss='L1', downsample=True, mask=False):
        super(MultiScaleLoss, self).__init__()
        # self.downscale = downscale
        self.downsample=downsample
        self.mask = mask
        self.weights = torch.Tensor(scales).fill_(1).cuda() if weights is None else torch.Tensor(weights).cuda()
        assert (len(self.weights) == scales)

        if type(loss) is str:
            if loss == 'L1':
                self.loss = nn.L1Loss()
            elif loss == 'MSE':
                self.loss = nn.MSELoss()
            elif loss == 'SmoothL1':
                self.loss = nn.SmoothL1Loss()
        else:
            self.loss = loss
        # self.multiScales = [nn.AvgPool2d(self.downscale * (2 ** i), self.downscale * (2 ** i)) for i in range(scales)]
        if self.downsample:
            self.multiScales = [nn.AvgPool2d(self.downscale * (2 ** i), self.downscale * (2 ** i)) for i in range(scales)]
        else:
            self.multiScales = [nn.Upsample(scale_factor=self.downscale * (2 ** i), mode='bilinear', align_corners=True)
                                for i in range(scales)]
        # self.multiScales = [nn.MaxPool2d(self.downscale*(2**i), self.downscale*(2**i)) for i in range(scales)]
        # self.multiScales = [nn.Upsample(scale_factor=self.downscale*(2**i), mode='bilinear', align_corners=True) for i in range(scales)]
        # print('self.multiScales: ', self.multiScales, ' self.downscale: ', self.downscale)
        # self.multiScales = [nn.functional.adaptive_avg_pool2d(self.downscale*(2**i), self.downscale*(2**i)) for i in range(scales)]

    def forward(self, input, target):
        # print(len(input))
        if (type(input) is tuple) or (type(input) is list):
            out = 0

            for i, input_ in enumerate(input):

                target_ = self.multiScales[i](target)

                if self.mask:
                    # work for sparse
                    mask = target > 0
                    mask.detach_()

                    mask = mask.type(torch.cuda.FloatTensor)
                    pooling_mask = self.multiScales[i](mask)

                    # use unbalanced avg
                    target_ = target_ / pooling_mask

                    mask = target_ > 0
                    mask.detach_()
                    input_ = input_[mask]
                    target_ = target_[mask]

                EPE_ = EPE(input_, target_)
                out += self.weights[i] * EPE_
        else:
            out = self.loss(input, self.multiScales[0](target))
        return out


def SupervisedLoss(scales=6, weights=None, loss='L1', downsample=True, mask=False):
    if weights is None:
        weights = (0.005, 0.01, 0.02, 0.08, 0.32)
    if scales == 1 and type(weights) is not tuple:
        weights = (weights,)
    return MultiScaleLoss(scales, weights, loss, mask)

# class SupervisedLoss(nn.modules.Module):
#     def __init__(self, weights, levels, device=torch.device('cpu')):
#         super(SupervisedLoss, self).__init__()
#         self.weights = weights
#         self.device = device
#         self.levels = levels
#
#     def forward(self, pred_disps, gt_disp, mask):
#         loss = []
#         for i in range(self.levels):
#             pred_disp = pred_disps[i]
#             pred_disp = F.interpolate(pred_disp, size=(gt_disp.size(-2), gt_disp.size(-1)),
#                                       mode='bilinear', align_corners=False) * (gt_disp.size(-1) / pred_disp.size(-1)).to(self.device)
#
#             loss.append(self.weights[i] *  F.smooth_l1_loss(pred_disp[mask], gt_disp[mask]))
#         return sum(loss)