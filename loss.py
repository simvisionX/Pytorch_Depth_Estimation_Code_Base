import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedLoss(nn.modules.Module):
    def __init__(self, weights, levels, device=torch.device('cpu')):
        super(SupervisedLoss, self).__init__()
        self.weights = weights
        self.device = device
        self.levels = levels

    def forward(self, pred_disps, gt_disp, mask):
        loss = []
        for i in range(self.levels):
            pred_disp = pred_disps[i]
            pred_disp = F.interpolate(pred_disp, size=(gt_disp.size(-2), gt_disp.size(-1)),
                                      mode='bilinear', align_corners=False) * (gt_disp.size(-1) / pred_disp.size(-1)).to(self.device)

            loss.append(self.weights[i] *  F.smooth_l1_loss(pred_disp[mask], gt_disp[mask]))
        return sum(loss)