import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn
from torch.nn.init import kaiming_normal
from .submodules import *


class DispNet(nn.Module):
    def __init__(self, levels=6, input_channel=3):
        super(DispNet, self).__init__()

        self.levels = levels
        self.input_Cahnnel = input_channel

        self.conv1 = convbnrelu(6, 64, 7, 2, 3, 1)
        self.conv2 = convbnrelu(64, 128, 5, 2, 2, 1)
        self.conv3a = convbnrelu(128, 256, 5, 2, 2, 1)
        self.conv3b = convbnrelu(256, 256, 3, 1, 1, 1)
        self.conv4a = convbnrelu(256, 512, 3, 2, 1, 1)
        self.conv4b = convbnrelu(512, 512, 3, 1, 1, 1)
        self.conv5a = convbnrelu(512, 512, 3, 2, 1, 1)
        self.conv5b = convbnrelu(512, 512, 3, 1, 1, 1)
        self.conv6a = convbnrelu(512, 1024, 3, 2, 1, 1)
        self.conv6b = convbnrelu(1024, 1024, 3, 1, 1, 1)

        # pr6 + loss6
        self.pr_6 = predict_flow(1024, 1)

        self.upconv5 = upconvbnrelu(1024, 512, 4, 2, 1, 1)
        self.upconv5_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv5 = nn.Conv2d(1025, 512, 3, 1, 1, 1)

        # pr5 + loss5
        self.pr_5 = predict_flow(512, 1)

        self.upconv4 = upconvbnrelu(512, 256, 4, 2, 1, 1)
        self.upconv4_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv4 = nn.Conv2d(769, 256, 3, 1, 1, 1)

        # pr4 + loss4
        self.pr_4 = predict_flow(256, 1)

        self.upconv3 = upconvbnrelu(256, 128, 4, 2, 1, 1)
        self.upconv3_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv3 = nn.Conv2d(385, 128, 3, 1, 1, 1)

        # pr3 + loss3
        self.pr_3 = predict_flow(128, 1)

        self.upconv2 = upconvbnrelu(128, 64, 4, 2, 1, 1)
        self.upconv2_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv2 = nn.Conv2d(193, 64, 3, 1, 1, 1)

        # pr2 + loss2
        self.pr_2 = predict_flow(64, 1)

        self.upconv1 = upconvbnrelu(64, 32, 4, 2, 1, 1)
        self.upconv1_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv1 = nn.Conv2d(97, 32, 3, 1, 1, 1)

        # pr1 + loss1
        self.pr_1 = predict_flow(32, 1)

        if self.levels == 7:
            self.upconv0 = upconvbnrelu(32, 16, 4, 2, 1, 1)
            self.upconv0_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
            self.iconv0 = nn.Conv2d(17 + self.input_channel, 16, 3, 1, 1, 1)

            self.pr_0 = predict_flow(16, 1)

        # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):

        img_left = x[:, :3, :, :]
        output = []

        encode_1 = self.conv1(x)
        encode_2 = self.conv2(encode_1)
        encode_3 = self.conv3b(self.conv3a(encode_2))
        encode_4 = self.conv4b(self.conv4a(encode_3))
        encode_5 = self.conv5b(self.conv5a(encode_4))

        encode_6 = self.conv6b(self.conv6a(encode_5))
        output_0 = self.pr_6(encode_6)
        output.append(output_0)

        decode_5 = self.iconv5(torch.cat([self.upconv5(encode_6), self.upconv5_2(output_0), encode_5], 1))
        output_1 = self.pr_5(decode_5)
        output.append(output_1)

        decode_4 = self.iconv4(torch.cat([self.upconv4(decode_5), self.upconv4_2(output_1), encode_4], 1))
        output_2 = self.pr_4(decode_4)
        output.append(output_2)

        decode_3 = self.iconv3(torch.cat([self.upconv3(decode_4), self.upconv3_2(output_2), encode_3], 1))
        output_3 = self.pr_3(decode_3)
        output.append(output_3)

        decode_2 = self.iconv2(torch.cat([self.upconv2(decode_3), self.upconv2_2(output_3), encode_2], 1))
        output_4 = self.pr_2(decode_2)
        output.append(output_4)

        decode_1 = self.iconv1(torch.cat([self.upconv1(decode_2), self.upconv1_2(output_4), encode_1], 1))
        output_5 = self.pr_1(decode_1)
        output.append(output_5)

        if self.levels == 7:
            decode_0 = self.iconv0(torch.cat([self.upconv0(decode_1), self.upconv0_2(output_5), img_left], 1))
            output_6 = self.pr_0(decode_0)
            output.append(output_6)

        return output


if __name__ == "__main__":
    model = DispNet()
    print(model)