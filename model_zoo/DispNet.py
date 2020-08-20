import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn


def convbnrelu(in_channel, out_channel, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

def upconvbnrelu(in_channel, out_channel, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


class DispNet(nn.Module):
    def __init__(self):
        super(DispNet, self).__init__()
        self.conv1 = convbnrelu(6, 64, 7, 2, 3, 1)
        self.conv2 = convbnrelu(64, 128, 5, 2, 2, 1)
        self.conv3a = convbnrelu(128, 256, 5, 2, 2, 1)
        self.conv3b = convbnrelu(256, 256, 3, 1, 1, 1)
        self.conv4a = convbnrelu(256, 512, 3, 2, 1, 1)
        self.conv4b = convbnrelu(512, 512, 3, 1, 1, 1)
        self.conv5a = convbnrelu(512, 512, 3, 1, 1, 1)
        self.conv5b = convbnrelu(512, 512, 3, 1, 1, 1)
        self.conv6a = convbnrelu(512, 1024, 3, 2, 1, 1)
        self.conv6b = convbnrelu(1024, 1024, 3, 1, 1, 1)

        # pr6 + loss6
        self.conv7 = convbnrelu(1024, 1, 3, 1, 1, 1)

        self.upconv1 = upconvbnrelu(1024, 512, 4, 2, 1, 1)
        self.conv8 = convbnrelu(1025, 512, 3, 1, 1, 1)

        # pr5 + loss5
        self.conv9 = convbnrelu(512, 1, 3 ,1 ,1 ,1)

        self.upconv2 = upconvbnrelu(512, 256, 4, 2, 1, 1)
        self.conv10 = convbnrelu(769, 256, 3, 1, 1, 1)

        # pr4 + loss4
        self.conv11 = convbnrelu(256, 1, 3, 1, 1, 1)

        self.upconv3 = upconvbnrelu(256, 128, 4, 2, 1, 1)
        self.conv12 = convbnrelu(385, 128, 3, 1, 1, 1)

        # pr3 + loss3
        self.conv13 = convbnrelu(128, 1, 3, 1, 1, 1)

        self.upconv4 = upconvbnrelu(128, 64, 4, 2, 1, 1)
        self.conv14 = convbnrelu(193, 64, 3, 1, 1, 1)

        # pr2 + loss2
        self.conv15 = convbnrelu(64, 1, 3, 1, 1, 1)

        self.upconv5 = upconvbnrelu(64, 32, 4, 2, 1, 1)
        self.conv16 = convbnrelu(97, 32, 3, 1, 1, 1)

        # pr1 + loss1
        self.conv17 = convbnrelu(32, 1, 3, 1, 1, 1)

    def forward(self, x):
        output = []

        encode_1 = self.conv1(x)
        encode_2 = self.conv2(encode_1)
        encode_3 = self.conv3b(self.conv3a(encode_2))
        encode_4 = self.conv4b(self.conv4a(encode_3))
        encode_5 = self.conv5b(self.conv5a(encode_4))

        decode_0 = self.conv6b(self.conv6a(encode_5))
        output_0 = self.conv7(decode_0)
        output.append(output_0)

        decode_1 = self.conv8(self.upconv1(decode_0) + output_0 + encode_5)
        output_1 = self.conv9(decode_1)
        output.append(output_1)

        decode_2 = self.conv10(self.upconv2(decode_1) + output_1 + encode_4)
        output_2 = self.conv11(decode_2)
        output.append(output_2)

        decode_3 = self.conv12(self.upconv3(decode_2) + output_2 + encode_3)
        output_3 = self.conv13(decode_3)
        output.append(output_3)

        decode_4 = self.conv14(self.upconv4(decode_3) + output_3 + encode_4)
        output_4 = self.conv15(decode_4)
        output.append(output_4)

        decode_5 = self.conv16(self.upconv5(decode_4) + output_4 + encode_5)
        output_5 = self.conv17(decode_5)
        output.append(output_5)

        return output


if __name__ == "__main__":
    model = DispNet()
    print(model)