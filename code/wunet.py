from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F


# def doubleConv(input_channel, output_filters, kernel_size=3, stride=1, padding=1):

class Unet(nn.Module):
    def __init__(self, layers, input_channel=2, features_root=64, filter_size=3, pool_size=2, stride=1):
        super(Unet, self).__init__()


        self.layers = layers
        self.feature_root = features_root
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.input_channel = input_channel
        self.stride = stride

        # Convoluational layers
        self.downConv = nn.Sequential(
                                            nn.Conv2d(self.input_channel,self.features, self.filter_size, self.stride),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.input_channel, self.features, self.filter_size, self.stride)
                                     )

        self.upConv = nn.Sequential(
                                        nn.Conv2d(self.input_channel, self.features, self.filter_size, self.stride),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(self.input_channel,self.features, self.filter_size, self.stride),
                                        nn.ReLU(inplace=True)
                                   )

        self.MaxPool = nn.MaxPool2d(kernel_size=pool_size, stride=2)

    def forward(self, x, input_channel=2, features_root=64, filter_size=3, pool_size=2, stride=1):

        x = self.downConv()
        x = self.downconv1(x)
        x = self.drelu(x)
        x = self.downconv2(x)
        c1 = self.drelu(x)
        x = self.dmaxpool(c1)


        x = self.downconv3(x)
        x = self.drelu(x)
        x = self.downconv4(x)
        c2 = self.drelu(x)
        x  = self.dmaxpool(c2)

        x = self.downconv5(x)
        x = self.drelu(x)
        x = self.downconv6(x)
        c3 = self.drelu(x)
        x = self.dmaxpool(c3)

        x = self.downconv7(x)
        x = self.drelu(x)
        x = self.downconv8(x)
        x = self.drelu(x)


        x = self.trans(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([c3, x], dim=1)

        #
        x = self.upconv1(x)
        x = self.urelu(x)
        x = self.upconv2(x)
        x = self.urelu(x)

        x = self.trans1(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([c2, x], dim=1)
        #

        x = self.upconv3(x)
        x = self.urelu(x)
        x = self.upconv4(x)
        x = self.urelu(x)

        x = self.trans2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([c1, x], dim=1)

        x = self.upconv5(x)
        x = self.urelu(x)
        x = self.upconv6(x)
        x = self.urelu(x)


        return x

# #
# #
# mdl = Unet()
#
# input = torch.ones([1,2,256,256], dtype=torch.float32)
#
# print(input.size())
#
# # out = mdl(input)
# out = mdl(input)
# print("Outsize:", out.size())
