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

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.downconv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.downconv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.drelu1    = nn.ReLU(inplace=True)
        self.dmaxp1    = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downconv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.downconv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.drelu2    = nn.ReLU(inplace=True)
        self.dmaxp2    = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downconv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.downconv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.drelu3    = nn.ReLU(inplace=True)
        self.dmaxp3    = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downconv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.downconv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.drelu4    = nn.ReLU(inplace=True)


        # upsample

        self.trans = nn.ConvTranspose2d(512,256, kernel_size=3, stride=1, padding=1)
        self.upconv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.urelu1  = nn.ReLU(inplace=True)

        self.trans1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upconv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upconv4 = nn.Conv2d(128, 128,  kernel_size=3, stride=1, padding=1)
        self.urelu2  = nn.ReLU(inplace=True)

        self.trans2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.upconv5 = nn.Conv2d(128,64, kernel_size=3, stride=1, padding=1)
        self.upconv6 = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        self.urelu3  = nn.ReLU(inplace=True)



    def forward(self, x):
        x = self.downconv1(x)
        x = self.downconv2(x)
        c1 = self.drelu1(x)
        x = self.dmaxp1(c1)


        x = self.downconv3(x)
        x = self.downconv4(x)
        c2 = self.drelu2(x)
        x  = self.dmaxp2(c2)

        x = self.downconv5(x)
        x = self.downconv6(x)
        c3 = self.drelu3(x)
        x = self.dmaxp3(c3)

        x = self.downconv7(x)
        x = self.downconv8(x)
        x = self.drelu4(x)


        x = self.trans(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([c3, x], dim=1)

        #
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.urelu1(x)

        x = self.trans1(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([c2, x], dim=1)
        #

        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.urelu2(x)

        x = self.trans2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([c1, x], dim=1)

        x = self.upconv5(x)
        x = self.upconv6(x)
        x = self.urelu3(x)


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
