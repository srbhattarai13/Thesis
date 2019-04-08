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
        nn.init.xavier_uniform_(self.downconv1.weight)
        # self.batchNorm = nn.BatchNorm2d(self.down)
        self.downconv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.downconv2.weight)
        self.drelu    = nn.ReLU(inplace=True)
        self.dmaxpool    = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downconv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.downconv3.weight)
        self.downconv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.downconv4.weight)

        # self.drelu    = nn.ReLU(inplace=True)
        # self.dmaxp2    = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downconv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.downconv5.weight)

        self.downconv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.downconv6.weight)

        # self.drelu3    = nn.ReLU(inplace=True)
        # self.dmaxp3    = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downconv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.downconv7.weight)

        self.downconv8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.downconv8.weight)

        # self.drelu4    = nn.ReLU(inplace=True)


        # upsample


        self.urelu  = nn.ReLU(inplace=True)

        # self.trans = nn.ConvTranspose2d(512,256, kernel_size=3, stride=1, padding=1)
        self.upconv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.upconv1.weight)

        self.upconv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.upconv1.weight)

        self.trans1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upconv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.upconv3.weight)
        self.upconv4 = nn.Conv2d(128, 128,  kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.upconv4.weight)
        # self.urelu2  = nn.ReLU(inplace=True)

        self.trans2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.upconv5 = nn.Conv2d(128,64, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.upconv5.weight)
        self.upconv6 = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.upconv6.weight)
        # self.urelu3  = nn.ReLU(inplace=True)



    def forward(self, x):
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


        # x = self.trans(x)
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
