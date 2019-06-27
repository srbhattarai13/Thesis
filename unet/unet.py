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
from torchsummary import summary

torch.nn.Module.dump_patches = True

class Unet(nn.Module):
    def __init__(self, pool_size=2, kernel_size=3):
        super(Unet, self).__init__()

        # self.relu = nn.ReLU(inplace=True)
        self.dmaxpool = nn.MaxPool2d(kernel_size=pool_size, stride=2)

        #downsampple
        self.downconv1_l1 = nn.Conv2d(6, 64, kernel_size=kernel_size, stride=1, padding=1)
        self.bnl1_1 = nn.BatchNorm2d(64)
        self.downconv2_l1 = nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=1)
        self.bnl1_2 = nn.BatchNorm2d(64)

        self.downconv1_l2 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, padding=1)
        self.bnl2_1 = nn.BatchNorm2d(128)
        self.downconv2_l2 = nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=1)
        self.bnl2_2 = nn.BatchNorm2d(128)

        self.downconv1_l3 = nn.Conv2d(128, 256, kernel_size=kernel_size, stride=1, padding=1)
        self.bnl3_1 = nn.BatchNorm2d(256)
        self.downconv2_l3 = nn.Conv2d(256, 256, kernel_size=kernel_size, stride=1, padding=1)
        self. bnl3_2 = nn.BatchNorm2d(256)

        self.downconv1_l4 = nn.Conv2d(256, 512, kernel_size=kernel_size, stride=1, padding=1)
        self.bnl4_1 = nn.BatchNorm2d(512)
        self.downconv2_l4 = nn.Conv2d(512, 512, kernel_size=kernel_size, stride=1, padding=1)
        self.bnl4_2 = nn.BatchNorm2d(512)

        # upsample
        self.deconv_l1  =  nn.ConvTranspose2d(512, 256, pool_size, stride=pool_size)
        self.upconv1_l1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.upconv2_l1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.deconv_l2  =  nn.ConvTranspose2d(256, 128, pool_size, stride=pool_size)
        self.upconv1_l2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upconv2_l2 = nn.Conv2d(128, 128,  kernel_size=3, stride=1, padding=1)

        self.deconv_l3 =  nn.ConvTranspose2d(128, 64, pool_size, stride=pool_size)
        self.upconv1_l3 = nn.Conv2d(128,64, kernel_size=3, stride=1, padding=1)
        self.upconv2_l3= nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.final_conv_l3 = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)



    def forward(self, x):
        x  = self.bnl1_1(self.downconv1_l1(x))
        c1 = self.bnl1_2(self.downconv2_l1(x))
        x  = self.dmaxpool(c1)

        x  = self.bnl2_1(self.downconv1_l2(x))
        c2 = self.bnl2_2(self.downconv2_l2(x))
        x  = self.dmaxpool(c2)

        x  = self.bnl3_1(self.downconv1_l3(x))
        c3 = self.bnl3_2(self.downconv2_l3(x))
        x  = self.dmaxpool(c3)

        x = self.bnl4_1(self.downconv1_l4(x))
        x = self.bnl4_2(self.downconv2_l4(x))
        x = self.deconv_l1(x)

        x = torch.cat([c3, x], dim=1)
        x = self.upconv1_l1(x)
        x = self.upconv2_l1(x)
        x = self.deconv_l2(x)

        x = torch.cat([c2, x], dim=1)
        x = self.upconv1_l2(x)
        x = self.upconv2_l2(x)
        x = self.deconv_l3(x)

        x = torch.cat([c1, x], dim=1)
        x = self.upconv1_l3(x)
        x = self.upconv2_l3(x)
        x = self.final_conv_l3(x)
        #
        x = torch.tanh(x)

        return x


# # #
# mdl = Unet()
# mdl = mdl.cuda(0)
#
#
# # input = torch.ones([1,6,360,640], dtype=torch.float32)
#
# # input = input.cuda(0)
# # print(input.size())
#
# # out = mdl(input)
# # out = mdl(input)
# # print("Outsize:", out.size())
# summary(mdl, (6,360,640))