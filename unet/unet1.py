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

        # self.dmaxpool = nn.MaxPool2d(kernel_size=pool_size, stride=2)

        #downsampple
        self.downconv1_l1 = nn.Conv2d(12, 64, kernel_size=kernel_size, stride=1, padding=1)
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
        self.bnl3_2 = nn.BatchNorm2d(256)

        self.downconv1_l4 = nn.Conv2d(256, 512, kernel_size=kernel_size, stride=1, padding=1)
        self.bnl4_1 = nn.BatchNorm2d(512)
        self.downconv2_l4 = nn.Conv2d(512, 512, kernel_size=kernel_size, stride=1, padding=1)
        self.bnl4_2 = nn.BatchNorm2d(512)


        # upsample
        self.deconv_l1  =  nn.ConvTranspose2d(512, 256, pool_size, stride=pool_size)
        self.upconv1_l1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bnl5_1 = nn.BatchNorm2d(256)
        self.upconv2_l1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bnl5_2 = nn.BatchNorm2d(256)


        self.deconv_l2  =  nn.ConvTranspose2d(256, 128, pool_size, stride=pool_size)
        self.upconv1_l2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bnl6_1 = nn.BatchNorm2d(128)
        self.upconv2_l2 = nn.Conv2d(128, 128,  kernel_size=3, stride=1, padding=1)
        self.bnl6_2 = nn.BatchNorm2d(128)


        self.deconv_l3 =  nn.ConvTranspose2d(128, 64, pool_size, stride=pool_size)
        self.upconv1_l3 = nn.Conv2d(128,64, kernel_size=3, stride=1, padding=1)
        self.bnl7_1 = nn.BatchNorm2d(64)

        self.upconv2_l3= nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bnl7_2 = nn.BatchNorm2d(64)

        self.final_conv_l3 = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)



    def forward(self, x):
        x  = F.relu(self.bnl1_1(self.downconv1_l1(x)))
        c1 = F.relu(self.bnl1_2(self.downconv2_l1(x)))
        x  = F.max_pool2d(c1, (2,2))

        x  = F.relu(self.bnl2_1(self.downconv1_l2(x)))
        c2 = F.relu(self.bnl2_2(self.downconv2_l2(x)))
        x  = F.max_pool2d(c2, (2,2))

        x  = F.relu(self.bnl3_1(self.downconv1_l3(x)))
        c3 = F.relu(self.bnl3_2(self.downconv2_l3(x)))
        x  = F.max_pool2d(c3, (2,2))

        x = F.relu(self.bnl4_1(self.downconv1_l4(x)))
        x = F.relu(self.bnl4_2(self.downconv2_l4(x)))
        x = self.deconv_l1(x)

        x = torch.cat([c3, x], dim=1)
        x = F.relu(self.bnl5_1(self.upconv1_l1(x)))
        x = F.relu(self.bnl5_2(self.upconv2_l1(x)))
        x = self.deconv_l2(x)

        x = torch.cat([c2, x], dim=1)
        x = F.relu(self.bnl6_1(self.upconv1_l2(x)))
        x = F.relu(self.bnl6_2(self.upconv2_l2(x)))
        x = self.deconv_l3(x)

        x = torch.cat([c1, x], dim=1)
        x = F.relu(self.bnl7_1(self.upconv1_l3(x)))
        x = F.relu(self.bnl7_2(self.upconv2_l3(x)))

        x = F.relu(self.final_conv_l3(x))

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
