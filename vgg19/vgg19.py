import torchvision.models as models
import torch.nn as nn
import scipy.misc as misc
import torch
import matplotlib.pyplot as plt

import numpy

from torchvision.utils import make_grid


class vggFeats(nn.Module):
    def __init__(self, train=False):
        super(vggFeats, self).__init__()

        vgg19Model = models.vgg19(pretrained=train)

        new_features = nn.Sequential(*list(vgg19Model.features.children())[:-1])

        self.features = new_features

        self.predConv = nn.Conv2d(512,512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.predRelu = nn.ReLU(inplace=True)



    def forward(self, x):
        x = self.features(x)
        x = self.predRelu(self.predConv(x))

        return x

# vgg = vggFeats()
#
#
# # for k, v in vgg.state_dict().items():
# #     print(k)
#
# for layer in vgg.modules():
#     print(layer)

# #
# # visualizing the filter
# filter = vgg.features[0].weight.detach().clone()
#
# filter = filter - filter.min()
# filter = filter / filter.max()
#
# img = make_grid(filter)
#
# plt.imshow(img.permute(1,2,0))
# plt.show()
#
# #
# #













































# Code sample for using visualdl
# from visualdl import LogWriter
#
#
# from random import random
#
# logw = LogWriter("./random_log", sync_cycle=10000)
#
# # create scalars in mode train and test
# with logw.mode("Train") as logger:
#     scalar0 = logger.scalar("scratch/scalar")
#
#
# with logw.mode("Test") as logger:
#     scalar1 = logger.scalar("scratch/scalar")
#
#
#
# # add scalar records
# for step in range(200):
#     scalar0.add_record(step,step*1. /200)
#     scalar1.add_record(step, 1. -step* 1. /200)
