import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data.dataset as Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys,os
from inspect import isclass
import math
from utils import flow_utils, tools
from flowsd import FlowNet2SD
import models
from datainput import Datainput
import argparse
from tensorboardX import SummaryWriter
from losses import MultiScale
from utils import tools
TAG_CHAR = np.array([202021.25], np.float32)

def writeFlow(filename, uv, v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()




# parsing the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--file',  help= 'Csv file Path', required=True)
parser.add_argument('--is_cropped',help='if you want to train with cropped frames', default=False)
parser.add_argument('--dataset', help=' Name of dataset' , required=True)
parser.add_argument('--batch_size',type = int, default=1, help= 'Enter the batch size')
parser.add_argument('--epoch', type=int, default=5, help=" number of epoch")
parser.add_argument('--lrate', type=float, default=0.0001, help=" examples: --lrate 0.002")
parser.add_argument("--rgb_max", type=float, default=255.)

args = parser.parse_args()




#specifying the name of summary file
# writer = SummaryWriter('loss/'+ str(args.dataset) +'/train_'+ str(args.batch_size) + '_'+ str(args.lrate) +'_256_256_crop' +str(args.dataset))

#data inputs
data_opt_img = Datainput(args.file, args.is_cropped)
train_loader = torch.utils.data.DataLoader(data_opt_img,batch_size=args.batch_size, shuffle=False)
#
# # model definition, loss, optimizer
flownet2 = FlowNet2SD(args)
model_path = '../flownet2-pytorch/pretrained/FlowNet2-SD_checkpoint.pth.tar'
pretrained_dict = torch.load(model_path)['state_dict']
flownet2.load_state_dict(pretrained_dict)
flownet2.cuda()
flownet2.train()

# MSE = torch.nn.MSELoss()
multiloss = MultiScale(args)
optimizer = torch.optim.Adam(flownet2.parameters(), lr = args.lrate)
iteration = 0
total_loss = 0

for epoch in range(args.epoch):
    for batch_number, batch in enumerate(train_loader):
        iteration = iteration + 1

        inFrame = batch['inputFrame']
        target = batch['targetFlow']

        output = flownet2(inFrame)
        losses = multiloss(output,target)

        print(' Dataset: {}  Loss: {}  Batch_size: {} Epoch: {}  Iteration: {}   Remaning Epoch: {} Learning Rate: {} '.format(args.dataset, losses[0].item(), args.batch_size, epoch+1, iteration, args.epoch-1 - epoch, args.lrate))
        # writer.add_scalar('train_loss_'+ str(args.dataset) , losses[0].item(), iteration)

        losses = [torch.mean(loss_value) for loss_value in losses]
        loss_val = losses[0]
        # Zero the gradients
        optimizer.zero_grad()

        # Backpropagation
        loss_val.backward()

        # update the parameters
        optimizer.step()


        if batch_number == 50:
            break

    break

        # if iteration % 2000 == 0:
        #     torch.save(flownet2, 'checkpoints/'+ str(args.dataset) + '/batch_'+ str(args.batch_size) + '_epoch_' + str(epoch) + '_Iteration_' + str(iteration) +'_' +str(args.lrate) + '_384_640_crop' +str(args.dataset)+ '.pth.tar')
        #
        #
