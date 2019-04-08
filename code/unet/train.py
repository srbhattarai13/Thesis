import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data.dataset as Dataset
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import sys,os

# from .utils import Datainput
# from .unet import Unet

from unet import Unet
from utils import Datainput
import argparse

from tensorboardX import SummaryWriter

# Running command
# python code/train.py --file data/traingAVENUE.csv --dataset Avenue

# parsing the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--file',  help= 'Csv file Path', required=True)
parser.add_argument('--dataset', help=' Name of dataset' , required=True)
# parser.add_argument('--training', type= bool, help="Type True for Pretrained, False: for Scratch Model")
parser.add_argument('--batch_size',type = int, default=15, help= 'Enter the batch size')
parser.add_argument('--epoch', type=int, default=60, help=" number of epoch")
parser.add_argument('--lrate', type=float, default=0.0001, help=" examples: --lrate 0.002")
# parser.add_argument('--name', help=" Type the name extension", required=True)
args = parser.parse_args()

#specifying the name of summary file
writer = SummaryWriter('logs/'+ str(args.dataset) +'/train_'+ str(args.batch_size) + '_'+ str(args.lrate) +'_RGB_OptFlow_relu_BatchNorm')

#data inputs
data_opt_img = Datainput(args.file)
train_loader = torch.utils.data.DataLoader(data_opt_img,batch_size=args.batch_size, shuffle=True)

# model definition, loss, optimizer
unet = Unet()

torch.cuda.empty_cache()
unet = unet.cuda()
MSE = torch.nn.MSELoss()
# l1loss = torch.nn.L1Loss()
optimizer = torch.optim.Adam(unet.parameters(), lr = args.lrate)
# optimizer = torch.optim.Optimizer(unet.parameters())

# optimizer = torch.optim.SGD(unet.parameters(),lr=args.lrate)#, momentum=0.9, weight_decay =0.0005)

iteration = 0

for epoch in range(args.epoch):
    for batch_number, batch in enumerate(train_loader):
        unet.train()  # Training mode

        inFrame = batch['inputFrame']
        nextFlow = batch['targetFlow']

        # predicting the next flow
        predFlow = unet(inFrame)

        # compute the loss
        loss = MSE(predFlow, nextFlow)

        print(' Dataset: {}  Loss: {}  Batch_size: {} Epoch: {}  Iteration: {}   Remaning Epoch: {} Learning Rate: {} '.format(args.dataset, loss.item(), args.batch_size, epoch+1, iteration, args.epoch-1 - epoch, args.lrate))
        writer.add_scalar('train_loss_'+ str(args.dataset) , loss.item(), iteration)

        iteration = iteration + 1

        # Zero the gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # update the parameters
        optimizer.step()


    if epoch % 3 == 0:
        torch.save(unet, 'checkpoints/'+ str(args.dataset) + '/newtraining/NET_batch_'+ str(args.batch_size) + '_epoch_' + str(epoch) + '_' +str(args.lrate) + '_RGB_Flow_relu_BatchNorm'+ '.pt')

