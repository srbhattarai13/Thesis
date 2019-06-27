import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data.dataset as Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys,os

from datainput import Datainput
import argparse
from tensorboardX import SummaryWriter
from vgg19 import vggFeats


# parsing the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--file',  help= 'Csv file Path', required=True)
parser.add_argument('--dataset', help=' Name of dataset' , required=True)
parser.add_argument('--batch_size',type = int, default=32, help= 'Enter the batch size')
parser.add_argument('--epoch', type=int, default=10, help=" number of epoch")
parser.add_argument('--lrate', type=float, default=0.0001, help=" examples: --lrate 0.002")
args = parser.parse_args()

#specifying the name of summary file
writer = SummaryWriter('logs/'+ str(args.dataset) +'/train_'+ str(args.batch_size) + '_'+ str(args.lrate) +'_vgg19_nexPred_Again' +str(args.dataset))

#data inputs
data_opt_img = Datainput(args.file)
train_loader = torch.utils.data.DataLoader(data_opt_img,batch_size=args.batch_size, shuffle=True)
#
# # model definition, loss, optimizer
vgg19 = vggFeats()
vgg19.train()
vgg19.cuda()

MSE = torch.nn.MSELoss()
optimizer = torch.optim.Adam(vgg19.parameters(), lr = args.lrate)
iteration = 0
for epoch in range(args.epoch):
    for batch_number, batch in enumerate(train_loader):
        iteration = iteration + 1

        inFrame = batch['inputFrame']
        nextFeats = batch['targetFeats']

        predFeats = vgg19(inFrame)

        # compute the loss
        loss = MSE(predFeats, nextFeats)


        print(' Dataset: {}  Loss: {}  Batch_size: {} Epoch: {}  Iteration: {}   Remaning Epoch: {} Learning Rate: {} '.format(args.dataset, loss.item(), args.batch_size, epoch+1, iteration, args.epoch-1 - epoch, args.lrate))
        writer.add_scalar('train_loss_'+ str(args.dataset) , loss.item(), iteration)


        # Zero the gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # update the parameters
        optimizer.step()



        if iteration % 1000 == 0:
            torch.save(vgg19, 'checkpoints/'+ str(args.dataset) + '/batch_'+ str(args.batch_size) + '_epoch_' + str(epoch) + '_Iteration_' + str(iteration) +'_' +str(args.lrate) + '_vgg_nextPred_Feats_again' +str(args.dataset)+ '.pth.tar')


