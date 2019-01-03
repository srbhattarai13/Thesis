import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data.dataset as Dataset
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


from utils import Datainput
from u-net import Unet
import argparse


from tensorboardX import SummaryWriter



# parsing the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--file',  help= 'Csv file Path', required=True)
parser.add_argument('--dataset', help=' Name of dataset' , required=True)
# parser.add_argument('--training', type= bool, help="Type True for Pretrained, False: for Scratch Model")
parser.add_argument('--batch_size',type = int, default=10, help= 'Enter the batch size', required=True)
parser.add_argument('--epoch', type=int, default=5, help=" number of epoch")
parser.add_argument('--lrate', type=float, default=0.01, help=" examples: --lrate 0.002")
# parser.add_argument('--name', help=" Type the name extension", required=True)
args = parser.parse_args()


#specifying the name of summary file
writer = SummaryWriter('logs/'+ str(args.dataset) +'/train_'+ str(args.batch_size) + '_'+ str(args.lrate))


#data inputs
data_opt_img = Datainput(args.file)
train_loader = torch.utils.data.DataLoader(data_opt_img,batch_size=args.batch_size, shuffle=True)


# model definition, loss, optimizer
unet = Unet()
unet.train() # Training mode
unet = unet.cuda()
MSE = torch.nn.MSELoss()
optimizer = torch.optim.Adam(unet.parameters(), 0.001)

iteration = 0

for epoch in range(args.epoch):
    for batch_number, batch in enumerate(train_loader):

        iteration = iteration + 1

        predFlow = unet(batch['frame'])
        gtFlow = batch['flow']

        # compute the loss
        loss = MSE(predFlow, gtFlow)

        print(' Dataset: {}  Loss: {}  Batch_size: {} Epoch: {}  Iteration: {}   Remaning Epoch: {} Learning Rate: {} '.format(args.dataset, loss.item(), args.batch_size, epoch+1, iteration, args.epoch-1 - epoch, args.lrate))

        writer.add_scalar('train_loss_'+ str(args.dataset) + '_' + str(args.name) , loss.item(), iteration)

        # Zero the gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # update the parameters
        optimizer.step()


    if epoch % 2 == 0:
        torch.save(unet.state_dict(), 'checkpoints/'+ str(args.dataset) + '/NET_batch_'+ str(args.batch_size) + '_epoch_' + str(epoch)+ '_' + str(args.name) +str(args.lrate)+ '.pt')

