import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import os
import argparse
import time
from PIL import Image
from tensorboardX import SummaryWriter
from unet import Unet
import pickle
import evaluate

# parsing the arguments
parser = argparse.ArgumentParser()
parser.add_argument('optFlow', help=' Path to the video testing folder')
parser.add_argument('dataset', help=' Name of dataset')

args = parser.parse_args()

dataset_dir = args.optFlow
dataset = str(args.dataset)


from tensorboardX import SummaryWriter


def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:, :, 0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()
    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2 ))

    return flow.astype(np.float32)


def loadFlow(name):
    Flow = readFlow(name)
    Flow = torch.from_numpy(Flow)
    Flow = Flow.permute(2,0,1)

    optVar = torch.autograd.Variable(Flow.cuda())
    return optVar

def normalizedData(inTensor):
    inTensor[inTensor > 20.0 ] = 20.0
    inTensor[ inTensor < -20.0] = -20.0
    inTensor = torch.div(inTensor, 20.0)
    return inTensor



unet = Unet()
unet = unet.cuda()
MSE = torch.nn.MSELoss()
optimizer = torch.optim.SGD(unet.parameters(), 0.0001)
epochs = 60

writer = SummaryWriter('logs/'+ str(args.dataset) +'_trained_manual_unsq_sgd')

video_names = sorted(os.listdir(dataset_dir + 'optical_flow/'))

numHis = 4
k = 0
lr = 0.0001
for epoch in range(0,epochs):

    for vid in range(len(video_names)):

        opts = sorted(os.listdir(str(dataset_dir) + 'optical_flow/' + video_names[vid] + '/'))

        for i in range(numHis, len(opts)):
            k += 1
            opt_tensor1 = loadFlow(dataset_dir + 'optical_flow/' + video_names[vid] + '/' + opts[i-4])
            opt_tensor2 = loadFlow(dataset_dir + 'optical_flow/' + video_names[vid] + '/' + opts[i-3])
            opt_tensor3 = loadFlow(dataset_dir + 'optical_flow/' + video_names[vid] + '/' + opts[i-2])
            opt_tensor4 = loadFlow(dataset_dir + 'optical_flow/' + video_names[vid] + '/' + opts[i-1])

            opt_target = loadFlow(dataset_dir + 'optical_flow/' + video_names[vid] + '/' + opts[i])
            opt_target = torch.unsqueeze(opt_target,0)
            opt_target = normalizedData(opt_target)
            opt_target = torch.autograd.Variable(opt_target)  # torch.from_numpy(nextFlow)
            opt_target = opt_target.cuda()

            mergedTensor1 = torch.cat((opt_tensor1, opt_tensor2), 0)
            mergedTensor2 = torch.cat((opt_tensor3, opt_tensor4), 0)

            inputFlow = torch.cat((mergedTensor1, mergedTensor2), 0)
            inputFlow = torch.unsqueeze(inputFlow, 0)
            inputFlow = normalizedData(inputFlow)
            inputFlow = torch.autograd.Variable(inputFlow)  # torch.from_numpy(prvFlow)
            inputFlow = inputFlow.cuda()


            prdctFlow = unet(inputFlow)

            loss = MSE(prdctFlow, opt_target)

            print('Dataset: {}  Loss: {}  Epoch: {}  Iteration: {}   Remaning Epoch: {} Learning Rate: {} '.format(args.dataset, loss.item(), epoch + 1, k, epochs - 1, lr))
            writer.add_scalar('train_loss_' + str(args.dataset), loss.item(), k)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if epoch % 3 == 0:
        torch.save(unet, 'checkpoints/'+ str(args.dataset) + '/trained_manual/NET_batch_'+ '_epoch_' + str(epoch) + '_' +str(lr) + 'MSE_trainedManual_SGD'+ '.pt')







