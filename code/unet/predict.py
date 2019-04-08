import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import time
from PIL import Image
from tensorboardX import SummaryWriter
from unet import Unet
from utils import Datainput

import pickle

# parsing the arguments
parser = argparse.ArgumentParser()
parser.add_argument('optFlow', help=' Path to optical flow of testing videos')
parser.add_argument('dataset', help=' Name of dataset')
args = parser.parse_args()

dataset_dir = args.optFlow
dataset = args.dataset


def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)


# loading the data points
testSet = Datainput(dataset_dir)
test_loader = torch.utils.data.DataLoader(testSet, batch_size=1, shuffle=False)


# loading model
models_list = ['NET_batch_10_epoch_18_0.0001_Adam_CrtNorm_Org.pt']
for model in models_list:
    # mdl.load_state_dict(torch.load('checkpoints/' + dataset + '/' + model))
    mdl = torch.load('checkpoints/' + dataset + '/' + model)
    mdl.eval()
    mdl.cuda()

    i = -1
    # print(len(test_loader))
    for k, test in enumerate(test_loader):

        if test['vidName'].item() == 2:
            i = i + 1

            # print(test['inFlow'].size())
            predFlow = mdl(test['inFlow'])
            predFlow = predFlow.squeeze()
            predFlow = predFlow * 20 # Denormalized
            predFlow = predFlow.permute(1, 2, 0)
            tosaveFlow = predFlow.detach()
            tosaveFlow = tosaveFlow.cpu().data.numpy().squeeze()


            print('Saved Successfully {}th Frame '.format(i))

            # save prediction
            writeFlow('/media/data/Shreeram/datasets/avenue/testing/Prediction/Prediction_Org_size/flow/02_epoch18_withDeNorm/' + '%04d.flo'%i , tosaveFlow)






