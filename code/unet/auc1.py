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


# optimizer = torch.optim.Adam()
# L1 = torch.nn.L1Loss()
MSE = torch.nn.MSELoss()


# loading model
# models_list= ['NET_batch_10_epoch_9_0.0001_Adam_CrtNorm_Org.pt']

models_list = sorted(os.listdir('checkpoints/avenue/l1loss/'))
# models_list = ['NET_batch_10_epoch_3_0.0001MSE_4numHis.pt']

l1loss = nn.L1Loss()

for model in models_list:
    # writer = SummaryWriter('logs/Avenue/Test/test_l1loss_' + str(model).strip('.pt'))
    print(model)
    mdl = torch.load('checkpoints/' + dataset + '/l1loss/' + model)
    mdl.eval()
    mdl.cuda()
    video_names = sorted(os.listdir(dataset_dir + 'optical_flow/'))

    # extracting each frames of every videos
    lossVidWise = []
    k = 0
    numHis = 3
    for vid in range(len(video_names)):
        opts = sorted(os.listdir(str(dataset_dir) + 'optical_flow/' + video_names[vid] + '/'))
        vid1 = vid+1

        lossFrameWise = np.empty(shape=(len(opts) + 1, ), dtype=np.float32)


        for i in range(numHis, len(opts)):
            k += 1
            opt_tensor1 = loadFlow(dataset_dir + 'optical_flow/' + video_names[vid] + '/' + opts[i-3])
            opt_tensor2 = loadFlow(dataset_dir + 'optical_flow/' + video_names[vid] + '/' + opts[i-2])
            opt_tensor3 = loadFlow(dataset_dir + 'optical_flow/' + video_names[vid] + '/' + opts[i-1])

            opt_target = loadFlow(dataset_dir + 'optical_flow/' + video_names[vid] + '/' + opts[i])
            opt_target = normalizedData(opt_target)
            opt_target = torch.unsqueeze(opt_target,0)

            mergedTensor1 = torch.cat((opt_tensor1, opt_tensor2), 0)

            inputFlow = torch.cat((mergedTensor1, opt_tensor3), 0)
            inputFlow = torch.unsqueeze(inputFlow, 0)
            inputFlow = normalizedData(inputFlow)


            prdctFlow = mdl(inputFlow)

            loss = MSE(prdctFlow, opt_target)

            # writer.add_scalar('test_loss_' + dataset, 1- loss.item(), k)

            lossFrameWise[i] = loss.item()


            print(' Model: {} Dataset:{}  Video {} Frame: {}  MSELoss : {} TotalModels: {} l1loss'.format(str(model).strip('.pt'), dataset, vid1,str(opts[i]).strip('.flo'),  1 - loss.item(), len(models_list)))



        lossFrameWise[0:numHis] = lossFrameWise[numHis]

        # lossFrameWise[len(opts)] = lossFrameWise[len(opts)-1]



        lossFrameWise -= lossFrameWise.min()
        lossFrameWise /= lossFrameWise.max()

        lossVidWise.append(1 - lossFrameWise)

    testResult = {'dataset': dataset, 'score':lossVidWise}

    with open('results/'+ dataset + '/L1_MSE_cr/result_' + str(model).strip('.pt') , 'wb') as writer:
        pickle.dump(testResult, writer, pickle.HIGHEST_PROTOCOL)





    # auc = []
    results = evaluate.evaluate('compute_auc', 'results/'+ dataset + '/L1_MSE_cr/result_' + str(model).strip('.pt'))
    # auc.append(results)

    print(results)

    # # auc = { 'model': str(model).strip('.pt'), 'auc': results}
    with open('data/test_results_L1_MSE_cr.txt', 'a') as f:
        f.write(str(model).strip('.pt') + ', ' + str(results )+ '\n')







