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
from code.unet import Unet
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


# loading the frames
def load_frame(filename, resize_height=256, resize_width=256):
    img = Image.open(filename)
    img_resized = img.resize((resize_width, resize_height),Image.ANTIALIAS)
    img_numpy = np.array(img_resized)
    img_numpy = img_numpy.astype(dtype=np.float32)
    img_numpy = (img_numpy / 127.5) - 1.0
    inImg = torch.from_numpy(img_numpy)

    return inImg.permute(2,0,1)

def loadFlow(name):
    Flow = readFlow(name)
    Flow = torch.from_numpy(Flow)
    Flow = Flow.permute(2,0,1)

    optVar = torch.autograd.Variable(Flow.cuda())
    return optVar

def Normalize(tensor):
    mean = (0.0484, 0.0043)
    std = (0.4076, 0.0948)

    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)

    return tensor

def deNormalize(tensor):
    mean = (0.0484, 0.0043)
    std = (0.4076, 0.0948)

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    return tensor

def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)

# MSE = nn.MSELoss(reduction='none')
# MSE = nn.MSELoss()


# models_list = sorted(os.listdir('checkpoints/avenue/newtraining/'))
models_list = ['NET_batch_15_epoch_3_0.0001_RGB_Flow_tanh_BatchNorm.pt']



for model in models_list:
    # writer = SummaryWriter('logs/Avenue/Test/test_l1loss_' + str(model).strip('.pt'))
    mdl = torch.load('checkpoints/' + dataset + '/newtraining/' + model)
    mdl.eval()
    mdl.cuda()
    video_names = sorted(os.listdir(dataset_dir + 'frames/'))

    # extracting each frames of every videos
    lossVidWise = []
    k = 0
    numHis = 4
    for vid in range(len(video_names)):
        opts = sorted(os.listdir(str(dataset_dir) + 'frames/' + video_names[vid] + '/'))
        vid1 = vid+1

        # if video_names[vid] == '01':
        #     continue

        lossFrameWise = np.empty(shape=(len(opts) , ), dtype=np.float32)


        for i in range(numHis, len(opts)-1):
            k += 1
            frame1 = load_frame(dataset_dir + 'frames/' + video_names[vid] + '/' + opts[i-4])
            frame2 = load_frame(dataset_dir + 'frames/' + video_names[vid] + '/' + opts[i-3])
            frame3 = load_frame(dataset_dir + 'frames/' + video_names[vid] + '/' + opts[i-2])
            frame4 = load_frame(dataset_dir + 'frames/' + video_names[vid] + '/' + opts[i-1])


            frame12 = torch.cat((frame1, frame2), 0)
            frame34 = torch.cat((frame3, frame4), 0)
            inputFrame = torch.cat((frame12, frame34), 0)
            inputFrame = torch.unsqueeze(inputFrame, 0)


            #
            optflow_name = '/%04d.flo' % i
            targetFlow = loadFlow(dataset_dir + 'optical_flow_resize/01/' + optflow_name)
            targetFlow = Normalize(targetFlow)
            targetFlow = torch.unsqueeze(targetFlow,0)



            # saving predicted flow
            predictFlow = mdl(inputFrame.cuda())

            predictFlow = predictFlow.squeeze()
            predictFlow = deNormalize(predictFlow)


            predictFlow = predictFlow.permute(1, 2, 0)
            predictFlow = predictFlow.detach()
            tosave = predictFlow.cpu().data.numpy().squeeze()
            print(tosave.size)

            print('Saved Successfully for vide: {}  size:{}  frame: {} '.format(vid, tosave.size, i))
            # save prediction
            writeFlow('/media/data/Shreeram/datasets/avenue/testing/Prediction/rgbflow/flow/batchnorm/01/' + '/%04d.flo' % i,tosave)


        break








