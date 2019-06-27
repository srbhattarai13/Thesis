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

# parsing the arguments
parser = argparse.ArgumentParser()
parser.add_argument('optFlow', help=' Path to the video testing folder')
parser.add_argument('dataset', help=' Name of dataset')

args = parser.parse_args()

dataset_dir = args.optFlow
dataset = args.dataset



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


def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)

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
MSE = torch.nn.MSELoss()
L1 = torch.nn.L1Loss()

writer = SummaryWriter('logs/Check_loss_Inference')

# loading model
models_list= ['NET_batch_2_epoch_42_0.0001_Adam_WithNorm_ReSized.pt']

for model in models_list:

    mdl = torch.load('checkpoints/' + dataset + '/' + model)
    mdl.eval()
    mdl.cuda()
    # print("model list")
    # extracting the videos names
    video_names = sorted(os.listdir(dataset_dir + 'optical_flow_resize/'))

    # extracting each frames of every videos
    for vid in range(len(video_names)):

        opts = sorted(os.listdir(str(dataset_dir) + 'optical_flow_resize/' + video_names[vid] + '/'))

        vid1 = vid+1
        print('%02d'%vid1)

        for k in range(0,1):
            for i in range(0, len(opts)-3):
                # print(optflow_names)
                opt_tensor1 = loadFlow(dataset_dir + 'optical_flow_resize/' + video_names[vid] + '/' + opts[i])
                opt_tensor2 = loadFlow(dataset_dir + 'optical_flow_resize/' + video_names[vid] + '/' + opts[i+1])
                opt_tensor3 = loadFlow(dataset_dir + 'optical_flow_resize/' + video_names[vid] + '/' + opts[i+2])

                opt_target = loadFlow(dataset_dir + 'optical_flow_resize/' + video_names[vid] + '/' + opts[i+3])


                third_tensor = torch.cat((opt_tensor1, opt_tensor2), 0)
                inflow_tensor = torch.cat((third_tensor, opt_tensor3), 0)
                inflow_tensor = torch.unsqueeze(inflow_tensor, 0)

                nmTn = normalizedData(inflow_tensor)

                print(torch.max(nmTn))

                print(torch.max(dnmt))

                # print('Normalization Max: {} Denormalization: { }'.format(torch.max(nmTn), torch.max(DenormalizedData(nmTn))))



                # print(inflow_tensor.size())
                # predFlow = mdl(normalizedData(inflow_tensor))
                #
                # loss = MSE(predFlow,normalizedData(opt_target))
                # L1loss = L1(predFlow,normalizedData(opt_target))

                # print(' MSE LOSS: {} L1Loss: {} '.format(loss.item(), L1loss.item()))
                # writer.add_scalar('train_loss_', loss.item(), i)
                # writer.add_scalar('L1loss', L1loss.item())

                # if i == 1434:
                break



            # predFlow = predFlow.squeeze()
            # predFlow = predFlow.permute(1,2,0)

            # save predicted optical flow
            # saveFlow = predFlow.cpu().data.numpy().squeeze()
            #
            # if not os.path.exists('/media/data/Shreeram/datasets/avenue/testing/Prediction_resize_withNorm/prediction_flo_resize_withNorm/01_using_inference_resize/' + '%2d'%vid1):
            #     os.makedirs('/media/data/Shreeram/datasets/avenue/testing/Prediction_resize_withNorm/prediction_flo_resize_withNorm/01_using_inference_resize/' + '%2d'%vid1)
            #
            # writeFlow('/media/data/Shreeram/datasets/avenue/testing/Prediction_resize_withNorm/prediction_flo_resize_withNorm/01_using_inference_resize/' + '/%04d.flo'%i , saveFlow)
            # # print('Saved successfully {}th frame'.format(i))
            # #+ '%2d'%vid1 +
            # # break

        break


