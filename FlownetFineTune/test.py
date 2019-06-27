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
import pickle
from evaulate import evaluate
from flowsd import FlowNet2SD
from scipy.misc import imread,imresize
# parsing the arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset', help=' Path to the video testing folder')
parser.add_argument('name', help=' Name of dataset')
parser.add_argument("--rgb_max", type=float, default=255.)


args = parser.parse_args()

dataset_dir = args.dataset
dataset = str(args.name)

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


# loading the frames
def np_load_frame(frame1, frame2, width = 256, height = 256):
    im1 = imread(frame1)
    im2 = imread(frame2)

    im1 = imresize(im1, (height, width))
    im2 = imresize(im2, (height, width))
    ims = np.array([[im1, im2]]).transpose((0, 4, 1, 2, 3)).astype(np.float32)


    ims_tensor = torch.from_numpy(ims)
    # Tensor Normalization
    ims_normalize = (ims_tensor / 127.5) - 1.0
    ims_v = Variable(ims_normalize.cuda(), requires_grad=True)

    return ims_v

def loadFlow(name):
    Flow = readFlow(name)
    Flow = torch.from_numpy(Flow)
    Flow = Flow.permute(2,0,1)

    optVar = torch.autograd.Variable(Flow.cuda())
    return optVar

def Normalize(tensor):
    # mean & std of Avenue
    # mean = (0.0484, 0.0043)
    # std = (0.4076, 0.0948)

    # Mean & std of ShanghaiTech
    mean = (0.0484, 0.0043)
    std = (0.4076, 0.0948)
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)

    return tensor

def deNormalize(tensor):
    # # Mean & Std of Avenue
    # mean = (0.0484, 0.0043)
    # std = (0.4076, 0.0948)

    # Mean & std of ShanghaiTech
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




MSE = nn.MSELoss(reduction='none')
# MSE = nn.MSELoss()
models_list = sorted(os.listdir('checkpoints/'  + dataset +'/'))
# models_list = ['batch_64_epoch_1_Iteration_6000_0.0001_fnet_nextPict_ShTech.pth.tar']
for model in models_list:
    # writer = SummaryWriter('logs/Avenue/Test/test_l1loss_' + str(model).strip('.pt'))
    mdl = torch.load('checkpoints/'  + dataset +'/' + model)
    mdl.cuda()
    mdl.eval()

    video_names = sorted(os.listdir(dataset_dir + 'testing/frames/'))

    # extracting each frames of every videos
    lossVidWise = []
    k = 0
    numHis = 1
    for vid in range(len(video_names)):
        opts = sorted(os.listdir(str(dataset_dir) + 'testing/frames/' + video_names[vid] + '/'))
        vid1 = vid+1



        lossFrameWise = np.empty(shape=(len(opts) , ), dtype=np.float32)


        for i in range(numHis, len(opts)-1):
            k += 1
            inputFrame = np_load_frame(dataset_dir + 'testing/frames/' + video_names[vid] + '/' + opts[i], dataset_dir + 'testing/frames/' + video_names[vid] + '/' + opts[i+1])

            optflow_name = '/%04d.flo' % i
            targetFlow = readFlow(dataset_dir + 'testing/optical_flow_resize/' + video_names[vid]  + optflow_name)
            targetFlow_tensor = torch.from_numpy(targetFlow)
            targetFlow_tensor = Normalize(targetFlow_tensor)
            targetFlow_arranged = targetFlow_tensor.permute(2, 0, 1)

            predictFlow = mdl(inputFrame)

            #top 5 % MSE Loss
            loss = MSE(predictFlow, targetFlow_arranged.cuda())
            loss = loss.squeeze()
            channelAvg = loss.mean(0)
            flat_loss = channelAvg.view(-1)
            sortLoss = torch.sort(flat_loss, descending=True)
            top5Loss = sortLoss[0][0:int(flat_loss.size()[0] * 0.05)]
            finalLoss = top5Loss.mean()




            lossFrameWise[i] = 1- finalLoss.item()
            print(' Model: {} Dataset:{}  Video {} Frame1: {} Frame2: {}  MSELoss : {} TotalModels: {}'.format(str(model).strip('.pt'), dataset, vid1, str(opts[i]).strip('.flo'), str(opts[i+1]), 1 - finalLoss.item(), len(models_list)))

        # coping the values for first 2 frames and last 2 frames
        lossFrameWise[0:numHis] = lossFrameWise[numHis]
        lossFrameWise[len(opts)-1] = lossFrameWise[len(opts)-2]

        # Normalization
        lossFrameWise -= lossFrameWise.min()
        lossFrameWise /= lossFrameWise.max()

        # Appending video level scores
        lossVidWise.append(lossFrameWise)


    testResult = {'dataset': dataset, 'score': lossVidWise}

    with open('result/' + dataset + '/' + str(model).strip('.pth.tar') , 'wb') as writer:
        pickle.dump(testResult, writer, pickle.HIGHEST_PROTOCOL)

    # # auc = []
    results = evaluate('compute_auc', 'result/' + dataset + '/' + str(model).strip('.pth.tar'))

# results = evaluate('compute_auc', 'result/shanghaitech/batch_64_epoch_1_Iteration_6000_1e-05_fnet_nextPict_ShTec')
    # auc.append(results)
#
# print(results)

#
# with open('data/test_result_shanghaitech.txt', 'a') as f:
#     f.write('shanghaitech, shanghaitech/batch_64_epoch_1_Iteration_6000_1e-05_fnet_nextPict_ShTec, ' + str(results) + '\n')
    # # auc = { 'model': str(model).strip('.pt'), 'auc': results}
    with open('data/test_result_MSE'  + dataset +'.txt', 'a') as f:
        f.write(str(model).strip('.pth.tar') + ', ' + str(results) + ', MSEtop5%' + '\n')






















             #
            # mergedTensor1 = torch.cat((opt_tensor1, opt_tensor2), 0)
            # mergedTensor2 = torch.cat((opt_tensor3, opt_tensor4), 0)
            #
            # inputFlow = torch.cat((mergedTensor1, mergedTensor2), 0)
            # inputFlow = torch.unsqueeze(inputFlow, 0)
            #
            #
            # prdctFlow = mdl(inputFlow)

'''
            prdctFlow = prdctFlow.squeeze()
            prdctFlow = deNormalize(prdctFlow)
            prdctFlow = prdctFlow.permute(1, 2, 0)
            prdctFlow = prdctFlow.detach()
            tosaveFlow = prdctFlow.cpu().data.numpy().squeeze()

            print('Saved Successfully for vide: {}  {}th Frame '.format(vid, i))

            # save prediction
            writeFlow('/media/data/Shreeram/datasets/avenue/testing/Prediction/meanstd/flow/' + '%04d.flo' % i, tosaveFlow)
        break
             For highest top 5% from MSE loss

            loss = MSE(prdctFlow, opt_target)
            loss = loss.squeeze()
            chn_averaged = loss.mean(0)
            flat_loss = chn_averaged.view(-1)
            sorted_loss = torch.sort(flat_loss, descending=True)
            top5 = sorted_loss[0][0:int(flat_loss.size()[0] * 0.05)]
            loss = top5.mean()

            lossFrameWise[i] = loss.item()


            print(' Model: {} Dataset:{}  Video {} Frame: {}  MSELoss : {} TotalModels: {}'.format(str(model).strip('.pt'), dataset, vid1,str(opts[i]).strip('.flo'),  1 - loss.item(), len(models_list)))

        lossFrameWise[0:numHis] = lossFrameWise[numHis]

        # lossFrameWise[len(opts)] = lossFrameWise[len(opts)-1]

        lossFrameWise -= lossFrameWise.min()
        lossFrameWise /= lossFrameWise.max()

        lossVidWise.append(1 - lossFrameWise)

    testResult = {'dataset': dataset, 'score':lossVidWise}

    with open('results/'+ dataset + '/4numhis_tanh/tanh_result_top5_' + str(model).strip('.pt') , 'wb') as writer:
        pickle.dump(testResult, writer, pickle.HIGHEST_PROTOCOL)

    # # auc = []
    results = evaluate.evaluate('compute_auc', 'results/'+ dataset + '/4numhis_tanh/tanh_result_top5_' + str(model).strip('.pt'))
    # # auc.append(results)

    print(results)

    # # auc = { 'model': str(model).strip('.pt'), 'auc': results}
    with open('data/test_results_4numHis_tanh_top5.txt', 'a') as f:
        f.write(str(model).strip('.pt') + ', ' + str(results) + '\n')
#
# print("Max auc: ", max(auc))


'''





