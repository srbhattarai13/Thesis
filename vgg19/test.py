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
from scipy.misc import imread,imresize



img_norm = transforms.Compose([transforms.Resize((224,224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
parser = argparse.ArgumentParser()
parser.add_argument('--frames', help=' Path to the video testing folder')
parser.add_argument('dataset', help=' Path to the video testing folder')


args = parser.parse_args()

dataset_dir = args.frames
dataset = str(args.dataset)

MSE = nn.MSELoss(reduction='none')
# MSE = nn.MSELoss()
models_list = sorted(os.listdir('checkpoints/shanghaitech' +'/'))
numHis = 1
for model in models_list:
    # writer = SummaryWriter('logs/Avenue/Test/test_l1loss_' + str(model).strip('.pt'))
    mdl = torch.load('checkpoints/shanghaitech/' + model)
    mdl.eval()
    mdl.cuda()


    # get video's name
    video_names = sorted(os.listdir(str(dataset_dir) + 'frames/'))

    lossVidWise = []

    for vid in range(len(video_names)):
        vid1 = vid+1
        frames = sorted(os.listdir(str(dataset_dir) +'frames/' + video_names[vid] + '/'))


        lossFrameWise = np.empty(shape=(len(frames) , ), dtype=np.float32)

        for i in range(numHis,len(frames)):
            image = Image.open(str(dataset_dir) +'frames/' + video_names[vid] + '/' + frames[i-numHis])

            image = img_norm(image).float()
            image = Variable(image).cuda()
            image = image.unsqueeze(0)

            features = mdl(image) #.cpu().data

            targetFeats = torch.from_numpy(np.load(dataset_dir + 'vggfeatures/' + video_names[vid]  + "/%04d.npy" % (i)))

            # loss = MSE(features, targetFeats.cuda())


            # top 5 % MSE Loss
            loss = MSE(features, targetFeats.cuda())
            loss = loss.squeeze()
            channelAvg = loss.mean(0)
            flat_loss = channelAvg.view(-1)
            sortLoss = torch.sort(flat_loss, descending=True)
            top5Loss = sortLoss[0][0:int(flat_loss.size()[0] * 0.05)]
            finalLoss = top5Loss.mean()

            # printing the progress
            print(" Model: {} Video: {} Frame: {}  Target Features: {}  loss: {} ".format(model,video_names[vid], frames[i-numHis], "%04d.npy" %(i), finalLoss.item()) )

            lossFrameWise[i] = finalLoss.item()

        # coping the values for first  and last frame
        lossFrameWise[0:numHis] = lossFrameWise[numHis]
        lossFrameWise[len(frames)-1] = lossFrameWise[len(frames)-2]

        # Normalization
        lossFrameWise -= lossFrameWise.min()
        lossFrameWise /= lossFrameWise.max()

        # Appending video level scores
        lossVidWise.append(1- lossFrameWise)


    # saving the result
    testResult = {'dataset': dataset, 'score': lossVidWise}

    with open('result/' + dataset + '/' + str(model).strip('.pth.tar') , 'wb') as writer:
        pickle.dump(testResult, writer, pickle.HIGHEST_PROTOCOL)

    # evaluating the result
    results = evaluate('compute_auc', 'result/' + dataset + '/' + str(model).strip('.pth.tar'))

    # saving a result of different trained model in text file
    with open('data/test_result_MSE_vgg_'  + dataset +'.txt', 'a') as f:
        f.write(str(model).strip('.pth.tar') + ', ' + str(results) + ', top5%MSE' + '\n')






















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





               