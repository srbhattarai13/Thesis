import argparse
import os
import numpy as np
import torchvision.models as models
import torch.nn as nn
import scipy.misc as misc
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

class vggFeats(nn.Module):
    def __init__(self, train=False):
        super(vggFeats, self).__init__()

        vgg19Model = models.vgg19(pretrained=train)

        new_features = nn.Sequential(*list(vgg19Model.features.children())[:-1])

        self.features = new_features

    def forward(self, x):
        x = self.features(x)

        return x

img_norm = transforms.Compose([transforms.Resize((224,224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
parser = argparse.ArgumentParser()
parser.add_argument('--frames', help=' Path to the video testing folder')

args = parser.parse_args()

dataset_dir = args.frames

vggModel = vggFeats()
vggModel.eval()
vggModel.cuda()

video_names = sorted(os.listdir(dataset_dir +'frames/'))

for vid in range(len(video_names)):
    frames = sorted(os.listdir(str(dataset_dir) +'frames/' + video_names[vid] + '/'))

    for i in range(len(frames)):
        image = Image.open(str(dataset_dir) +'frames/' + video_names[vid] + '/' + frames[i])
        image = img_norm(image).float()
        image = Variable(image).cuda()
        image = image.unsqueeze(0)

        features = vggModel(image).cpu().data


        if not os.path.exists(dataset_dir  +'vggfeatures/' + video_names[vid]):
            os.makedirs(dataset_dir  +'vggfeatures/' + video_names[vid])
        #
        np.save(dataset_dir +'vggfeatures/'  + video_names[vid] + '/%04d' % i, features)
        print(dataset_dir +'vggfeatures/'  + video_names[vid] + '/%04d' % i)




