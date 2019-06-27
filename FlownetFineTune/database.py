import torch
import torch.utils.data as data

import os, math, random
from os.path import *
import numpy as np

from glob import glob
import utils.frame_utils as frame_utils

from scipy.misc import imread, imresize


class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1 + self.th), self.w1:(self.w1 + self.tw), :]


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size

    def __call__(self, img):
        return img[(self.h - self.th) // 2:(self.h + self.th) // 2, (self.w - self.tw) // 2:(self.w + self.tw) // 2, :]



class ImagesFromFolder(data.Dataset):
    def __init__(self, args, is_cropped = True, root='/path/to/frames/only/folder', iext='jpg', replicates=1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        images = sorted(glob(join(root, '*.' + iext)))
        self.image_list = []
        for i in range(len(images) - 1):
            im1 = images[i]
            im2 = images[i + 1]
            self.image_list += [[im1, im2]]

        self.size = len(self.image_list)

        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0] % 64) or (
                self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0]) // 64) * 64
            self.render_size[1] = ((self.frame_size[1]) // 64) * 64

        args.inference_size = self.render_size

    def __getitem__(self, index):
        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        images = [img1, img2]
        image_size = img1.shape[:2]
        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))

        images = np.array(images).transpose(3, 0, 1, 2)
        images = torch.from_numpy(images.astype(np.float32))

        return [images], [torch.zeros(images.size()[0:1] + (2,) + images.size()[-2:])]

    def __len__(self):
        return self.size * self.replicates


