import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.misc import imread
import torch
from torch.autograd import Variable
from scipy.misc import imresize
from math import ceil
from math import floor
import matplotlib.pyplot as plt
from flowsd import FlowNet2SD

from PIL import Image
import argparse, os, sys, subprocess
UNKNOWN_FLOW_THRESH = 1e7

TAG_CHAR = np.array([202021.25], np.float32)

divisor = 64.
Scale_width = 0.9375
Scale_height= 1.0
org_h = 360
org_w = 640

def make_color_wheel():
  """
  Generate color wheel according Middlebury color code
  :return: Color wheel
  """
  RY = 15
  YG = 6
  GC = 4
  CB = 11
  BM = 13
  MR = 6

  ncols = RY + YG + GC + CB + BM + MR

  colorwheel = np.zeros([ncols, 3])

  col = 0

  # RY
  colorwheel[0:RY, 0] = 255
  colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
  col += RY

  # YG
  colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
  colorwheel[col:col + YG, 1] = 255
  col += YG

  # GC
  colorwheel[col:col + GC, 1] = 255
  colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
  col += GC

  # CB
  colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
  colorwheel[col:col + CB, 2] = 255
  col += CB

  # BM
  colorwheel[col:col + BM, 2] = 255
  colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
  col += + BM

  # MR
  colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
  colorwheel[col:col + MR, 0] = 255

  return colorwheel

def compute_color(u, v):
  """
  compute optical flow color map
  :param u: optical flow horizontal map
  :param v: optical flow vertical map
  :return: optical flow in color code
  """
  [h, w] = u.shape
  img = np.zeros([h, w, 3])
  nanIdx = np.isnan(u) | np.isnan(v)
  u[nanIdx] = 0
  v[nanIdx] = 0

  colorwheel = make_color_wheel()
  ncols = np.size(colorwheel, 0)

  rad = np.sqrt(u ** 2 + v ** 2)

  a = np.arctan2(-v, -u) / np.pi

  fk = (a + 1) / 2 * (ncols - 1) + 1

  k0 = np.floor(fk).astype(int)

  k1 = k0 + 1
  k1[k1 == ncols + 1] = 1
  f = fk - k0

  for i in range(0, np.size(colorwheel, 1)):
    tmp = colorwheel[:, i]
    col0 = tmp[k0 - 1] / 255
    col1 = tmp[k1 - 1] / 255
    col = (1 - f) * col0 + f * col1

    idx = rad <= 1
    col[idx] = 1 - rad[idx] * (1 - col[idx])
    notidx = np.logical_not(idx)

    col[notidx] *= 0.75
    img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

  return img


def flow_to_image(flow, display=False):
  """
  Convert flow into middlebury color code image
  :param flow: optical flow map
  :return: optical flow image in middlebury color
  """
  u = flow[:, :, 0]
  v = flow[:, :, 1]

  maxu = -999.
  maxv = -999.
  minu = 999.
  minv = 999.

  idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
  u[idxUnknow] = 0
  v[idxUnknow] = 0

  maxu = max(maxu, np.max(u))
  minu = min(minu, np.min(u))

  maxv = max(maxv, np.max(v))
  minv = min(minv, np.min(v))

  rad = np.sqrt(u ** 2 + v ** 2)
  maxrad = max(-1, np.max(rad))

  if display:
    print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu, maxu, minv, maxv))

  u = u / (maxrad + np.finfo(float).eps)
  v = v / (maxrad + np.finfo(float).eps)

  img = compute_color(u, v)

  idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
  img[idx] = 0

  return np.uint8(img)


def writeFlow(filename, uv, v=None):
    """ Write optical flow to file.
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()

def np_load_frame(frame1, frame2):
    '''
    :param frame1: First frame's path
    :param frame2: second frame's path
    :param crop_size: by default (256, 256)
    :return: tensor of stacked frames
    '''
    img0 = imread(frame1)
    img1 = imread(frame2)

    im1_resized = imresize(img0,crop_size)
    im2_resized = imresize(img1,crop_size)

    ims = np.array([[im1_resized, im2_resized]]).transpose((0, 4, 1, 2, 3)).astype(np.float32)
    print(ims.shape)
    ims_tensor = torch.from_numpy(ims)

    ims_v = Variable(ims_tensor.cuda(), requires_grad=True)

    return ims_v


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', help=' Path to the video testing folder')
    parser.add_argument("--rgb_max", type=float, default=255.)

    args = parser.parse_args()
    dataset_dir = args.frames


    #loading the model
    flownet2 = FlowNet2SD(args)
    path = '../flownet2-pytorch/pretrained/FlowNet2-SD_checkpoint.pth.tar'
    pretrained_dict = torch.load(path)['state_dict']
    # model_dict = flownet2.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    flownet2.load_state_dict(pretrained_dict)
    flownet2.eval()
    flownet2.cuda()

    video_names = sorted(os.listdir(dataset_dir + 'frames/'))

    for vid in range(len(video_names)):

        if os.path.exists(dataset_dir + 'optflow_norm/' + video_names[vid]):
            continue


        frames = sorted(os.listdir(str(dataset_dir) + 'frames/' + video_names[vid] + '/'))

        for i in range(len(frames)-1):

            # loading frames
            frame1_path = str(dataset_dir) + 'frames/' + video_names[vid] + '/' + frames[i]
            frame2_path = str(dataset_dir) + 'frames/' + video_names[vid] + '/' + frames[i+1]

            # printing the progress
            print('Datasets: {} Video: {} Frame1:{} Frame2:{} '.format(str(dataset_dir).split('/')[3],video_names[vid], frames[i], frames[i+1]))
            inputFrame = np_load_frame(frame1_path, frame2_path)

            prediction = flownet2(inputFrame).cpu().data
            prediction = prediction[0].numpy().transpose((1,2,0))


            if not os.path.exists(dataset_dir + 'optflow_norm/' + video_names[vid]):
                os.makedirs(dataset_dir + 'optflow_norm/' + video_names[vid])

            writeFlow(dataset_dir + 'optflow_norm/' + video_names[vid] + '/%04d.flo' % i,prediction)

            # flow_im = flow_to_image(prediction)
            # image = Image.fromarray(flow_im, 'RGB')
            # image = image.resize((int(scale_width * adapted_width), int(scale_height * adapted_height)),Image.ANTIALIAS)
            # image.save('imgs/tst/checkGarekoe_'+str(i) +'.png')

            # break

            # if i ==20:
            #     break
        # break