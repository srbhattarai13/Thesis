import torch
import numpy as np
from torch.autograd import Variable

import matplotlib.pyplot as plt
from PIL import Image
import glob

# x = torch.rand(10)
# print(x)
# print('size of x', x.size())
#
#
# tm = torch.FloatTensor([23,24,25,27827])
# print(tm.size())

#from_numpy() # converts a numpy array into a torch tensor
#
# panda = np.array(Image.open('code/learning/dlwithpytorch/sr.jpg').resize((224,224)))
# panda_tensor = torch.from_numpy(panda)
# print(panda_tensor.size())
#
#
# # Image._show(panda)
# plt.imshow(panda_tensor[50:200, 60:130,0].numpy())
#
# sales = torch.eye(3,3)
# sales[0,1]
# plt.show()
#

#
# data_path = '/media/data/Shreeram/Downloads/'
# # read cat images fromd isk
# cats = glob(data_path +'*.jpg')
#
# print(cats)

# Convert images into numpy arraygs
# cat_imags = np.array([np.array(Image.open(cat).resize((224,224))) for cat in cats[:64]])
#
# cat_imgs = cat_imgs.reshape(-1,224,224,3)
# cat_tensors = torch.from_numpy(cat_imgs)
# print(cat_tensors.size())
#


#gradients
#
# x = Variable(torch.ones(2,2), requires_grad=True)
#
# y = x.mean()
#
# y.backward()
#
# print(x.grad, x.grad_fn)
#
#
# print(y.grad_fn)



# Creating data for our neural Network

def get_data():
    train_x = np.asarray([3.3,4.4.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
    train_y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])


    dtype = torch.FloatTensor
    X = Variable(torch.from_numpy(train_x).type(dtype),requires_grad=False).view(17,1)
    Y = Variable(torch.from_numpy(train_y).type(dtype),requires_grad=False)

    return X,Y

def get_weights():
    w = Variable(torch.randn(1), requires_grad=True)
    b = Variable(torch.randn(1), requires_grad=True)

    return w,b


# network implementation
def simple_network(x):
    y_pred = torch.matmul(x,w) + b
    return y_pred


# loss function
def loss_fn(y,y_pred):
    loss = (y_pred-y).pow(2).sum()
    for param in [w,b]:
        if not param.grad is None: param.grad.data.zero_()

    loss.backward()

    return loss.data[0]



def optimize(learning_rate):
    w.data -= learning_rate * w.grad.data
    b.data -= learning_rate * b.grad.data


# dataset class
from torch.utils.data import Dataset
class DogsAndCatsDataset(Dataset):
    def __init__(self, root_dir, size=(224,224)):
        self.files = glob(root_dir)
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.asarray(Image.open(self.files[idx]).resize(self.size))
        label = self.files[idx].split('/')[-2]

        return img,label



#dataloader class

Dataloader = DataLoader(dogsdset,batc_size=32,num_workers=2)
for img,labels in dataloader:
    # applying DL
    pass





