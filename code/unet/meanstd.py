import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


from utils1 import Datainput

dataset = Datainput('data/ped1_allframes.csv')
loader = DataLoader(dataset, batch_size=10, num_workers=1, shuffle=False)

mean = 0.
std = 0.
nb_samples = 0.


for k,data in enumerate(loader):

    print(data.size())

    break
#     batch_samples = data.size(0)
#     data = data.view(batch_samples, data.size(1), -1)
#     mean += data.mean(2).sum(0)
#     std += data.std(2).sum(0)
#     nb_samples += batch_samples
#
#     print('Mean: {} std: {} '.format(mean, std))
#
#
#
# mean /= nb_samples
# std /= nb_samples
#
# print('Mean: {}, Std: {}'.format(mean, std))


# class MyDataset(Dataset):
#     def __init__(self):
#         self.data = torch.randn(200, 2, 360, 640)
#
#     def __getitem__(self, index):
#         x = self.data[index]
#         return x
#
#     def __len__(self):
#         return len(self.data)
#
# dataset = MyDataset()
# loader = DataLoader(dataset, batch_size=10, num_workers=1, shuffle=False)
#
# mean = 0.
# std = 0.
# nb_samples = 0.
#
# for data in loader:
#     batch_samples = data.size(0)
#     data = data.view(batch_samples, data.size(1), -1)
#     mean += data.mean(2).sum(0)
#     std += data.std(2).sum(0)
#     nb_samples += batch_samples
#
#     print('Mean: {}, Std: {}'.format(mean, std))
#
#
#
# mean /= nb_samples
# std /= nb_samples
#
# print('final Mean: {}  Std: {} '.format(mean, std))
