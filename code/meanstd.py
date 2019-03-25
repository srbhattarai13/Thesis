import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


from utils import Datainput

dataset = Datainput('data/allframes.csv')
loader = DataLoader(dataset, batch_size=10, num_workers=1, shuffle=True)

mean = 0.
std = 0.
nb_samples = 0.

for data in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    print(data.size())

    break
    # mean  = data.mean()
    # print(mean(2))


