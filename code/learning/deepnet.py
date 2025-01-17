import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F # don't understand why we use Functional

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # input channel, out channels, 5x5 convolution kernel
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16, 5)

        # an affine operation y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)



    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        print("x",x)

        x = F.max_pool2d(F.relu(self.conv2(x)),2)

        x = x.view(-1, self.num_float_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)


        return x




    def num_float_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features

net = Net()

print(net)
# print(net)
# param = list(net.parameters())
# print(len(param))
# print(param[0].size())

#
# input = torch.randn(1,1,32,32)
# out = net(input)
# print(out)
#
# print(out.size())
#
#
#
# target = torch.randn(10)
# target = target.view(1,-1) # make it same shape as output
#
# criterion = nn.MSELoss()
#
# print(target.size())
#
# loss = criterion(out,target)
# print(loss)
#
#
# print(loss.grad_fn)
#
# print(loss.grad_fn.next_functions[0][0])
#
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])
#
#
# net.zero_grad()
#
# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)
#
# loss.backward()
#
# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)
#





