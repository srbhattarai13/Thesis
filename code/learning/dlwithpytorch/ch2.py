import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn

import torchvision.models




# inp = Variable(torch.randn(1,10))


# mylayer = nn.Linear(in_features=10, out_features=5, bias=True)
#
# print(mylayer(inp))
#
# print(mylayer.weight)
# print('Bias', mylayer.bias)
#
# sample_data = Variable(torch.Tensor([[1,2,-1,-1]]))
#
# myrelu = nn.ReLU()
# print(myrelu(sample_data))
#

#
# class MyFirstNetwork(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MyFirstNetwork,self).__init()
#         self.layer1 = nn.Linear(input_size,hidden_size)
#         self.layer2 = nn.Linear(hidden_size,output_size)
#
#
#     def forward(self,input):
#         out = self.layer1(input)
#         out = nn.ReLU(out)
#         out = self.layer2(out)
#         return out
#
#
#
# loss = nn.MSELoss()
# input = Variable(torch.randn(3,5),requires_grad=True)
# target = Variable(torch.randn(3,5))
# output = loss(input,target)
#
# output.backward()
#
#
# optimizer = optim.SGD(model.parameters(), lr=0.01)
# for input, target in dataset:
#     optimizer.zero_grad()
#     output = model(input)
#     loss = loss_fn(output, target)
#     loss.backward()
#     optimizer.step()
#
#
#


'''

glob = return all the files in the particular path
iglob = return iterator instead of loading names into memory


shuffle = np.random.permutation(no_of_images)

# creating a validation set

os.mkdir(os.path.join(path,'valid'))
for t in ['train', valid']:
    for folder in ['dog/','cat/']:
        os.mkdir(os.path.join(path,t,folder))
        
        
        
for i in shuffle[:2000]:
    folder = files[i].split('/')[-1].split('.')[0]
    image = files[i].split('/')[-1]
    os.rename(files[i], os.path.join(path,'valid', folder, image))
    
    
# loading data into PyTorch tensor

ImageFolder -> load images along with their associated labels when data is presented in the aforementioned format


simple_transform = transform.Compose([transforms.Scale((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
                                    
                                    
train = ImageFolder('dosandcats/train/', simple_transform)
valid = ImageFolder('dogsandcats/valid',simple_transform



visualizing the Image tensor

def imshow(inp):
    #imshow for tensor
    inp = np.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    
    
# loading PyTorch tensors as batches
train_data_gen = torch.utils.data.DataLoader(train, batch_size=64, num_workers=3)
valid_data_gen = torch.utils.data.DataLoader(valid, batch_size=64, num_workers=3)





# training
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {} / {}'.format(epoch, num_epochs - 1 ))
        print('-' * 10)
        
        # Each epoch has a traiing and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True) # set model to training mode
            else:
                model.train(False) # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                
                # wrap them in Variable
                if is_cuda:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                    
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward outputs = model(inputs)
                _, preds = torch.max(outputs.data,1)
                loss = criterion(outputs, labels)
                
                #backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]
                
                
                print('{} loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                
                # deep copy the model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    
        print()
        
        
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60 , time_elapsed % 60))
        print('Best val Acc: {:.4f}'.format(best_acc))
        
        # load best model weights
        model.load_state_dict(best_model_wts)
        
        return model
        


'''

model_ft = models.resnet18(pretrained=True)
num_fts = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs,2)
