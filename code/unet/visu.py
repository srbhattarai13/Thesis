
import torch
from unet import Unet



model = Unet()
model.load_state_dict(torch.load('checkpoints/Avenue/NET_batch_10_epoch_0_0.0001.pt'))

torch.save(model, 'saved_model.pt')
