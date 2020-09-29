import torch
import torchvision
from torchvision import models
from torch.hub import load_state_dict_from_url
import torch.nn as nn
from inceptionV3 import InceptionV3PyTorch

# Define the architecture by modifying resnet.
# Original code is here
# https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py
class InceptionV3TransferPyTorch(nn.Module):
    def __init__(self, num_classes=3):
        super(InceptionV3TransferPyTorch, self).__init__()
        self.layer1 = models.inception_v3(num_classes=1000, pretrained=True, aux_logits=True)
        #self.layer1 = InceptionV3PyTorch(pretrained=True, aux_logits=True)
        self.layer2 = nn.Sequential(nn.Linear(1000, 512))
        self.fc = nn.Linear(512, num_classes)

    # Reimplementing forward pass.
    # Replacing the following code
    # https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py#L197-L213
    def forward(self, x):
        print('x=>',x.shape)
        x = self.layer1(x)
        print('out1=>',x[0].shape)
        x = self.layer2(x[0])
        print('out2=>',x.shape)
        sizeList = list(x.shape)
        print('len(sizeList)=>',len(sizeList))
        if len(sizeList) > 1:
            x = torch.flatten(x, 1)
        print('x=>',x.shape)
        return x

    def forwardEval(self, x):
        print('x=>',x.shape)
        x = self.layer1(x)
        print('out1=>',x.shape)
        x = self.layer2(x)
        print('out2=>',x.shape)
        sizeList = list(x.shape)
        print('len(sizeList)=>',len(sizeList))
        if len(sizeList) > 1:
            x = torch.flatten(x, 1)
        print('x=>',x.shape)
        return x