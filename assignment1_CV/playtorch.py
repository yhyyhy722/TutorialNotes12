import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

m = nn.Linear(20, 30)
input = torch.randn(128, 20)
import ipdb; ipdb.set_trace()
output = m(input)
#(output.shape)

a = torch.randn(1,3,224,224)
# a = a.flatten(start_dim=1)
# a = a.reshape(1,3*224*224)

# a.contiguous()
# a = a.view(1,3,224*224)

b = torch.randn(1,3,224,224)
c=torch.stack((a,b),dim=1)
print(c.shape)
'''
class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                #nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(128,256, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                #nn.MaxPool2d(kernel_size=2, stride=2)
            )
        ])
        self.layer3 = nn.ModuleDict({
            'layer1': nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                #nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            'layer2': nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                #nn.MaxPool2d(kernel_size=2, stride=2)
            )
        })     

    def forward(self, x):
        x = self.layer1(x)
        for layer in self.layer2:
            x = layer(x)
        x = self.layer3['layer1'](x)
        x = self.layer3['layer2'](x)
        return x
    
input = torch.randn(1,3,224,224)
net = myNet()
output = net(input)
print(output.shape)
'''
