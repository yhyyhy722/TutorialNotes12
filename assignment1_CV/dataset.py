import torch
import torch.nn as nn
from glob import glob
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class FlowerDataset(Dataset):
    def __init__(self, data_file, transform=None):
        data_file = open(data_file, 'r')
        self.data_list = data_file.readlines()[1:]
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img = self.data_list[idx].split(',')[0]
        label = torch.tensor(int(self.data_list[idx].split(',')[1])).long()
        img = Image.open('flower_dataset/'+img).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label