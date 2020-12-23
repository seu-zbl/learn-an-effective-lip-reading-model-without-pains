import glob
import os

from .cvtransforms import *

import torch
from torch.utils.data import Dataset


class LRWDataset(Dataset):
    def __init__(self, phase, args):

        with open('label_sorted.txt') as myfile:
            self.labels = myfile.read().splitlines()
        
        self.list = []
        self.phase = phase
        self.args = args
        
        if not hasattr(self.args, 'is_aug'):
            setattr(self.args, 'is_aug', True)

        for (i, label) in enumerate(self.labels):
            files = glob.glob(os.path.join('lrw_roi_80_116_175_211_npy_gray_pkl', label, phase, '*.pkl'))
            files = sorted(files)

            self.list += [file for file in files]

    def __getitem__(self, idx):
        tensor = torch.load(self.list[idx])
        inputs = tensor.get('video') / 255.0

        if self.phase == 'train':
            batch_img = RandomCrop(inputs, (88, 88))
            batch_img = HorizontalFlip(batch_img)
        elif self.phase == 'val' or self.phase == 'test':
            batch_img = CenterCrop(inputs, (88, 88))
        else:
            raise ValueError('train, val or test is expected!')
        
        result = {'video': torch.FloatTensor(batch_img[:, np.newaxis, ...]), 'label': tensor.get('label'),
                  'duration': 1.0 * tensor.get('duration')}
        # print(result['video'].size())

        return result

    def __len__(self):
        return len(self.list)

    @staticmethod
    def load_duration(file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.find('Duration') != -1:
                    duration = float(line.split(' ')[1])
        
        tensor = torch.zeros(29)
        mid = 29 / 2
        start = int(mid - duration / 2 * 25)
        end = int(mid + duration / 2 * 25)
        tensor[start: end] = 1.0
        return tensor
