import os
import random
import time
from collections import defaultdict
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader


class LRW1000_Dataset(Dataset):

    def __init__(self, index_file, target_dir):

        self.data = []
        self.index_file = index_file
        self.target_dir = target_dir
        lines = []

        with open(index_file, 'r') as f:
            lines.extend([line.strip().split(',') for line in f.readlines()])

        self.data_root = 'LRW1000_Public/images'
        self.padding = 40

        pinyins = sorted(np.unique([line[2] for line in lines]))
        self.data = [(line[0], int(float(line[3]) * 25) + 1, int(float(line[4]) * 25) + 1, pinyins.index(line[2]))
                     for line in lines]
        max_len = max([data[2] - data[1] for data in self.data])
        data = list(filter(lambda data: data[2] - data[1] <= self.padding, self.data))
        self.lengths = [data[2] - data[1] for data in self.data]
        self.pinyins = pinyins

        self.va_dict = self.get_video_audio_map()
        self.class_dict = defaultdict(list)

        for item in data:
            audio_file = self.va_dict.get(item)
            assert (audio_file is not None)
            audio_file = 'LRW1000_Public/audio/' + audio_file + '.npy'
            if os.path.exists(audio_file):
                item = (item[0], audio_file, item[1], item[2], item[3])
                self.class_dict[item[-1]].append(item)

        self.data = []
        self.unlabel_data = []
        for k, v in self.class_dict.items():
            n = len(v)

            self.data.extend(v[:n])

    def get_video_audio_map(self):

        self.anno = 'LRW1000_Public/info/all_audio_video.txt'
        with open(self.anno, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            lines = [line.split(',') for line in lines]
            va_dict = {}
            for (v, a, _, pinyin, op, ed) in lines:
                op = float(op) * 25 + 1
                ed = float(ed) * 25 + 1
                pinyin = self.pinyins.index(pinyin)
                op, ed = int(op), int(ed)
                va_dict[(v, op, ed, pinyin)] = a

        return va_dict

    def __len__(self):
        return len(self.data)

    def load_video(self, item):
        # load video into a tensor
        (path, mfcc, op, ed, label) = item
        inputs, border = self.load_images(os.path.join(self.data_root, path), op, ed)

        result = {
            'video': inputs.astype(np.uint8),
            'label': int(label),
            'duration': border.astype(np.bool)
        }

        savename = os.path.join(target_dir, f'{path}_{op}_{ed}.pkl')
        torch.save(result, savename)

        return result

    def __getitem__(self, idx):

        r1 = self.load_video(self.data[idx])

        if len(self.unlabel_data) > 0 and self.phase == 'train':
            r2 = self.load_video(random.choice(self.unlabel_data))
            r2 = {'unlabel_' + k: v for k, v in r2.items()}
            r1.update(r2)

        return r1

    def load_images(self, path, op, ed):
        center = (op + ed) / 2
        length = (ed - op + 1)

        op = int(center - self.padding // 2)
        ed = int(op + self.padding)
        left_border = max(int(center - length / 2 - op), 0)
        right_border = min(int(center + length / 2 - op), self.padding)
        # print(length, center, op, ed, left_border, right_border)

        files = [os.path.join(path, '{}.jpg'.format(i)) for i in range(op, ed)]
        files = filter(lambda path: os.path.exists(path), files)
        files = [cv2.imread(file) for file in files]
        files = [cv2.cvtColor(file, cv2.COLOR_RGB2GRAY) for file in files]
        files = [cv2.resize(file, (96, 96)) for file in files]
        files = np.stack(files, 0)
        t = files.shape[0]
        files = files.reshape(t, 96, 96)

        tensor = np.zeros((40, 96, 96))
        border = np.zeros(40)
        tensor[:t, ...] = files
        border[left_border: right_border] = 1.0

        return tensor, border


if __name__ == '__main__':

    for subset in ['trn', 'val', 'tst']:
        target_dir = f'LRW1000_Public_pkl_test/{subset}'
        index_file = f'LRW1000_Public/info/{subset}_1000.txt'

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        loader = DataLoader(
            LRW1000_Dataset(index_file, target_dir),
            batch_size=96,
            num_workers=8,
            shuffle=False,
            drop_last=False)

        tic = time.time()
        for i, batch in enumerate(loader):
            toc = time.time()
            eta = ((toc - tic) / (i + 1) * (len(loader) - i)) / 3600.0
            print(f'eta:{eta:.5f}')
