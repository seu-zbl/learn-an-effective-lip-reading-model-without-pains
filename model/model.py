from .video_cnn import VideoCNN
import torch
import torch.nn as nn


class VideoModel(nn.Module):

    def __init__(self, args, dropout=0.5):
        super(VideoModel, self).__init__()
        
        self.args = args
        self.video_cnn = VideoCNN(se=self.args.se)
        if self.args.border:
            in_dim = 512 + 1
        else:
            in_dim = 512
        self.gru = nn.GRU(in_dim, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)

        self.dropout = nn.Dropout(p=dropout)

        # TODO: 因为就10个数字，类别很少，我觉得中间插入一个全连接层是否更好些，例如256隐层单元
        if self.finetune:
            self.fc1 = nn.Linear(1024*2, 256)
            self.v_cls = nn.Linear(256, self.args.n_class)
        else:
            self.v_cls = nn.Linear(1024*2, self.args.n_class)

    def forward(self, v, border=None):
        self.gru.flatten_parameters()

        f_v = self.video_cnn(v)
        f_v = self.dropout(f_v)

        if self.args.border:
            border = border[:, :, None]
            h, _ = self.gru(torch.cat([f_v, border], -1))
        else:
            h, _ = self.gru(f_v)

        if self.finetune:
            y1 = self.fc1(self.dropout(h))
            y_v = self.v_cls(self.dropout(y1)).mean(1)
        else:
            y_v = self.v_cls(self.dropout(h)).mean(1)
        return y_v
