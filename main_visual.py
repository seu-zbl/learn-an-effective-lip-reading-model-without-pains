import os
import time
import argparse
import numpy as np

from model import *
from LSR import LSR

import torch.optim as optim
from torch.utils.data import DataLoader


torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser.add_argument('--gpus', type=str, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--test_interval', type=float, required=True)
parser.add_argument('--n_class', type=int, required=True)
parser.add_argument('--num_workers', type=int, required=True)
parser.add_argument('--max_epoch', type=int, required=True)
parser.add_argument('--test', type=str2bool, required=True)
# load opts
parser.add_argument('--weights', type=str, required=False, default=None)
# save prefix
parser.add_argument('--save_prefix', type=str, required=True)
# dataset
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--border', type=str2bool, required=True)
parser.add_argument('--mixup', type=str2bool, required=True)
parser.add_argument('--label_smooth', type=str2bool, required=True)
parser.add_argument('--se', type=str2bool, required=True)
# finetune
parser.add_argument('--finetune', type=str2bool, required=True, default=False)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

if args.dataset == 'lrw':
    from utils import LRWDataset as Dataset
elif args.dataset == 'lrw1000':
    from utils import LRW1000_Dataset as Dataset
else:
    raise Exception('lrw or lrw1000')

video_model = VideoModel(args).cuda()


def parallel_model(model):
    """
    多GPU计算

    :param model:
    :return:
    """
    model = nn.DataParallel(model)
    return model


def load_missing(model, pretrained_dict):
    """
    利用预训练模型对当前模型中具有同样结构的层，进行参数初始化

    :param model: 当前模型
    :param pretrained_dict: 加载的预训练模型参数
    :return:
    """
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict.keys() and v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items() if k not in pretrained_dict.keys()]

    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict), len(model_dict)))
    print('miss matched params:', missed_params)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


# TODO: finetune 需要测试下代码是否有问题
def freezing(model):
    """
    冻结全连接层之前的所有层

    :param model:
    :return:
    """
    for name, param in model.named_parameters():
        if 'v_cls' in name or 'fc1' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


if args.finetue:
    freezing(video_model)
    optim_video = optim.Adam(filter(lambda p: p.requires_grad, video_model.parameters()),
                             lr=args.lr, weight_decay=1e-4)
else:
    optim_video = optim.Adam(video_model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_video, T_max=args.max_epoch, eta_min=1e-6)

if args.weights is not None:
    print('load weights')
    weight = torch.load(args.weights, map_location=torch.device('cpu'))
    load_missing(video_model, weight.get('video_model'))

video_model = parallel_model(video_model)


def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
    """
    数据集加载DataLoader

    :param dataset:
    :param batch_size:
    :param num_workers:
    :param shuffle:
    :return:
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=True
    )
    return loader


def add_msg(msg, k, v):
    """
    逗号分割信息

    :param msg:
    :param k:
    :param v:
    :return:
    """
    if msg != '':
        msg = msg + ','
    msg = msg + k.format(v)
    return msg


def test():
    with torch.no_grad():
        dataset = Dataset('test', args)
        print('Start Testing, Data Length:', len(dataset))
        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)

        print('start testing')
        v_acc = []
        total = 0

        for i_iter, input in enumerate(loader):
            video_model.eval()

            tic = time.time()
            video = input.get('video').cuda(non_blocking=True)
            label = input.get('label').cuda(non_blocking=True)
            total = total + video.size(0)
            # names = input.get('name')
            border = input.get('duration').cuda(non_blocking=True).float()

            if args.border:
                y_v = video_model(video, border)
            else:
                y_v = video_model(video)

            v_acc.extend((y_v.argmax(-1) == label).cpu().numpy().tolist())
            toc = time.time()
            if i_iter % 10 == 0:
                msg = ''
                msg = add_msg(msg, 'v_acc={:.5f}', np.array(v_acc).reshape(-1).mean())
                msg = add_msg(msg, 'eta={:.5f}', (toc - tic) * (len(loader) - i_iter) / 3600.0)
                print(msg)

        acc = float(np.array(v_acc).reshape(-1).mean())
        msg = 'v_acc_{:.5f}_'.format(acc)
        return acc, msg


def showLR(optimizer):
    """
    获取学习率

    :param optimizer:
    :return:
    """
    lr = []
    for param_group in optimizer.param_groups:
        lr += ['{:.6f}'.format(param_group['lr'])]
    return ','.join(lr)


def AdjustLR(optimizer):
    """
    减半学习率

    :param optimizer:
    :return:
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.5


def train():
    dataset = Dataset('train', args)
    print('Start Training, Data Length:', len(dataset))

    loader = dataset2dataloader(dataset, args.batch_size, args.num_workers)
    tot_iter = 0
    best_acc = 0.0
    alpha = 0.2
    max_epoch = args.max_epoch
    for epoch in range(max_epoch):
        lsr = LSR()

        for i_iter, input in enumerate(loader):
            tic = time.time()

            video_model.train()
            video = input.get('video').cuda(non_blocking=True)
            label = input.get('label').cuda(non_blocking=True).long()
            border = input.get('duration').cuda(non_blocking=True).float()

            loss = {}

            if args.label_smooth:
                loss_fn = lsr
            else:
                loss_fn = nn.CrossEntropyLoss()

            if args.mixup:
                lambda_ = np.random.beta(alpha, alpha)
                index = torch.randperm(video.size(0)).cuda(non_blocking=True)
                mix_video = lambda_ * video + (1 - lambda_) * video[index, :]
                mix_border = lambda_ * border + (1 - lambda_) * border[index, :]
                label_a, label_b = label, label[index]

                if args.border:
                    y_v = video_model(mix_video, mix_border)
                else:
                    y_v = video_model(mix_video)

                loss_bp = lambda_ * loss_fn(y_v, label_a) + (1 - lambda_) * loss_fn(y_v, label_b)

            else:
                if args.border:
                    y_v = video_model(video, border)
                else:
                    y_v = video_model(video)

                loss_bp = loss_fn(y_v, label)

            loss['CE V'] = loss_bp

            optim_video.zero_grad()
            loss_bp.backward()
            optim_video.step()

            toc = time.time()
            if tot_iter % 10 == 0:
                msg = 'epoch={},train_iter={},eta={:.5f}'\
                    .format(epoch, tot_iter, (toc - tic) * (len(loader) - i_iter) / 3600.0)
                for k, v in loss.items():
                    msg += ',{}={:.5f}'.format(k, v)
                msg = msg + str(',lr=' + str(showLR(optim_video)))
                msg = msg + str(',best_acc={:2f}'.format(best_acc))
                print(msg)

            test_interval = int((len(loader) - 1) * args.test_interval)
            if tot_iter % test_interval == 0:
                acc, msg = test()
                if acc > best_acc:
                    savename = '{}_iter_{}_epoch_{}_{}.pt'.format(args.save_prefix, tot_iter, epoch, msg)
                    temp = os.path.split(savename)[0]
                    if not os.path.exists(temp):
                        os.makedirs(temp)
                    torch.save({
                        'video_model': video_model.module.state_dict(),
                    }, savename)

                if tot_iter != 0:
                    best_acc = max(acc, best_acc)

            tot_iter += 1

        scheduler.step()


if __name__ == '__main__':
    if args.test:
        acc, msg = test()
        print(f'acc={acc}')
        exit()

    train()
