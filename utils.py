import sys
import logging
import os
import torch
import random
import shutil
import datetime
import numpy as np


def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = os.path.join(log_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%d %I:%M:%S')

    fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    return log_dir


def set_seeds(seed, use_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def calc_params(model):
    return sum(p.numel() for p in model.parameters())


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= n
    return rt


def bn_recalc(model, args, arch, loader, use_gpu):
    model.train()
    img_cnt = 0

    for batch_idx, (img, _) in enumerate(loader):
        if use_gpu:
            img = img.cuda()
        _ = model(img, arch)

        img_cnt += img.size(0) * args.world_size
        if img_cnt > args.bn_recalc_imgs:
            break


def save_checkpoint(state, is_best, save_dir, ckpt_name, keep_num):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, ckpt_name)
    torch.save(state, save_path)
    if is_best:
        shutil.copy(save_path, os.path.join(save_dir, 'best_model.pth.tar'))

    all_ckpt = np.array([i for i in os.listdir(save_dir) if i not in ['log.txt', 'best_model.pth.tar']])
    all_ep = np.int32([i.split('ckpt_ep')[1].split('.')[0] for i in all_ckpt])
    sorted_ckpt = all_ckpt[np.argsort(all_ep)[::-1]]
    remove_path = [os.path.join(save_dir, name) for name in sorted_ckpt[keep_num:]]
    for path in remove_path:
        os.remove(path)
