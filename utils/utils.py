import sys
import logging
import os
import re
import torch
import random
import shutil
import math
import numpy as np
from .flop_utils import get_flops


def setup_logger(log_dir):
    try:
        os.makedirs(log_dir)
    except FileExistsError:
        pass
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%d %I:%M:%S')
    fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


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


def recalc_bn(model, arch, loader, use_gpu, bn_recalc_imgs, world_size):
    model.train()
    img_cnt = 0

    for batch_idx, (img, _) in enumerate(loader):
        if use_gpu:
            img = img.cuda()
        model(img, arch)

        img_cnt += img.size(0) * world_size
        if img_cnt > bn_recalc_imgs:
            break


def uniform_constraint_sampling(num_layers, num_candidates, flop_table, local_rank=0):
    flop_scope = [290, 360]
    flop_step = 10
    sampled_scope_idx = random.randint(0, math.ceil((flop_scope[1] - flop_scope[0]) / flop_step) - 1)
    sampled_scope = [flop_scope[0] + sampled_scope_idx * flop_step, flop_scope[0] + (sampled_scope_idx + 1) * flop_step]

    arch = [random.randint(0, num_candidates - 1) for _ in range(num_layers)]
    flops = get_flops(arch, flop_table) / 1e6
    if not sampled_scope[0] <= flops <= sampled_scope[1]:
        cnt = 0
        timeout = 1
        while cnt < timeout:
            arch = [random.randint(0, num_candidates - 1) for _ in range(num_layers)]
            flops = get_flops(arch, flop_table) / 1e6
            cnt += 1
            if sampled_scope[0] <= flops <= sampled_scope[1]:
                return arch, flops

        if local_rank == 0:
            logging.info('Sample FLOPs timeout, expected FLOPs scope [{}, {}]M, sampled FLOPs {:.2f}M'.format(
                sampled_scope[0], sampled_scope[1], flops))

    return arch, flops


def save_checkpoint(state, is_best, save_dir, ckpt_name, keep_num):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, ckpt_name)
    torch.save(state, save_path)
    if is_best:
        shutil.copy(save_path, os.path.join(save_dir, 'best_model.bin'))

    ckpt_head = re.split(r'\d+', ckpt_name)[0]
    all_ckpt = np.array([file for file in os.listdir(save_dir) if re.match(ckpt_head, file) is not None])
    all_ep = np.int32([re.findall(r'\d+', ckpt)[0] for ckpt in all_ckpt])
    sorted_ckpt = all_ckpt[np.argsort(all_ep)[::-1]]
    remove_path = [os.path.join(save_dir, name) for name in sorted_ckpt[keep_num:]]
    for path in remove_path:
        os.remove(path)


def save_search_history(state, save_dir, history_name, keep_num):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, history_name)
    torch.save(state, save_path)

    history_head = re.split(r'\d', history_name)[0]
    all_history = np.array([file for file in os.listdir(save_dir) if re.match(history_head, file) is not None])
    all_iter = np.int32([re.findall(r'\d+', history)[0] for history in all_history])
    sorted_history = all_history[np.argsort(all_iter)[::-1]]
    remove_path = [os.path.join(save_dir, name) for name in sorted_history[keep_num:]]
    for path in remove_path:
        os.remove(path)
