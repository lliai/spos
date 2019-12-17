import os
import torch
import torch.nn as nn
import numpy as np
import time
import datetime
import argparse
import random
import logging
import copy
from model import SPOS
from utils import AverageMeter, accuracy, setup_logger, set_seeds, bn_recalc, calc_params
from dataset import create_loader, create_bn_loader
from apex.parallel import DistributedDataParallel as DDP


parser = argparse.ArgumentParser('arch-search')
parser.add_argument('--distributed', type=bool, default=False, help='distributed mode or not')
parser.add_argument('--gpu-devices', type=str, default='4, 5, 6, 7', help='chosen gpu devices')
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--num-block-type', type=int, default=4, help='num of block types in each layer')
parser.add_argument('--num-layer-list', type=list, default=[4, 4, 8, 4], help='layer num list of choice blocks')
parser.add_argument('--in-channel-list', type=list, default=[16, 64, 160, 320, 640], help='in channel list of choice blocks')

parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--num-classes', type=int, default=1000, help='number of training classes')
parser.add_argument('--bn-recalc-imgs', type=int, default=20000, help='num of train images to recalc bn statistics')
parser.add_argument('--pop-size', type=int, default=50, help='population size in evolution search')
parser.add_argument('--start-iter', type=int, default=0, help='start iteration')
parser.add_argument('--max-iter', type=int, default=20, help='max iteration in evolution search')
parser.add_argument('--topk', type=int, default=10, help='topk models to be selected in each evolution iteration')
parser.add_argument('--mut-prob', type=float, default=0.1, help='mutation probability in evolution search')

parser.add_argument('--ckpt-path', type=str, default='', help='path to checkpoint')
parser.add_argument('--history-path', type=str, default='', help='path to search history file')
parser.add_argument('--exp-dir', type=str, default='search_exp', help='experiment directory')
parser.add_argument('--train-path', type=str, default='./supernet_train_data.csv', help='path to train dataset')
parser.add_argument('--val-path', type=str, default='./supernet_val_data.csv', help='path to val dataset')
parser.add_argument('--local_rank', type=int, default=0, help='DDP local rank')
parser.add_argument('--world_size', type=int, default=1, help='DDP world size')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices


def main():
    args.exp_dir = setup_logger(args.exp_dir)
    if args.local_rank == 0:
        logging.info(args)

    use_gpu = False
    if args.gpu_devices is not None and torch.cuda.is_available():
        use_gpu = True

    if use_gpu and args.local_rank == 0:
        logging.info('Currently using GPU: {}'.format(args.gpu_devices))
    elif not use_gpu and args.local_rank == 0:
        logging.info('Currently using CPU')

    set_seeds(args.seed, use_gpu)

    model = SPOS(args.in_channel_list, args.num_layer_list, args.num_classes, args.num_block_type)
    if args.local_rank == 0:
        logging.info('Model size: {:.3f}M'.format(calc_params(model) / 1e6))

    if use_gpu:
        if args.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            args.world_size = torch.distributed.get_world_size()
            logging.info('Training in distributed mode (process {}/{})'.format(args.local_rank, args.world_size - 1))

            model = DDP(model.cuda(), delay_allreduce=True)
        else:
            model = nn.DataParallel(model).cuda()

    val_loader = create_loader(args.val_path, args.batch_size, use_gpu, args.distributed)
    bn_loader = create_bn_loader(args.train_path, args.batch_size, use_gpu, args.distributed)

    if args.ckpt_path:
        if os.path.exists(args.ckpt_path):
            checkpoint = torch.load(args.ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])

            if args.local_rank == 0:
                logging.info('Loaded checkpoint from \'{}\''.format(args.ckpt_path))
                logging.info('Epoch: {}\tPrec@1: {:.2f}%\tPrec@5: {:.2f}%'
                             .format(checkpoint['epoch'] - 1, checkpoint['prec1'], checkpoint['prec5']))
        else:
            if args.local_rank == 0:
                logging.info('No checkpoint found in \'{}\''.format(args.ckpt_path))

    try:
        evolution_search(model, val_loader, bn_loader, use_gpu)
    except KeyboardInterrupt:
        print('Keyboard interrupt (process {}/{})'.format(args.local_rank, args.world_size - 1))


def evolution_search(model, val_loader, bn_loader, use_gpu):
    inf_time = 0
    total_topk_arch = []
    total_topk_prec1 = []
    st_time = time.time()

    num_crossover = args.pop_size // 2
    num_mutation = args.pop_size // 2
    pop = []
    while len(pop) < args.pop_size:
        pop_gen = [random.randint(0, args.num_block_type - 1) for _ in range(sum(args.num_layer_list))]
        if pop_gen not in pop:
            pop.append(pop_gen)

    if args.history_path:
        if os.path.isfile(args.history_path):
            history = torch.load(args.history_path)
            args.start_iter = history['iter']
            pop = history['pop']
            total_topk_arch = history['topk_arch']
            total_topk_prec1 = history['topk_prec1']
            if args.local_rank == 0:
                logging.info('Loaded evolved population from \'{}\''.format(args.history_path))
                logging.info('Start iter: {}'.format(args.start_iter))
        else:
            if args.local_rank == 0:
                logging.info('No history file found in \'{}\''.format(args.history_path))

    if args.local_rank == 0:
        logging.info('==> Start evolution searching')
    raw_params = copy.deepcopy(model.state_dict())

    for itr in range(args.start_iter, args.max_iter):
        st_inf_time = time.time()

        all_prec1 = []
        for idx, arch in enumerate(pop):
            bn_recalc(model, args, arch, bn_loader, use_gpu)
            prec1 = inference(model, itr, idx, arch, val_loader, use_gpu)
            model.load_state_dict(raw_params)
            all_prec1.append(prec1)

        torch.cuda.synchronize()
        inf_time += round(time.time() - st_inf_time)

        topk_idx = np.argsort(all_prec1)[::-1][:args.topk]
        topk_pop = np.array(pop)[topk_idx].tolist()
        topk_prec1 = np.array(all_prec1)[topk_idx].tolist()

        new_pop1 = crossover(topk_pop, num_crossover)
        new_pop2 = mutation(topk_pop, num_mutation, new_pop1)
        pop = []
        pop.extend(new_pop1)
        pop.extend(new_pop2)

        total_topk_arch.extend(topk_pop)
        total_topk_prec1.extend(topk_prec1)
        total_topk_idx = np.argsort(total_topk_prec1)[::-1]
        temp_arch, temp_prec1 = [], []

        for idx in total_topk_idx:
            if len(temp_arch) >= args.topk:
                break
            if total_topk_arch[idx] not in temp_arch:
                temp_arch.append(total_topk_arch[idx])
                temp_prec1.append(total_topk_prec1[idx])
        if len(temp_arch) < args.topk:
            top_idx = total_topk_idx[0]
            for _ in range(args.topk - len(temp_arch)):
                temp_arch.append(total_topk_arch[top_idx])
                temp_prec1.append(total_topk_prec1[top_idx])

        total_topk_arch, total_topk_prec1 = temp_arch, temp_prec1

        if args.local_rank == 0:
            logging.info('-' * 40)
            logging.info('Results')
            for arch, prec1 in zip(total_topk_arch, total_topk_prec1):
                logging.info('Arch: {}\tPrec@1: {:.2f}%'.format(arch, prec1))
            logging.info('-' * 40)

            state = {'iter': itr + 1, 'pop': pop, 'topk_arch': total_topk_arch, 'topk_prec1': total_topk_prec1}
            save_name = 'history_iter' + str(itr) + '.pth.tar'
            save_path = os.path.join(args.exp_dir, save_name)
            if not os.path.exists(args.exp_dir):
                os.mkdir(args.exp_dir)
            torch.save(state, save_path)

    elapsed = round(time.time() - st_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    inf_time = str(datetime.timedelta(seconds=inf_time))
    if args.local_rank == 0:
        logging.info('Finished, total elapsed time (h:m:s): {}, inference time (h:m:s): {}'.format(elapsed, inf_time))


def inference(model, itr, pop_idx, arch, val_loader, use_gpu):
    model.eval()
    all_prec1 = AverageMeter()

    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(val_loader):
            if use_gpu:
                img, label = img.cuda(), label.cuda()

            output = model(img, arch)
            prec1 = accuracy(output, label, topk=(1, ))
            all_prec1.update(prec1.item(), img.size(0))

    if args.local_rank == 0:
        logging.info('Iter: [{}][{}/{}]\tArch: {}\tPrec@1: {:.2f}%'
                     .format(itr, pop_idx, args.pop_size - 1, arch, all_prec1.avg))

    return all_prec1.avg


def crossover(raw_pop, num_pop):
    new_pop = []
    while len(new_pop) < num_pop:
        pop = copy.deepcopy(raw_pop)
        id1, id2 = random.sample(range(len(pop)), 2)
        st_pos, ed_pos = random.sample(range(len(pop[0])), 2)
        if st_pos > ed_pos:
            st_pos, ed_pos = ed_pos, st_pos
        temp = pop[id1].copy()
        pop[id1][st_pos:ed_pos + 1] = pop[id2][st_pos:ed_pos + 1]
        pop[id2][st_pos:ed_pos + 1] = temp[st_pos:ed_pos + 1]

        if pop[id1] not in new_pop:
            new_pop.append(pop[id1])
        if pop[id2] not in new_pop and len(new_pop) < num_pop:
            new_pop.append(pop[id2])

    return new_pop


def mutation(raw_pop, num_pop, pop_cross):
    new_pop = []
    while len(new_pop) < num_pop:
        pop = copy.deepcopy(raw_pop)
        id = random.sample(range(len(pop)), 1)[0]
        for l in range(len(pop[0])):
            if random.random() < args.mut_prob:
                choices = [i for i in range(args.num_block_type) if i != pop[id][l]]
                pop[id][l] = random.sample(choices, 1)[0]

        if pop[id] not in new_pop and pop[id] not in pop_cross:
            new_pop.append(pop[id])

    return new_pop


if __name__ == '__main__':
    main()
