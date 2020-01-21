import os
import torch
import torch.nn as nn
import logging
import argparse
import random
import time
import datetime
from copy import deepcopy
from collections import OrderedDict
from apex.parallel import DistributedDataParallel as DDP

from models import SPOS_Supernet
from utils import AverageMeter, set_seeds, setup_logger, calc_params, accuracy, reduce_tensor, write_flops, recalc_bn, \
    uniform_constraint_sampling, save_checkpoint, create_optimizer, create_scheduler, create_criterion
from datasets import save_split_data, create_supernet_dataset, create_bn_dataset
from evolution_search import evolution_search


parser = argparse.ArgumentParser()
parser.add_argument('--distributed', action='store_true', help='distributed mode')
parser.add_argument('--gpu_devices', type=str, default='4, 5', help='available gpu devices')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--num_block_type', type=int, default=4, help='num of block types in each layer')
parser.add_argument('--num_layer_list', type=list, default=[4, 4, 8, 4], help='layer num list of choice blocks')
parser.add_argument('--in_channel_list', type=list, default=[16, 64, 160, 320, 640], help='in channel list of choice blocks')

# Arch train settings
parser.add_argument('--start_epoch', type=int, default=1, help='start epoch (default is 1)')
parser.add_argument('--total_epochs', type=int, default=120, help='total epochs')
parser.add_argument('--disp_freq', type=int, default=50, help='display frequency')
parser.add_argument('--val_freq', type=int, default=1, help='validate frequency')
parser.add_argument('--ckpt_keep_num', type=int, default=5, help='max ckpt num to keep')

parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--lr', type=float, default=0.5, help='initial learning rate')
parser.add_argument('--optim_type', type=str, default='sgd', help='optimizer type')
parser.add_argument('--sched_type', type=str, default='step', help='lr scheduler type')
parser.add_argument('--warmup_proportion', type=int, default=0, help='proportion of warmup steps')
parser.add_argument('--num_classes', type=int, default=1000, help='number of training classes')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing epsilon')
parser.add_argument('--bn_recalc_imgs', type=int, default=20000, help='num of train images to recalc bn statistics')

# Arch search settings
parser.add_argument('--flop_table_path', type=str, default='./flop_table.bin', help='path to save and load flop table')
parser.add_argument('--max_flops', type=int, default=330, help='max evaluation FLOPs(M)')
parser.add_argument('--pop_size', type=int, default=50, help='population size')
parser.add_argument('--start_search_iter', type=int, default=1, help='start iteration (default is 1)')
parser.add_argument('--total_search_iters', type=int, default=20, help='max iteration in evolution search')
parser.add_argument('--topk', type=int, default=10, help='topk models to be selected at the end of each evolution iteration')
parser.add_argument('--mut_prob', type=float, default=0.1, help='mutation probability')
parser.add_argument('--history_keep_num', type=int, default=1, help='max search history num to keep')

parser.add_argument('--exp_dir', type=str, default='./search_exp', help='experiment directory')
parser.add_argument('--resume_path', type=str, default='./search_exp/20191230-222527/ckpt_ep120.bin', help='path to resume checkpoint')
parser.add_argument('--history_path', type=str, default='', help='path to search history file')
parser.add_argument('--raw_train_dir', type=str, default='/home/wangguangrun/ILSVRC2012/train', help='directory to raw train dataset')
parser.add_argument('--train_path', type=str, default='./supernet_train_data.csv', help='path to save and load split train data')
parser.add_argument('--val_path', type=str, default='./supernet_val_data.csv', help='path to save and load split val data')
parser.add_argument('--local_rank', type=int, default=0, help='DDP local rank')
parser.add_argument('--world_size', type=int, default=1, help='DDP world size')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices


def main():
    args.exp_dir = os.path.join(args.exp_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    setup_logger(args.exp_dir)

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

    model = SPOS_Supernet(args.in_channel_list, args.num_layer_list, args.num_classes, args.num_block_type)
    if args.local_rank == 0:
        logging.info('Model size: {:.2f}M'.format(calc_params(model) / 1e6))

    if not os.path.exists(args.flop_table_path):
        write_flops(args.flop_table_path)
        if args.local_rank == 0:
            logging.info('Flop table has been saved to \'{}\''.format(args.flop_table_path))
    args.flop_table = torch.load(args.flop_table_path)

    if use_gpu:
        if args.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            args.world_size = torch.distributed.get_world_size()
            logging.info('Training in distributed mode (process {}/{})'.format(args.local_rank + 1, args.world_size))

            model = DDP(model.cuda(), delay_allreduce=True)
        else:
            model = nn.DataParallel(model).cuda()

    if not os.path.exists(args.train_path) or not os.path.exists(args.val_path):
        save_split_data(args.raw_train_dir, args.train_path, args.val_path)
        if args.local_rank == 0:
            logging.info('Supernet train and val data have been split and saved to \'{}\' and \'{}\''
                         .format(args.train_path, args.val_path))

    train_dataset, train_loader = create_supernet_dataset(
        args.train_path, args.batch_size, use_gpu, args.distributed, is_training=True)
    val_dataset, val_loader = create_supernet_dataset(args.val_path, args.batch_size * 4, use_gpu, args.distributed)
    bn_dataset, bn_loader = create_bn_dataset(args.train_path, args.batch_size, use_gpu, args.distributed)

    args.num_sched_steps = len(train_loader) * args.total_epochs
    args.num_warmup_steps = int(args.num_sched_steps * args.warmup_proportion)
    optimizer = create_optimizer(model, args.optim_type, args.lr, args.weight_decay, args.momentum)
    criterion = create_criterion(args.num_classes, args.label_smooth)
    scheduler = create_scheduler(optimizer, args.sched_type, args.num_sched_steps, args.num_warmup_steps)
    criterion = criterion.cuda() if use_gpu else criterion
    optim_tools = [optimizer, criterion, scheduler]

    if args.resume_path:
        if os.path.exists(args.resume_path):
            checkpoint = torch.load(args.resume_path, map_location='cpu')
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except RuntimeError:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)

            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            args.start_epoch = checkpoint['epoch'] + 1
            if args.local_rank == 0:
                logging.info('Loaded checkpoint from \'{}\''.format(args.resume_path))
                logging.info('Start epoch: {}\tPrec@1: {:.2f}%\tPrec@5: {:.2f}%'
                             .format(args.start_epoch, checkpoint['prec1'], checkpoint['prec5']))
        else:
            if args.local_rank == 0:
                logging.info('No checkpoint found in \'{}\''.format(args.resume_path))

    try:
        train(model, optim_tools, train_loader, val_loader, bn_loader, use_gpu)
        evolution_search(model, val_loader, bn_loader, use_gpu, args)
    except KeyboardInterrupt:
        print('Keyboard interrupt (process {}/{})'.format(args.local_rank + 1, args.world_size))


def train(model, optim_tools, train_loader, val_loader, bn_loader, use_gpu):
    best_prec1, best_prec5 = 0, 0
    best_epoch = args.start_epoch - 1
    optimizer, criterion, scheduler = optim_tools
    st_time = time.time()

    if args.local_rank == 0:
        logging.info('==> Start training')

    for epoch in range(args.start_epoch, args.total_epochs + 1):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_epoch(model, epoch, optim_tools, train_loader, use_gpu)

        if epoch % args.val_freq == 0 or epoch == args.total_epochs:
            arch = [random.randint(0, args.num_block_type - 1) for _ in range(sum(args.num_layer_list))]

            if args.local_rank == 0:
                logging.info('-' * 40)
                logging.info('==> Start evaluation')
                logging.info('Eval arch: {}'.format(arch))
                logging.info('Recalculating bn statistics...')

            raw_params = deepcopy(model.state_dict())
            recalc_bn(model, arch, bn_loader, use_gpu, args.bn_recalc_imgs, args.world_size)
            prec1, prec5 = validate(model, arch, val_loader, use_gpu)
            model.load_state_dict(raw_params)

            is_best = prec1 > best_prec1
            if is_best:
                best_prec1 = prec1
                best_prec5 = prec5
                best_epoch = epoch

            if args.local_rank == 0:
                logging.info('-' * 40)

                state = {'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'prec1': prec1,
                         'prec5': prec5,
                         'epoch': epoch}
                ckpt_name = 'ckpt_ep' + str(epoch) + '.bin'
                save_checkpoint(state, is_best, args.exp_dir, ckpt_name, args.ckpt_keep_num)

    elapsed = round(time.time() - st_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    if args.local_rank == 0:
        logging.info("==> Best prec@1 {:.2f}%, prec@5 {:.2f}%, achieved at epoch {}".format(best_prec1, best_prec5, best_epoch))
        logging.info("Finished, total training time (h:m:s): {}".format(elapsed))


def train_epoch(model, epoch, optim_tools, train_loader, use_gpu):
    model.train()
    optimizer, criterion, scheduler = optim_tools

    losses, train_time, data_time = [AverageMeter() for _ in range(3)]
    st_time = time.time()
    for batch_idx, (img, label) in enumerate(train_loader):
        data_time.update(time.time() - st_time)
        if use_gpu:
            img, label = img.cuda(), label.cuda()

        arch, flops = uniform_constraint_sampling(sum(args.num_layer_list), args.num_block_type, args.flop_table, args.local_rank)

        output = model(img, arch)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if not args.distributed:
            losses.update(loss.item(), img.size(0))

        if use_gpu:
            torch.cuda.synchronize()
        train_time.update(time.time() - st_time)

        if batch_idx == 0 or (batch_idx + 1) % args.disp_freq == 0 or batch_idx + 1 == len(train_loader):
            if args.distributed:
                reduced_loss = reduce_tensor(loss.detach(), args.world_size)
                losses.update(reduced_loss.item(), img.size(0))

            if args.local_rank == 0:
                lr = scheduler.get_lr()[0]
                logging.info('Epoch: [{}/{}][{}/{}]\t'
                             'LR: {:.2e}\t'
                             'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Train time: {train_time.val:.4f}s ({train_time.avg:.4f}s)\t'
                             'Load data time: {data_time.val:.4f}s ({data_time.avg:.4f}s)'
                             .format(epoch, args.total_epochs, batch_idx + 1, len(train_loader), lr,
                                     loss=losses, train_time=train_time, data_time=data_time))
        st_time = time.time()


def validate(model, arch, loader, use_gpu):
    model.eval()
    all_prec1, all_prec5, val_time = [AverageMeter() for _ in range(3)]

    st_time = time.time()
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(loader):
            if use_gpu:
                img, label = img.cuda(), label.cuda()

            output = model(img, arch)
            prec1, prec5 = accuracy(output, label, topk=(1, 5))

            if args.distributed:
                prec1 = reduce_tensor(prec1, args.world_size)
                prec5 = reduce_tensor(prec5, args.world_size)

            all_prec1.update(prec1.item(), img.size(0))
            all_prec5.update(prec5.item(), img.size(0))

            if use_gpu:
                torch.cuda.synchronize()
            val_time.update(time.time() - st_time)

            if args.local_rank == 0 and \
                    (batch_idx == 0 or (batch_idx + 1) % args.disp_freq == 0 or batch_idx + 1 == len(loader)):
                logging.info('Iter: [{}/{}]\t'
                             'Val time: {:.4f}s\t'
                             'Prec@1: {:.2f}%\t'
                             'Prec@5: {:.2f}%'
                             .format(batch_idx + 1, len(loader), val_time.avg, all_prec1.avg, all_prec5.avg))
            st_time = time.time()

    return all_prec1.avg, all_prec5.avg


if __name__ == '__main__':
    main()
