import os
import torch
import torch.nn as nn
import logging
import argparse
import time
import datetime
from collections import OrderedDict
from apex.parallel import DistributedDataParallel as DDP

from models import SPOS
from utils import AverageMeter, set_seeds, setup_logger, calc_params, accuracy, reduce_tensor, save_checkpoint, \
    create_optimizer, create_scheduler, create_criterion
from datasets import create_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--distributed', action='store_true', help='distributed mode')
parser.add_argument('--gpu_devices', type=str, default='4, 5', help='available gpu devices')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--num_block_type', type=int, default=4, help='num of block types in each layer')
parser.add_argument('--in_channel_list', type=list, default=[16, 64, 160, 320, 640], help='in channel list of choice blocks')
parser.add_argument('--num_layer_list', type=list, default=[4, 4, 8, 4], help='layer num list of choice blocks')
parser.add_argument('--best_arch', type=list, default=[1, 0, 3, 3, 2, 2, 1, 0, 1, 3, 2, 1, 3, 3, 1, 0, 3, 0, 0, 1],
                    help='best architecture')

parser.add_argument('--start_epoch', type=int, default=1, help='start epoch (default is 1)')
parser.add_argument('--total_epochs', type=int, default=240, help='total epochs')
parser.add_argument('--disp_freq', type=int, default=50, help='display frequency')
parser.add_argument('--val_freq', type=int, default=1, help='validate frequency')
parser.add_argument('--keep_num', type=int, default=1, help='max ckpt num to keep')

parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--lr', type=float, default=0.5, help='initial learning rate')
parser.add_argument('--optim_type', type=str, default='sgd', help='optimizer type')
parser.add_argument('--sched_type', type=str, default='step', help='lr scheduler type')
parser.add_argument('--warmup_proportion', type=int, default=0, help='proportion of warmup steps')
parser.add_argument('--num_classes', type=int, default=1000, help='number of training classes')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing epsilon')

parser.add_argument('--exp_dir', type=str, default='./train_exp', help='experiment directory')
parser.add_argument('--resume_path', type=str, default='', help='path to resume checkpoint')
parser.add_argument('--train_dir', type=str, default='/home/wangguangrun/ILSVRC2012/train', help='directory to train dataset')
parser.add_argument('--val_dir', type=str, default='/home/wangguangrun/ILSVRC2012/val', help='directory to val dataset')
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

    model = SPOS(args.in_channel_list, args.num_layer_list, args.best_arch, args.num_classes)
    if args.local_rank == 0:
        logging.info('Model size: {:.2f}M'.format(calc_params(model) / 1e6))

    if use_gpu:
        if args.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            args.world_size = torch.distributed.get_world_size()
            logging.info('Training in distributed mode (process {}/{})'.format(args.local_rank + 1, args.world_size))

            model = DDP(model.cuda(), delay_allreduce=True)
        else:
            model = nn.DataParallel(model).cuda()

    train_dataset, train_loader = create_dataset(args.train_dir, args.batch_size, use_gpu, args.distributed, is_training=True)
    val_dataset, val_loader = create_dataset(args.val_dir, args.batch_size * 4, use_gpu, args.distributed)

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
        train(model, optim_tools, train_loader, val_loader, use_gpu)
    except KeyboardInterrupt:
        print('Keyboard interrupt (process {}/{})'.format(args.local_rank + 1, args.world_size))


def train(model, optim_tools, train_loader, val_loader, use_gpu):
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
            if args.local_rank == 0:
                logging.info('-' * 40)
                logging.info('==> Start evaluation')

            prec1, prec5 = validate(model, val_loader, use_gpu)

            if args.local_rank == 0:
                logging.info('-' * 40)
            is_best = prec1 > best_prec1
            if is_best:
                best_prec1 = prec1
                best_prec5 = prec5
                best_epoch = epoch

            if args.local_rank == 0:
                state = {'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'prec1': prec1,
                         'prec5': prec5,
                         'epoch': epoch}
                ckpt_name = 'ckpt_ep' + str(epoch) + '.bin'
                save_checkpoint(state, is_best, args.exp_dir, ckpt_name, args.keep_num)

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

        output = model(img)
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


def validate(model, loader, use_gpu):
    model.eval()
    all_prec1, all_prec5, val_time = [AverageMeter() for _ in range(3)]

    st_time = time.time()
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(loader):
            if use_gpu:
                img, label = img.cuda(), label.cuda()

            output = model(img)
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
