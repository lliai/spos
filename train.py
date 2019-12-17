import os
import torch
import torch.nn as nn
import logging
import argparse
import time
import datetime
from model import SPOS
from utils import AverageMeter, set_seeds, setup_logger, calc_params, accuracy, reduce_tensor, save_checkpoint
from optim_utils import create_optimizer, create_scheduler, create_criterion
from dataset import create_loader
from apex.parallel import DistributedDataParallel as DDP


parser = argparse.ArgumentParser('best-subnet-training')
parser.add_argument('--eval-only', type=bool, default=False, help='evaluate only or not')
parser.add_argument('--distributed', type=bool, default=False, help='distributed mode or not')
parser.add_argument('--gpu-devices', type=str, default='4, 5, 6, 7', help='chosen gpu devices')
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--num-block-type', type=int, default=4, help='num of block types in each layer')
parser.add_argument('--in-channel-list', type=list, default=[16, 64, 160, 320, 640], help='in channel list of choice blocks')
parser.add_argument('--num-layer-list', type=list, default=[4, 4, 8, 4], help='layer num list of choice blocks')
parser.add_argument('--best-arch', type=list, default=[2, 2, 3, 0, 2, 0, 2, 1, 2, 2, 2, 1, 3, 1, 1, 3, 2, 0, 1, 1], help='best architecture')

parser.add_argument('--start-epoch', type=int, default=0, help='start epoch')
parser.add_argument('--total-epochs', type=int, default=360, help='total epochs')
parser.add_argument('--disp-freq', type=int, default=50, help='display frequency')
parser.add_argument('--val-freq', type=int, default=1, help='validate frequency')
parser.add_argument('--keep-num', type=int, default=10, help='max ckpt num to keep')

parser.add_argument('--crop-size', type=int, default=224, help='crop size of input images')
parser.add_argument('--batch-size', type=int, default=200, help='batch size')
parser.add_argument('--lr', type=float, default=0.5, help='initial learning rate')
parser.add_argument('--min-lr', type=float, default=0, help='minimal learning rate')
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
parser.add_argument('--scheduler', type=str, default='step', help='scheduler')
parser.add_argument('--num-classes', type=int, default=1000, help='number of training classes')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
parser.add_argument('--label-smooth', type=float, default=0.1, help='label smoothing epsilon')

parser.add_argument('--exp-dir', type=str, default='train_exp', help='experiment directory')
parser.add_argument('--resume-path', type=str, default='', help='path to resume checkpoint')
parser.add_argument('--save-dir', type=str, default='checkpoints', help='directory to save checkpoints')
parser.add_argument('--train-path', type=str, default='ILSVRC2012/train', help='path to train dataset')
parser.add_argument('--val-path', type=str, default='ILSVRC2012/val', help='path to val dataset')
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

    train_loader = create_loader(args.train_path, args.batch_size, use_gpu, args.distributed, is_training=True, train_best_subnet=True)
    val_loader = create_loader(args.val_path, args.batch_size, use_gpu, args.distributed, train_best_subnet=True)

    args.num_sched_iters = len(train_loader) * args.total_epochs
    optimizer = create_optimizer(model, args)
    criterion = create_criterion(args)
    scheduler = create_scheduler(optimizer, args)
    if use_gpu:
        criterion = criterion.cuda()
    optim_tools = [optimizer, criterion, scheduler]

    if args.resume_path:
        if os.path.exists(args.resume_path):
            checkpoint = torch.load(args.resume_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            args.start_epoch = checkpoint['epoch']
            if args.local_rank == 0:
                logging.info('Loaded checkpoint from \'{}\''.format(args.resume_path))
                logging.info('Start epoch: {}\tPrec@1: {:.2f}%\tPrec@5: {:.2f}%'
                             .format(args.start_epoch, checkpoint['prec1'], checkpoint['prec5']))
        else:
            if args.local_rank == 0:
                logging.info('No checkpoint found in \'{}\''.format(args.resume_path))

    if args.eval_only:
        try:
            arch = [0] * sum(args.num_layer_list)

            if args.local_rank == 0:
                logging.info('-' * 40)
                logging.info('==> Start evaluation')
                logging.info('Eval arch: {}'.format(arch))

            validate(model, arch, val_loader, use_gpu)

            if args.local_rank == 0:
                logging.info('-' * 40)
        except KeyboardInterrupt:
            print('Keyboard interrupt (process {}/{})'.format(args.local_rank, args.world_size - 1))

        return

    try:
        train(model, args.best_arch, optim_tools, train_loader, val_loader, use_gpu)
    except KeyboardInterrupt:
        print('Keyboard interrupt (process {}/{})'.format(args.local_rank, args.world_size - 1))


def train(model, arch, optim_tools, train_loader, val_loader, use_gpu):
    best_prec1 = 0
    best_prec5 = 0
    best_epoch = 0
    train_time = 0

    optimizer, criterion, scheduler = optim_tools
    st_time = time.time()
    if args.local_rank == 0:
        logging.info('==> Start training')

    for epoch in range(args.start_epoch, args.total_epochs):
        st_train_time = time.time()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_epoch(model, arch, epoch, optim_tools, train_loader, use_gpu)
        train_time += round(time.time() - st_train_time)

        if epoch % args.val_freq == 0:
            if args.local_rank == 0:
                logging.info('-' * 40)
                logging.info('==> Start evaluation')

            prec1, prec5 = validate(model, arch, val_loader, use_gpu)

            if args.local_rank == 0:
                logging.info('-' * 40)
            is_best = prec1 > best_prec1
            if is_best:
                best_prec1 = prec1
                best_prec5 = prec5
                best_epoch = epoch

            if args.local_rank == 0:
                state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                         'prec1': prec1, 'prec5': prec5, 'epoch': epoch + 1}
                ckpt_name = 'ckpt_ep' + str(epoch) + '.pth.tar'
                save_checkpoint(state, is_best, args.exp_dir, ckpt_name, args.keep_num)

    elapsed = round(time.time() - st_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    if args.local_rank == 0:
        logging.info("==> Best prec@1 {:.2f}%, prec@5 {:.2f}%, achieved at epoch {}".format(best_prec1, best_prec5, best_epoch))
        logging.info("Finished, total elapsed time (h:m:s): {}., training time (h:m:s): {}".format(elapsed, train_time))


def train_epoch(model, arch, epoch, optim_tools, train_loader, use_gpu):
    model.train()
    optimizer, criterion, scheduler = optim_tools

    losses, train_time, data_time = [AverageMeter() for _ in range(3)]
    st_time = time.time()
    for batch_idx, (img, label) in enumerate(train_loader):
        data_time.update(time.time() - st_time)
        if use_gpu:
            img, label = img.cuda(), label.cuda()

        output = model(img, arch)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if not args.distributed:
            losses.update(loss.item(), img.size(0))

        torch.cuda.synchronize()
        train_time.update(time.time() - st_time)

        if batch_idx % args.disp_freq == 0 or batch_idx == len(train_loader) - 1:
            if args.distributed:
                reduced_loss = reduce_tensor(loss.detach(), args.world_size)
                losses.update(reduced_loss.item(), img.size(0))

            if args.local_rank == 0:
                lr = scheduler.get_lr()[0]
                logging.info('Epoch: [{}][{}/{}]\t'
                             'Learning rate: {:.2e}\t'
                             'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Train time: {train_time.val:.3f}s ({train_time.avg:.3f}s)\t'
                             'Load data time: {data_time.val:.4f}s ({data_time.avg:.4f}s)\t'
                             .format(epoch, batch_idx, len(train_loader) - 1, lr,
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

            torch.cuda.synchronize()
            val_time.update(time.time() - st_time)

            if args.local_rank == 0 and (batch_idx % args.disp_freq == 0 or batch_idx == len(loader) - 1):
                logging.info('Iter: [{}/{}]\t'
                             'Eval time: {:.4f}s\t'
                             'Prec@1: {:.2f}%\t'
                             'Prec@5: {:.2f}%\t'
                             .format(batch_idx, len(loader) - 1, val_time.avg, all_prec1.avg, all_prec5.avg))
            st_time = time.time()

    return all_prec1.avg, all_prec5.avg


if __name__ == '__main__':
    main()
