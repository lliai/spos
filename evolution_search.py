import os
import torch
import numpy as np
import time
import datetime
import random
import logging
from copy import deepcopy
from utils import AverageMeter, accuracy, recalc_bn, reduce_tensor, save_search_history, get_flops


def evolution_search(model, val_loader, bn_loader, use_gpu, args):
    total_topk_pop, total_topk_prec1, total_topk_prec5, total_topk_flops = [], [], [], []
    st_time = time.time()

    num_crossover = args.pop_size // 2
    num_mutation = args.pop_size // 2
    pop = []
    evaluated_pop = []
    while len(pop) < args.pop_size:
        pop_gen = [random.randint(0, args.num_block_type - 1) for _ in range(sum(args.num_layer_list))]
        flops = get_flops(pop_gen, args.flop_table) / 1e6
        if pop_gen not in pop and flops <= args.max_flops:
            pop.append(pop_gen)
    evaluated_pop.extend(pop)

    if args.history_path:
        if os.path.isfile(args.history_path):
            history = torch.load(args.history_path)
            args.start_search_iter = history['iter'] + 1
            pop = history['pop']
            evaluated_pop = history['evaluated_pop']
            total_topk_pop = history['topk_pop']
            total_topk_prec1 = history['topk_prec1']
            total_topk_prec5 = history['topk_prec5']
            total_topk_flops = history['topk_flops']
            if args.local_rank == 0:
                logging.info('Loaded evolved population from \'{}\''.format(args.history_path))
                logging.info('Start iter: {}'.format(args.start_search_iter))
        else:
            if args.local_rank == 0:
                logging.info('No history file found in \'{}\''.format(args.history_path))

    if args.local_rank == 0:
        logging.info('==> Start evolution search')
    raw_params = deepcopy(model.state_dict())

    for itr in range(args.start_search_iter, args.total_search_iters + 1):
        all_prec1, all_prec5, all_flops = [], [], []
        for idx, arch in enumerate(pop):
            recalc_bn(model, arch, bn_loader, use_gpu, args.bn_recalc_imgs, args.world_size)
            prec1, prec5, flops = inference(model, itr, idx, arch, val_loader, use_gpu, args)
            model.load_state_dict(raw_params)
            all_prec1.append(prec1)
            all_prec5.append(prec5)
            all_flops.append(flops)

        topk_idx = np.argsort(all_prec1)[::-1][:args.topk]
        topk_pop = np.array(pop)[topk_idx].tolist()
        topk_prec1 = np.array(all_prec1)[topk_idx].tolist()
        topk_prec5 = np.array(all_prec5)[topk_idx].tolist()
        topk_flops = np.array(all_flops)[topk_idx].tolist()

        total_topk_pop.extend(topk_pop)
        total_topk_prec1.extend(topk_prec1)
        total_topk_prec5.extend(topk_prec5)
        total_topk_flops.extend(topk_flops)

        total_topk_idx = np.argsort(total_topk_prec1)[::-1][:args.topk]
        total_topk_pop = np.array(total_topk_pop)[total_topk_idx].tolist()
        total_topk_prec1 = np.array(total_topk_prec1)[total_topk_idx].tolist()
        total_topk_prec5 = np.array(total_topk_prec5)[total_topk_idx].tolist()
        total_topk_flops = np.array(total_topk_flops)[total_topk_idx].tolist()

        new_pop1 = crossover(total_topk_pop, num_crossover, evaluated_pop, args)
        evaluated_pop.extend(new_pop1)
        new_pop2 = mutation(total_topk_pop, num_mutation, evaluated_pop, args.mut_prob, args.num_block_type, args)
        evaluated_pop.extend(new_pop2)

        pop = []
        pop.extend(new_pop1)
        pop.extend(new_pop2)

        if args.local_rank == 0:
            logging.info('-' * 40)
            logging.info('Topk architectures:')
            for arch, flops, prec1, prec5 in zip(total_topk_pop, total_topk_flops, total_topk_prec1, total_topk_prec5):
                logging.info('Arch: {}\t'
                             'FLOPs: {:.2f}M\t'
                             'Prec@1: {:.2f}%\t'
                             'Prec@5: {:.2f}%'
                             .format(arch, flops, prec1, prec5))
            logging.info('-' * 40)

            state = {'iter': itr,
                     'pop': pop,
                     'evaluated_pop': evaluated_pop,
                     'topk_pop': total_topk_pop,
                     'topk_prec1': total_topk_prec1,
                     'topk_prec5': total_topk_prec5,
                     'topk_flops': total_topk_flops}
            history_name = 'history_iter' + str(itr) + '.bin'
            save_search_history(state, args.exp_dir, history_name, args.history_keep_num)

    elapsed = round(time.time() - st_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    if args.local_rank == 0:
        logging.info('Finished, total inference time (h:m:s): {}'.format(elapsed))


def inference(model, itr, pop_idx, arch, val_loader, use_gpu, args):
    model.eval()
    all_prec1 = AverageMeter()
    all_prec5 = AverageMeter()

    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(val_loader):
            if use_gpu:
                img, label = img.cuda(), label.cuda()

            output = model(img, arch)
            prec1, prec5 = accuracy(output, label, topk=(1, 5))

            if args.distributed:
                prec1 = reduce_tensor(prec1, args.world_size)
                prec5 = reduce_tensor(prec5, args.world_size)

            all_prec1.update(prec1.item(), img.size(0))
            all_prec5.update(prec5.item(), img.size(0))

        flops = get_flops(arch, args.flop_table) / 1e6
        if args.local_rank == 0:
            logging.info('Iter: [{}/{}][{}/{}]\t'
                         'Arch: {}\t'
                         'FLOPs: {:.2f}M\t'
                         'Prec@1: {:.2f}%\t'
                         'Prec@5: {:.2f}%'
                         .format(itr, args.total_search_iters, pop_idx + 1, args.pop_size, arch,
                                 flops, all_prec1.avg, all_prec5.avg))

    return all_prec1.avg, all_prec5.avg, flops


def crossover(raw_pop, num_pop, evaluated_pop, args):
    new_pop = []
    while len(new_pop) < num_pop:
        pop = deepcopy(raw_pop)
        id1, id2 = random.sample(range(len(pop)), 2)
        st_pos, ed_pos = random.sample(range(len(pop[0])), 2)
        if st_pos > ed_pos:
            st_pos, ed_pos = ed_pos, st_pos
        temp = pop[id1].copy()
        pop[id1][st_pos:ed_pos + 1] = pop[id2][st_pos:ed_pos + 1]
        pop[id2][st_pos:ed_pos + 1] = temp[st_pos:ed_pos + 1]

        flops1 = get_flops(pop[id1], args.flop_table) / 1e6
        flops2 = get_flops(pop[id2], args.flop_table) / 1e6

        if flops1 <= args.max_flops and pop[id1] not in new_pop and pop[id1] not in evaluated_pop:
            new_pop.append(pop[id1])
        if flops2 <= args.max_flops and pop[id2] not in new_pop and pop[id2] not in evaluated_pop \
                and len(new_pop) < num_pop:
            new_pop.append(pop[id2])

    return new_pop


def mutation(raw_pop, num_pop, evaluated_pop, mut_prob, num_candidates, args):
    new_pop = []
    while len(new_pop) < num_pop:
        pop = deepcopy(raw_pop)
        id = random.sample(range(len(pop)), 1)[0]
        for l in range(len(pop[0])):
            if random.random() < mut_prob:
                choices = [i for i in range(num_candidates) if i != pop[id][l]]
                pop[id][l] = random.sample(choices, 1)[0]

        flops = get_flops(pop[id], args.flop_table) / 1e6
        if flops <= args.max_flops and pop[id] not in new_pop and pop[id] not in evaluated_pop:
            new_pop.append(pop[id])

    return new_pop
