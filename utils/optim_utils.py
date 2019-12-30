import torch
import torch.nn as nn
import math
from torch.optim.lr_scheduler import LambdaLR


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            group_weight_decay.append(p)
        else:
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(group_weight_decay) + len(group_no_weight_decay)
    groups = [dict(params=group_weight_decay), dict(params=group_no_weight_decay, weight_decay=0.)]
    return groups


def create_optimizer(model, optim_type, lr, weight_decay, momentum=0.9):
    assert optim_type in ['sgd']

    if optim_type == 'sgd':
        return torch.optim.SGD(get_parameters(model), lr=lr, momentum=momentum, weight_decay=weight_decay)


def create_criterion(num_classes, label_smooth=0):
    if label_smooth > 0:
        return CrossEntropyLabelSmooth(num_classes, label_smooth)
    else:
        return nn.CrossEntropyLoss()


def create_scheduler(optimizer, sched_type, total_steps, warmup_steps=0, num_cycles=.5, last_epoch=-1):
    assert sched_type in ['step', 'cosine']

    if sched_type == 'step':
        def lr_lambda(cur_step):
            if cur_step < warmup_steps:
                return float(cur_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - cur_step) / float(max(1, total_steps - warmup_steps)))

        return LambdaLR(optimizer, lr_lambda, last_epoch)
    elif sched_type == 'cosine':
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0., 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2. * progress)))

        return LambdaLR(optimizer, lr_lambda, last_epoch)
