import torch
import torch.nn as nn


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


def create_optimizer(model, args):
    if args.optimizer == 'sgd':
        return torch.optim.SGD(
            get_parameters(model), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


def create_criterion(args):
    if args.label_smooth > 0:
        return CrossEntropyLabelSmooth(args.num_classes, args.label_smooth)
    else:
        return nn.CrossEntropyLoss()


def create_scheduler(optimizer, args):
    if args.scheduler == 'constant':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    elif args.scheduler == 'step':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (1.0 - step / args.num_sched_iters))
    elif args.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_sched_iters, args.min_lr)
