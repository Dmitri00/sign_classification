import torch
import torch.nn as nn
import torch.optim as optim
from os import makedirs as mkdir


def setup_optim(model, args, device_name='cuda'):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    return criterion, optimizer, lr_scheduler
