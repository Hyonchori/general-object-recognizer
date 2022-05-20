import torch
import torch.nn as nn


warmup_epochs = 3
warmup_lr = 0
basic_lr_per_img = 0.01 / 64.0
momentum = 0.9
weight_decay = 5e-4

def get_optimizer(batch_size, epoch, model: nn.Module):
    if epoch <= warmup_epochs:
        lr = warmup_lr
    else:
        lr = basic_lr_per_img * batch_size

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

    for k, v in model.named_modules():
        if hasattr(v, "bisa") and isinstance(v.bisa, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    optimizer = torch.optim.SGD(
        pg0, lr=lr, momentum=momentum, nesterov=True
    )
    optimizer.add_param_group(
        {"params": pg1, "weight_decay": weight_decay}
    )  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})
    return optimizer
