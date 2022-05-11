from copy import deepcopy

import torch
from thop import profile


def model_info(model, img_size=1280, verbose=False):
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    stride = 32
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device).type_as(next(model.parameters()))
    flops = profile(deepcopy(model), inputs=(img, ), verbose=False)[0]
    img_size = img_size if isinstance(img_size, list) else [img_size, img_size]
    flops *= img_size[0] / stride * img_size[1] / stride * 2 / 1e9
    n_p /= 1e6
    n_g /= 1e6
    model_name = model.name if hasattr(model, "name") else "Model"
    print(f"{model_name} Summary: {len(list(model.modules()))} layers, " +
          f"{n_p:.1f}M parameters, {n_g:.1f}M gradients, {flops:.2f} GFLOPs given image size {img_size}")