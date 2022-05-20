# YOLOv5 reference

import os
import time
from copy import deepcopy

import numpy as np
import thop
import torch
from torch.cuda import amp


def check_train_batch_size(model, img_size=1280):
    with amp.autocast():
        return autobatch(deepcopy(model).train(), img_size)


def autobatch(
        model: torch.nn.Module,
        img_size: int = (736, 1280),
        fraction: float = 0.9,
        batch_sizes: list = (1, 2, 4, 8, 16),
        half: bool = True
):
    # Usage
    #   import torch
    #   from Train.train_utils import autobatch
    #   model = ...
    #   print(autobatch(model))

    print(f"Computing optimal batch size for --img-size {img_size}")
    device = next(model.parameters()).device

    gb = 1 << 30  # bytes to GiB (1024 ** 3)
    d = str(device).upper()  # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)
    t = properties.total_memory / gb  # (GiB)
    r = torch.cuda.memory_reserved(device) / gb  # (GiB)
    a = torch.cuda.memory_allocated(device) / gb  # (GiB)
    f = t - (r + a)
    print(f"{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free")

    img = [torch.zeros(b, 3, img_size[0], img_size[1]).half() if half else torch.zeros(b, 3, img_size[0], img_size[1])
           for b in batch_sizes]
    y = profile(img, model, n=3, device=device)
    y = [x[2] for x in y if x]
    batch_sizes = batch_sizes[:len(y)]
    p = np.polyfit(batch_sizes, y, deg=1)
    b = int((f * fraction - p[1]) / p[0])
    dtype = "float" if not half else "half"
    print(f"Using batsh-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) using {dtype} type")
    return b


def profile(inputs, ops, device, n=10):
    results = []
    print(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
          f"{'input':>24s}{'output':>24s}")

    for x in inputs if isinstance(inputs, list) else [inputs]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, "to") else m
            m = m.half() if hasattr(m, "half") and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1e9 * 2
            except Exception:
                flops = 0
            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    y = y[0] if isinstance(y, tuple) else y
                    t[1] = time_sync()
                    _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                    t[2] = time_sync()

                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # GB
                s_in = tuple(x.shape) if isinstance(x, torch.Tensor) else "list"
                s_out = tuple(y.shape) if isinstance(y, torch.Tensor) else "list"
                p = sum(list(x.numel() for x in m.parameters())) if isinstance(m, torch.nn.Module) else 0  # parameters
                print(f'{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}')
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def get_total_and_free_memory_in_Mb(cuda_device):
    devices_info_str = os.popen(
        "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader"
    )
    devices_info = devices_info_str.read().strip().split("\n")
    if isinstance(cuda_device, torch.device):
        cuda_device = int(str(cuda_device).split(":")[-1])
    total, used = devices_info[int(cuda_device)].split(",")
    return int(total), int(used)


def occupy_memory(cuda_device, mem_ratio=0.95):
    """
    pre-allocate gpu memory for training to avoid memory Fragmentation.
    """
    if isinstance(cuda_device, torch.device):
        cuda_device = int(str(cuda_device).split(":")[-1])
    total, used = get_total_and_free_memory_in_Mb(cuda_device)
    max_mem = int(total * mem_ratio)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x
    time.sleep(5)
