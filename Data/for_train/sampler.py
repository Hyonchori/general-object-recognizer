# YOLOx reference

import itertools
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from torch.utils.data.sampler import Sampler


class YOLOBatchSampler(torchBatchSampler):
    """
    It works just like the class: 'torch.utils.data.sampler.BatchSampler'
    """

    def __init__(self, *args, input_dim=None, mosaic=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.new_input_dim = None
        self.mosaic = mosaic
