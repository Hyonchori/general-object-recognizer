# Neck of YOLOx
from typing import List

import torch
import torch.nn as nn

from .base_neck import BaseNeck
from ..backbones.base_backbone import BaseBackbone
from ..backbones.darknet import CSPDarknet
from ..building_blocks import BaseConv, CSPLayer, DWConv


class PAFPN(BaseNeck):
    def __init__(
            self,
            backbone: BaseBackbone = None,
            depth_mul: float = 1.0,
            width_mul: float = 1.0,
            in_features: List[str] = ("dark3", "dark4", "dark5"),
            in_channels: List[int] = (256, 512, 1024),
            out_features: List[str] = ("pan2", "pan1", "pan0"),
            depthwise: bool = False,
            act: str = "silu"
    ):
        if backbone is None:
            backbone = CSPDarknet(
                depth_mul,
                width_mul,
                out_features=in_features,
                depthwise=depthwise,
                act=act
            )
        super().__init__(backbone, depth_mul, width_mul, in_features, out_features)
        assert len(in_features) == len(in_channels), \
            "Length of in_features and in_channels should be same"
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width_mul), int(in_channels[1] * width_mul), 1, 1, act=act
        )
