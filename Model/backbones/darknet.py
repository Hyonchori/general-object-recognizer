# Darknet used in YOLOx
from typing import List
import torch.nn as nn

from .base_backbone import BaseBackbone
from ..building_blocks import BaseConv, CSPLayer, DWConv, Focus, SPPBottleNeck


class CSPDarknet(BaseBackbone):
    def __init__(
            self,
            depth_mul: float = 1.0,
            width_mul: float = 1.0,
            out_features: List[str] = ("dark3", "dark4", "dark5"),
            depthwise: bool = False,
            act: str = "silu"
    ):
        super().__init__(depth_mul, width_mul, out_features)
        Conv = DWConv if depthwise else BaseConv
        base_channels = int(self.width_mul * 64)
        base_depth = max(round(self.depth_mul * 3), 1)

        self.stem = Focus(3, base_channels, ksize=3, act=act)

        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act
            )
        )
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act
            )
        )
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act
            )
        )
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleNeck(base_channels * 16, base_channels * 16, act=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act
            )
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        if self.out_features is not None:
            return {k: v for k, v in outputs.items() if k in self.out_features}
        else:
            return x
