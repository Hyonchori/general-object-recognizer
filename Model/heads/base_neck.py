from typing import List
import torch.nn as nn

from ..backbones.base_backbone import BaseBackbone


class BaseNeck(nn.Module):
    """
    feature map: List[tensor(bn, C, W, H), ...] -> feature map: List[tensor(bn, c, w, h), ...]
    """

    def __init__(
            self,
            backbone: BaseBackbone,
            depth_mul: float = 1.0,
            width_mul: float = 1.0,
            in_features: List[str] = None,
            out_features: List[str] = None
    ):
        super().__init__()
        assert len(backbone.out_features) == len(in_features), \
            f"Length of backbone's out_features({backbone.out_features}) and neck's in_feature({in_features})"
        self.backbone = backbone
        self.depth_mul = depth_mul
        self.width_mul = width_mul
        self.in_features = in_features
        self.out_features = out_features

    """
       def forward(self, x):
           input: 
               list of feature map

           output :
               list of feature map or last feature map

           outputs = {}
           out_features = self.backbone(x)
           features = [out_features[f] for f in self.in_features]
           x = self.conv1(features[0])
           outputs["f1"] = x
           x = self.conv2(features[1])
           outputs["f2"] = x
           ...
           if self.out_features is not None:
               return {k: v for k, v in outputs.items() if k in self.out_features}
           else:
               return x
       """
