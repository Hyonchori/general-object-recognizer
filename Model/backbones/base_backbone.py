from typing import List
import torch.nn as nn


class BaseBackbone(nn.Module):
    """
    image: tensor(bn, C, W, H) -> feature map: List[tensor(bn, c, w, h), ...]
    """

    def __init__(
            self,
            depth_mul: float = 1.0,
            width_mul: float = 1.0,
            out_features: List[str] = None
    ):
        super().__init__()
        self.depth_mul = depth_mul
        self.width_mul = width_mul
        self.out_features = out_features

    """
        def forward(self, x):
            input: 
                images(bs, c, h, w)

            output :
                list of feature map or last feature map

            outputs = {}
            x = self.conv1(x)
            outputs["f1"] = x
            x = self.conv2(x)
            outputs["f2"] = x
            ...
            if self.out_features is not None:
                return {k: v for k, v in outputs.items() if k in self.out_features}
            else:
                return x
        """
