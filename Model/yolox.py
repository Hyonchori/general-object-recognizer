# YOLOx for object detection
import torch.nn as nn

from .backbones.darknet import CSPDarknet
from .necks.pafpn import PAFPN
from .heads.yolox_head import YOLOXHead


class YOLOX(nn.Module):
    def __init__(
            self,
            name: str = "YOLOx",
            backbone: CSPDarknet = None,
            neck: PAFPN = None,
            head: YOLOXHead = None
    ):
        super().__init__()
        if backbone is None:
            backbone = CSPDarknet()
        if neck is None:
            neck = PAFPN()
        if head is None:
            head = YOLOXHead(num_classes=80)  # for COCO dataset
        assert len(backbone.out_features) == len(neck.in_features) and \
               len(neck.out_features) == len(head.in_channels)
        self.name = name
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x
