from typing import List
import torch.nn as nn


class BaseDetector(nn.Module):
    """
    Integrate feature map and make detection (Bounding boxes, confidence, classes)
    from neck or backbone
    """
    def __init__(
            self,
            num_classes: int,
            width_mul: float = 1.0,
            strides: List[int] = (8, 16, 32),  # ratio between original image shape and feature map
            in_channels: List[int] = (256, 512, 1024)  # channels of each feature maps
    ):
        assert len(strides) == len(in_channels), \
            "Length of strides and in_channels should be same!"
        super().__init__()
        self.num_classes = num_classes
        self.width_mul = width_mul
        self.strides = strides  # used in making grid for prediction
        self.in_channels = in_channels

        '''
        self.convs = nn.ModuleList()
        for i in range(len(strides)):
            self.convs.append(
                Conv(
                    in_channels=self.in_channels[i]
                )
            )
            ...
        '''

    """
    def forward(self, feats):
        input: 
            list of feature map
    
        output :
            detection (bboxes, conf, cls)
    
        outputs = []
        for conv, feat in zip(self.convs, feats):
            x = conv(feat)
            outputs.append(x)
            ...
        return outputs
    """