import torch

from Data.for_train.coco_dataset import COCODataset
from Data.for_train.augmented_dataset import AugmentedDataset
from Data.data_utils import Preprocessing, plot_labels

from Model.backbones.darknet import CSPDarknet
from Model.necks.pafpn import PAFPN
from Model.heads.yolox_head import YOLOXHead
from Model.yolox import YOLOX

if __name__ == "__main__":
    '''val_dir = "/home/daton/Downloads/coco/val2017"
    val_path = "/home/daton/Downloads/coco/annotations_trainval2017/annotations/instances_val2017.json"

    val_dir = "/media/jhc/4AD250EDD250DEAF/dataset/coco/val2017"
    val_path = "/media/jhc/4AD250EDD250DEAF/dataset/coco/annotations_trainval2017/annotations/instances_val2017.json"

    img_size = (720, 1280)
    preproc = Preprocessing(img_size=img_size)
    dataset = COCODataset(val_dir, val_path, img_size=img_size, preproc=preproc)
    dataset = AugmentedDataset(dataset, img_size=img_size, preproc=preproc)

    for img0, img, labels0, labels, img_info, img_id in dataset:
        plot_labels(img0, labels0)'''

    backbone = CSPDarknet()
    is1 = torch.randn((2, 3, 736, 1280))
    bbo = backbone(is1)
    for k, o in bbo.items():
        print(k)
        print(o.shape)
    print("")

    neck = PAFPN()
    no = neck(bbo)
    for k, o in no.items():
        print(k)
        print(o.shape)
    print("")

    head = YOLOXHead(num_classes=80)
    head.eval()
    ho = head(no)
    print(ho.shape)

    model = YOLOX()
    model.eval()
    mo = model(is1)
    print(mo.shape)
