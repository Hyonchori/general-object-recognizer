import torch

from Data.for_train.coco_dataset import COCODataset
from Data.for_train.augmented_dataset import AugmentedDataset
from Data.data_utils import Preprocessing, plot_labels

from Model.yolox import YOLOX
from Model.model_utils import model_info

from Train.losses.detection_loss import get_detection_losses

if __name__ == "__main__":
    model = YOLOX(name="yolox")
    #model.eval()
    model_info(model)

    val_dir = "/home/daton/Downloads/coco/val2017"
    val_path = "/home/daton/Downloads/coco/annotations_trainval2017/annotations/instances_val2017.json"

    val_dir = "/media/jhc/4AD250EDD250DEAF/dataset/coco/val2017"
    val_path = "/media/jhc/4AD250EDD250DEAF/dataset/coco/annotations_trainval2017/annotations/instances_val2017.json"

    img_size = (720, 1280)
    preproc = Preprocessing(img_size=img_size)
    dataset = COCODataset(val_dir, val_path, img_size=img_size, preproc=preproc)
    dataset = AugmentedDataset(dataset, img_size=img_size, preproc=preproc)

    for img0, img, labels0, labels, img_info, img_id in dataset:
        print(img.shape)
        img = torch.from_numpy(img)[None]
        print(img.shape)
        pred = model(img)

        if model.training:
            outputs, x_shifts, y_shifts, expanded_strides, origin_preds, dtype = pred
            losses = get_detection_losses(
                img.shape[2:],
                outputs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels["bbox"],
                origin_preds,
                dtype
            )
        plot_labels(img0, labels0)


