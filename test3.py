
from Data.for_train.coco_dataset import COCODataset
from Data.for_train.augmented_dataset import AugmentedDataset


if __name__ == "__main__":
    val_dir = "/home/daton/Downloads/coco/val2017"
    ann_path = "/home/daton/Downloads/coco/annotations_trainval2017/annotations/instances_val2017.json"

    dataset = COCODataset(val_dir, ann_path)
    dataset = AugmentedDataset(dataset)

    for item in dataset:
        print(item)
        break
