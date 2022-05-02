import cv2

from Data.for_train.detection_dataset import COCODataset, AugmentedCOCODataset
from Data.for_train.detection_augmentations import TrainTransform


if __name__ == "__main__":
    val_dir = "/home/daton/Downloads/coco/val2017"
    val_path = "/home/daton/Downloads/coco/annotations_trainval2017/annotations/instances_val2017.json"

    preproc = TrainTransform(img_size=(720, 1280), swap=False)
    #dataset = COCODataset(val_dir, val_path, preproc=preproc)
    dataset = AugmentedCOCODataset(val_dir, val_path, preproc=preproc)

    for img0, img, targets, img_info, img_id in dataset:
        print(img.shape)
        print(img.dtype == "uint8")
        '''for t in targets:
            print(t, "??")
            cv2.rectangle(img, list(map(int, t[:2])), list(map(int, t[2:4])), [255, 0, 255], 2)
        '''
        cv2.imshow("img0", img0)
        cv2.imshow("img", img)
        cv2.waitKey(0)