import cv2
import numpy as np

from ..Data.data_utils import segcls2seg


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


def plot_labels(img, labels, window_name="img", wait=False):
    img = img.copy()
    ref_img = np.zeros_like(img)
    for label in labels:
        if label == "bbox":
            # bbox: (n, 5) = ((x1, y1, x2, y2, cls), (x1, y1, x2, y2, cls), ...)
            for bbox in labels["bbox"]:
                color = colors(bbox[-1], True)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                              color, 2)
        elif label == "segmentation":
            # seg: (n, s + 1) = (x1, y1, x2, y2, ... cls)
            for seg_cls in labels["segmentation"]:
                seg = segcls2seg(seg_cls)
                cls = seg_cls[-1]
                color = colors(cls, True)
                cv2.fillPoly(ref_img, [seg.astype(np.int64)], color)
    img = cv2.addWeighted(img, 1, ref_img, 0.5, 0)
    cv2.imshow(window_name, img)
    if wait:
        cv2.waitKey(0)
