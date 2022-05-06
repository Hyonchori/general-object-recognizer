import numpy as np


def segment2box(segment):
    # Convert 1 segment label to 1 box label, (xy1, xy2, ...) -> (xyxy)
    x, y = segment.T
    return np.array([x.min(), y.min(), x.max(), y.max()])
