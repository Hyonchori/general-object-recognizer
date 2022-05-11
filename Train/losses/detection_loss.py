

def get_detection_losses(
        img_size,
        outputs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        origin_preds,
        dtype
):
    bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
    obj_preds = outputs[:, :, 4]
    cls_preds = outputs[:, :, 5:]

    # calculate targets
    print(labels.shape)
    mixup = labels.shape[2] > 5
    if mixup:
        label_cut = labels[..., :5]
    else:
        label_cut = labels
