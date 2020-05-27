import torch


def intersection_over_union(many, first, second, is_corner_coordinates):
    if many:
        return __bbox_ious(first, second, is_corner_coordinates)
    return __bbox_iou(first, second, is_corner_coordinates)


def __bbox_ious(box1, box2, is_corner_coordinates=True):
    """Calculation of intersection over union function **for many predictions and ground truths** as used in YOLO
    rewrite in PyTorch.

    This is implemented as shown in https://github.com/CharlesPikachu/YOLO. Modifications are made for variable names
    only. All credits to @CharlesPikachu.
    """

    x_left = torch.min(box1[0], box2[0]) if is_corner_coordinates else torch.min(box1[0] - box1[2] / 2.0,
                                                                                 box2[0] - box2[2] / 2.0)
    x_right = torch.max(box1[2], box2[2]) if is_corner_coordinates else torch.max(box1[0] + box1[2] / 2.0,
                                                                                  box2[0] + box2[2] / 2.0)
    y_top = torch.min(box1[1], box2[1]) if is_corner_coordinates else torch.min(box1[1] - box1[3] / 2.0,
                                                                                box2[1] - box2[3] / 2.0)
    y_bottom = torch.max(box1[3], box2[3]) if is_corner_coordinates else torch.max(box1[1] + box1[3] / 2.0,
                                                                                   box2[1] + box2[3] / 2.0)

    box1_width = box1[2] - box1[0] if is_corner_coordinates else box1[2]
    box1_height = box1[3] - box1[1] if is_corner_coordinates else box1[3]
    box2_width = box2[2] - box2[0] if is_corner_coordinates else box2[2]
    box2_height = box2[3] - box2[1] if is_corner_coordinates else box2[3]

    raw_union_width = x_right - x_left
    raw_union_height = y_bottom - y_top
    intersection_width = box1_width + box2_width - raw_union_width
    intersection_height = box1_height + box2_height - raw_union_height
    mask = ((intersection_width <= 0) + (intersection_height <= 0) > 0)

    box1_area = box1_width * box1_height
    box2_area = box2_width * box2_height

    intersection = intersection_width * intersection_height
    intersection[mask] = 0
    union = box1_area + box2_area - intersection

    return intersection / union


def __bbox_iou(box1, box2, is_corner_coordinates=True):
    """Calculation of intersection over union function **for one prediction and ground truth ** as used in YOLO
    rewrite in PyTorch.

    This is implemented as shown in https://github.com/CharlesPikachu/YOLO. Modifications are made for variable names
    only. All credits to @CharlesPikachu.
    """

    x_left = min(box1[0], box2[0]) if is_corner_coordinates else min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
    x_right = max(box1[2], box2[2]) if is_corner_coordinates else max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
    y_top = min(box1[1], box2[1]) if is_corner_coordinates else min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
    y_bottom = max(box1[3], box2[3]) if is_corner_coordinates else max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)

    box1_width = box1[2] - box1[0] if is_corner_coordinates else box1[2]
    box1_height = box1[3] - box1[1] if is_corner_coordinates else box1[3]
    box2_width = box2[2] - box2[0] if is_corner_coordinates else box2[2]
    box2_height = box2[3] - box2[1] if is_corner_coordinates else box2[3]

    raw_union_width = x_right - x_left
    raw_union_height = y_bottom - y_top
    intersection_width = box1_width + box2_width - raw_union_width
    intersection_height = box1_height + box2_height - raw_union_height

    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0

    box1_area = box1_width * box1_height
    box2_area = box2_width * box2_height

    intersection = intersection_width * intersection_height
    union = box1_area + box2_area - intersection

    return intersection / union
