from torch import Tensor, min as torch_min, max as torch_max
from torch.nn import Module
from torch.nn.functional import avg_pool2d as torch_avg_pool2d


class Identity(Module):
    """Identity activation due to unavailability in this project's PyTorch version.

    This is implemented as shown in https://github.com/pytorch/pytorch/pull/19249 by @MilesCranmer. All credits to
    @MilesCranmer.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


class Reorg(Module):
    """Reorg layer as used in YOLOv2 rewrite in PyTorch.

    This is implemented as shown in https://github.com/marvis/pytorch-yolo2. Modifications are made for variable
    names only. All credits to @marvis.
    """

    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, input_feature_map: Tensor):
        assert (input_feature_map.data.dim() == 4)
        batch = input_feature_map.data.size(0)
        channels = input_feature_map.data.size(1)
        height = input_feature_map.data.size(2)
        width = input_feature_map.data.size(3)

        assert (height % self.stride == 0)
        assert (width % self.stride == 0)
        width_stride = self.stride
        height_stride = self.stride

        output_height = int(height / height_stride)
        output_width = int(width / width_stride)
        output_feature_map = input_feature_map.view(batch, channels, output_height, height_stride, output_width,
                                                    width_stride).transpose(3, 4).contiguous()
        output_feature_map = output_feature_map.view(batch, channels, output_height * output_width,
                                                     height_stride * width_stride).transpose(2, 3).contiguous()
        output_feature_map = output_feature_map.view(batch, channels, height_stride * width_stride, output_height,
                                                     output_width).transpose(1, 2).contiguous()
        output_feature_map = output_feature_map.view(batch, height_stride * width_stride * channels, output_height,
                                                     output_width)

        return output_feature_map


class GlobalAvgPool2d(Module):
    """GlobalAvgPool2d layer as used in YOLOv2 rewrite in PyTorch.

    This is implemented as shown in https://github.com/marvis/pytorch-yolo2. Modifications are made for variable
    names only. All credits to @marvis.
    """

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, input: Tensor):
        batch = input.data.size(0)
        channels = input.data.size(1)
        height = input.data.size(2)
        width = input.data.size(3)

        output = torch_avg_pool2d(input, (height, width))
        output = output.view(batch, channels)

        return output


def _bbox_ious(box1, box2, is_corner_coordinates=True):
    """Calculation of intersection over union function **for many predictions and ground truths** as used in YOLO
    rewrite in PyTorch.

    This is implemented as shown in https://github.com/CharlesPikachu/YOLO. Modifications are made for variable names
    only. All credits to @CharlesPikachu.
    """

    x_left = torch_min(box1[0], box2[0]) if is_corner_coordinates else torch_min(box1[0] - box1[2] / 2.0,
                                                                                 box2[0] - box2[2] / 2.0)
    x_right = torch_max(box1[2], box2[2]) if is_corner_coordinates else torch_max(box1[0] + box1[2] / 2.0,
                                                                                  box2[0] + box2[2] / 2.0)
    y_top = torch_min(box1[1], box2[1]) if is_corner_coordinates else torch_min(box1[1] - box1[3] / 2.0,
                                                                                box2[1] - box2[3] / 2.0)
    y_bottom = torch_max(box1[3], box2[3]) if is_corner_coordinates else torch_max(box1[1] + box1[3] / 2.0,
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


def _bbox_iou(box1, box2, is_corner_coordinates=True):
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
