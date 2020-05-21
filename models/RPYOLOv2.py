from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, LeakyReLU, MaxPool2d, Softmax

from models.originals import YOLOv2


class RPYOLOv2(YOLOv2):
    def __init__(self):
        super().__init__(class_count=120, detection_grid_size=(13, 13), bounding_boxes_per_cell=5)
