from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, LeakyReLU, MaxPool2d, Softmax

from models.modules import Identity, GlobalAvgPool2d, DetectionAdditionalLayers


class YOLOv2(Module):
    """Implementation of YOLOv2 model.

    Initializing this model would require information regarding available classes in the dataset and how many bounding
    boxes to generate per grid cell as specified in YOLOv2 publication since this model is a direct implementation of
    YOLOv2.

    Two types are available: recognizing and detecting (include recognizing). To set the model to recognizing task,
    simply call method ``model.recognizing(True)``, and to set the model to detection task call ``model.recognizing(
    False)``. These invocations will respectively select the model's structure, so no further modifications are
    required during training or inference. However, note that this class is designed to be used for recognition first
    then detection, hence changing from detection to recognition after successful training for recognition then
    detection would require users to ``load_dict`` the model's weight again since the detection layers would be removed.

    The flow of data can be split into two groups, where ``S`` denotes Section, ``P`` denotes Pool, ``pred`` denotes
    predictor, ``lX`` denotes layer_X, ``pass`` denotes passthrough, and ``AX`` denotes activation from layer X:
        Recognition:
            S1-P1-S2-P2-S3-P3-S4-P4-S5-P5-S6-pred
        Detection:
            S1-P1-S2-P2-S3-P3-S4-P4-S5-P5-S6-l19-l20-pass-l21-pred
                                     |__(A13)_______|
    , where A13 is the passthroughed input to passthrough layer, as specified by YOLOv2 config file and its publication.


    Arguments:
        class_count: Specifies the number of classes to be detected and recognized by the model.
        detection_grid_size: Also known as S in YOLO publications, this argument specifies the grid size measurement.
        bounding_boxes_per_cell: Also known as B in YOLO publications, this argument specifies how many bounding
        boxes will be generated per grid cell.
    """

    def __init__(self, class_count: int, detection_grid_size: tuple = (13, 13), bounding_boxes_per_cell: int = 5):
        super().__init__()

        self._is_recognition = True
        self.class_count = class_count
        self.detection_grid_size = detection_grid_size
        self.bounding_boxes_per_cell = bounding_boxes_per_cell

        self.backbone = self._create_backbone()
        self.predictor = self._select_predictor()

    @staticmethod
    def _validate_create_section_params(in_channels, out_channels):
        if in_channels is None:
            raise ValueError('in_channels must be specified')
        if out_channels is None:
            raise ValueError('out_channels must be specified')

    @staticmethod
    def _add_modules(sequential: Sequential, index, in_channels, out_channels, kernel_size, padding=1):
        sequential.add_module(f'Conv{index}',
                              Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     padding=padding))
        sequential.add_module(f'BatchNorm{index}', BatchNorm2d(num_features=out_channels))
        sequential.add_module(f'Activation{index}', LeakyReLU(negative_slope=.1, inplace=True))

    @staticmethod
    def _create_initial_sections(index=0, in_channels=None, out_channels=None):
        YOLOv2._validate_create_section_params(in_channels, out_channels)

        section = Sequential()
        YOLOv2._add_modules(section, index, in_channels, out_channels, kernel_size=3)

        return section

    @staticmethod
    def _create_middle_sections(index=0, in_channels=None, out_channels=None):
        YOLOv2._validate_create_section_params(in_channels, out_channels)

        section = Sequential()
        YOLOv2._add_modules(section, index, in_channels, out_channels, kernel_size=3)
        YOLOv2._add_modules(section, index + 1, out_channels, in_channels, kernel_size=1, padding=0)
        YOLOv2._add_modules(section, index + 2, in_channels, out_channels, kernel_size=3)

        return section

    @staticmethod
    def _create_final_sections(index=0, in_channels=None, out_channels=None):
        YOLOv2._validate_create_section_params(in_channels, out_channels)

        section = Sequential()
        YOLOv2._add_modules(section, index, in_channels, out_channels, kernel_size=3)
        YOLOv2._add_modules(section, index + 1, out_channels, in_channels, kernel_size=1, padding=0)
        YOLOv2._add_modules(section, index + 2, in_channels, out_channels, kernel_size=3)
        YOLOv2._add_modules(section, index + 3, out_channels, in_channels, kernel_size=1, padding=0)
        YOLOv2._add_modules(section, index + 4, in_channels, out_channels, kernel_size=3)

        return section

    def _create_backbone(self):
        backbone = Sequential()

        section = self._create_initial_sections(index=1, in_channels=3, out_channels=32)
        backbone.add_module('Section1', section)
        backbone.add_module('Pool1', MaxPool2d(kernel_size=2, stride=2))

        section = self._create_initial_sections(index=2, in_channels=32, out_channels=64)
        backbone.add_module('Section2', section)
        backbone.add_module('Pool2', MaxPool2d(kernel_size=2, stride=2))

        section = self._create_middle_sections(index=3, in_channels=64, out_channels=128)
        backbone.add_module('Section3', section)
        backbone.add_module('Pool3', MaxPool2d(kernel_size=2, stride=2))

        section = self._create_middle_sections(index=6, in_channels=128, out_channels=256)
        backbone.add_module('Section4', section)
        backbone.add_module('Pool4', MaxPool2d(kernel_size=2, stride=2))

        section = self._create_final_sections(index=9, in_channels=256, out_channels=512)
        backbone.add_module('Section5', section)
        backbone.add_module('Pool5', MaxPool2d(kernel_size=2, stride=2))

        section = self._create_final_sections(index=14, in_channels=512, out_channels=1024)
        backbone.add_module('Section6', section)

        return backbone

    def _select_predictor(self):
        predictor = Sequential()

        if self._is_recognition:
            predictor.add_module(f'ConvPredictor',
                                 Conv2d(self.backbone.Section6.Conv18.out_channels, self.class_count, kernel_size=1,
                                        padding=0))
            predictor.add_module(f'ActivationPredictor', Identity())
            predictor.add_module('GlobalAveragePooling', GlobalAvgPool2d())
            predictor.add_module('Softmax', Softmax(dim=1))
        else:
            out_channels = self.bounding_boxes_per_cell * (self.class_count + 5)
            predictor.add_module(f'ConvPredictor',
                                 Conv2d(in_channels=1024, out_channels=out_channels, kernel_size=1, padding=0))

        return predictor

    def recognizing(self, is_recognition=True):
        self._is_recognition = is_recognition

        if is_recognition and 'DetectionAdditionalLayers' in self.backbone.__dict__['_modules'].keys():
            self.backbone.__delattr__('DetectionAdditionalLayers')
        else:
            self.backbone.add_module('DetectionAdditionalLayers',
                                     DetectionAdditionalLayers(self.backbone.Section5.Conv13.out_channels))

        self.predictor = self._select_predictor()

    def forward(self, input: Tensor):
        return self._forward_recognition(input) if self._is_recognition else self._forward_detection(input)

    def _forward_recognition(self, x):
        out = self.backbone(x)
        out = self.predictor(out)
        return out

    def _forward_detection(self, x):
        x = self.backbone.Section1(x)
        x = self.backbone.Pool1(x)
        x = self.backbone.Section2(x)
        x = self.backbone.Pool2(x)
        x = self.backbone.Section3(x)
        x = self.backbone.Pool3(x)
        x = self.backbone.Section4(x)
        x = self.backbone.Pool4(x)
        x = self.backbone.Section5(x)
        from_layer_13 = x
        to_concat = self.backbone.Pool5(x)
        to_concat = self.backbone.Section6(to_concat)
        out = self.backbone.DetectionAdditionalLayers(from_layer_13, to_concat)
        out = self.predictor(out)
        return out

    def is_recognizing(self):
        return self._is_recognition
