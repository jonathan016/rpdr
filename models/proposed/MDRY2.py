from typing import Optional, Union

from torch import Tensor, device
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, LeakyReLU, MaxPool2d, Upsample, CrossEntropyLoss

from .RPYOLOv3 import YOLOv3PredictorConfiguration
from models import GlobalAvgPool2d, YOLOLossSpecification, YOLOLoss, ConcatenatingRoute, YOLOv3Block


class MDRY2(Module):
    """Implementation of M-DRY2 model, the lovechild of YOLOv2 and YOLOv3, with some additions.

    Initializing this model would require information regarding available classes in the dataset and how many bounding
    boxes to generate per grid cell (set to default as 3) as specified in YOLOv3 publication since this model is a
    follows most implementation details of YOLOv3, with exception of its backbone. The backbone is a modified
    Darknet-53 network, with fewer convolution layers, addition of pooling layers, and skip/residual connections.
    This is hoped to increase the speed of this YOLOv2-YOLOv3-lovechild model while maintaining its performance in
    terms of image classification/object detection preciseness (accuracy).

    Two types are available: recognizing and detecting (include recognizing). To set the model to recognizing task,
    simply call method ``model.recognizing()``, and to set the model to detection task call ``model.detecting)``.
    These invocations will respectively select the model's structure, so no further modifications are required during
    training or inference. However, note that this class is designed to be used for recognition first then detection,
    hence changing from detection to recognition after successful training for recognition then detection would
    require users to ``load_dict`` the model's weight again since the detection layers would be removed.

    The flow of data can be split into two groups, where ``S`` denotes Section, ``P`` denotes Pool, ``pred`` denotes
    predictor, ``AX`` denotes activation from layer X, and ``large_predictor``, ``medium_predictor``,
    and ``small_predictor`` are the YOLO predictors for detection across scales:
        - Recognition::

                S1-P1-S2-P2-S3-P3-S4-P4-S5-P5-S6-pred
        - Detection::

                S1-P1-S2-P2-S3-P3-S4-P4-S5-P5-S6--large_predictor
                                   |    |       |
                                   |    |_(A17)_|__________
                                   |            |          |
                                   |            |__(A37)__medium_predictor
                                   |                        |
                                   |_________(A10)__________|__________
                                                            |          |
                                                            |__(A45)__small_predictor

    To calculate loss, simply call ``model.loss(target).backward()``, since the prediction values are kept after
    forward pass. Recognizing state means the model will use ``CrossEntropyLoss`` as specified in PJReddie's
    implementation (https://github.com/pjreddie/darknet) on ``softmax_layer.c``'s ``forward_softmax_layer`` function, in
    ``softmax_x_ent_cpu`` method call from ``blas.c``, where ``error[i] = (t) ? -log(p) : 0;``. In detecting state,
    the model will use ``YOLOLoss`` layer - in this case, the ``version=3`` ``YOLOLoss`` layer.

    For using this model in CUDA devices, if possible please **only use the** ``.cuda()`` **method** as it automatically
    handles ``YOLOLoss`` transfer to CUDA device. If it is not possible, after calling ``.to()`` method, invoke the
    following: ::

            >>> model.to(cuda_device)   # Primary invocation
            >>> # Following invocations
            >>> model.use_cuda = True
            >>> if model.is_recognizing() is False:
            >>>     model.large_predictor_loss.set_cuda(True)
            >>>     model.medium_predictor_loss.set_cuda(True)
            >>>     model.small_predictor_loss.set_cuda(True)

    Arguments:
        class_count: Specifies the number of classes to be detected and recognized by the model.
        large_predictor_config: Specifies configuration for loss calculation for large_predictor.
        medium_predictor_config: Specifies configuration for loss calculation for medium_predictor.
        small_predictor_config: Specifies configuration for loss calculation for small_predictor.
        bounding_boxes_per_cell: Also known as B in YOLO publications, this argument specifies how many bounding
            boxes will be generated per grid cell.
    """

    # TODO Set anchor_boxes real values for all config
    def __init__(self, class_count: int = 120,
                 large_predictor_config: YOLOv3PredictorConfiguration = YOLOv3PredictorConfiguration(
                     anchor_boxes=[10, 13, 16, 30, 33, 23],
                     spec=YOLOLossSpecification(version=3, num_classes=120, max_object=15, is_multilabel=False)),
                 medium_predictor_config: YOLOv3PredictorConfiguration = YOLOv3PredictorConfiguration(
                     anchor_boxes=[30, 61, 62, 45, 59, 119],
                     spec=YOLOLossSpecification(version=3, num_classes=120, max_object=15, is_multilabel=False)),
                 small_predictor_config: YOLOv3PredictorConfiguration = YOLOv3PredictorConfiguration(
                     anchor_boxes=[116, 90, 156, 198, 373, 326],
                     spec=YOLOLossSpecification(version=3, num_classes=120, max_object=15, is_multilabel=False)),
                 bounding_boxes_per_cell: int = 3):
        super().__init__()

        self._is_recognition = True
        self.class_count = class_count
        self.bounding_boxes_per_cell = bounding_boxes_per_cell

        self.use_cuda = False

        self.large_predictor_loss = YOLOLoss(
            version=3, anchor_boxes=large_predictor_config.anchor_boxes, use_cuda=self.use_cuda,
            spec=large_predictor_config.spec, v3type='Large')
        self.medium_predictor_loss = YOLOLoss(
            version=3, anchor_boxes=medium_predictor_config.anchor_boxes, use_cuda=self.use_cuda,
            spec=medium_predictor_config.spec, v3type='Medium')
        self.small_predictor_loss = YOLOLoss(
            version=3, anchor_boxes=small_predictor_config.anchor_boxes, use_cuda=self.use_cuda,
            spec=small_predictor_config.spec, v3type='Small')

        self.backbone = self._create_backbone()
        self.predictor = self._select_predictor()

        self.recognition_prediction = None
        self.large_predictions = None
        self.medium_predictions = None
        self.small_predictions = None

    def _create_backbone(self):
        backbone = Sequential()

        section1 = Sequential()
        section1.add_module('Conv1', Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1))
        section1.add_module('BatchNorm1', BatchNorm2d(num_features=32))
        section1.add_module('Activation1', LeakyReLU(negative_slope=.1, inplace=True))

        backbone.add_module('Section1', section1)
        backbone.add_module('Pool1', MaxPool2d(kernel_size=2, stride=2))

        section2 = Sequential()
        section2.add_module('Conv1', Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1))
        section2.add_module('BatchNorm1', BatchNorm2d(num_features=64))
        section2.add_module('Activation1', LeakyReLU(negative_slope=.1, inplace=True))

        backbone.add_module('Section2', section2)
        backbone.add_module('Pool2', MaxPool2d(kernel_size=2, stride=2))

        backbone.add_module('Section3', YOLOv3Block(index=3, in_channels=64, out_channels=128,
                                                    convs_residual_count=3, initial_block_stride=1))
        backbone.add_module('Pool3', MaxPool2d(kernel_size=2, stride=2))

        backbone.add_module('Section4',
                            YOLOv3Block(index=10, in_channels=128, out_channels=256, convs_residual_count=3,
                                        initial_block_stride=1))
        backbone.add_module('Pool4', MaxPool2d(kernel_size=2, stride=2))

        backbone.add_module('Section5',
                            YOLOv3Block(index=17, in_channels=256, out_channels=512, convs_residual_count=4,
                                        initial_block_stride=1))
        backbone.add_module('Pool5', MaxPool2d(kernel_size=2, stride=2))

        backbone.add_module('Section6',
                            YOLOv3Block(index=26, in_channels=512, out_channels=1024, convs_residual_count=3,
                                        initial_block_stride=1))

        return backbone

    def _select_predictor(self):
        predictor = Sequential()

        if self._is_recognition:
            predictor.add_module(f'ConvPredictor',
                                 Conv2d(self.backbone.Section6.following_blocks[2].convs.Conv32.out_channels,
                                        self.class_count, kernel_size=1, padding=0))
            predictor.add_module('GlobalAveragePooling', GlobalAvgPool2d())
            # No softmax layer is added since CrossEntropyLoss, the used classification loss function, encapsulates
            # softmax layer in its loss calculation. See https://discuss.pytorch.org/t/vgg-output-layer-no-softmax/9273
            # for more detail. To get the output with highest value, simply call ``torch.max(output)``, or to get the
            # index of the output with highest value, call ``output.argmax()``.
        else:
            self._register_yolo_predictors(predictor)

        return predictor

    def _register_yolo_predictors(self, predictor_container: Sequential):
        large_predictor = self._create_yolov3_predictor(32, 1024, 'Large')
        medium_predictor = self._create_yolov3_predictor(40, 512, 'Medium')
        small_predictor = self._create_yolov3_predictor(48, 256, 'Small')

        predictor_container.add_module('large_predictor', large_predictor)
        predictor_container.add_module('medium_predictor', medium_predictor)
        predictor_container.add_module('small_predictor', small_predictor)

        return predictor_container

    def _create_yolov3_predictor(self, index, channels, type):
        predictor = Sequential()
        inner_channels = int(channels / 2)
        is_not_large_predictor = type != 'Large'

        if is_not_large_predictor:
            predictor.add_module(f'Conv{index}',
                                 Conv2d(in_channels=channels, out_channels=inner_channels, kernel_size=1, stride=1,
                                        padding=0))
            predictor.add_module(f'BatchNorm{index}', BatchNorm2d(num_features=inner_channels))
            predictor.add_module(f'Activation{index}', LeakyReLU(negative_slope=.1, inplace=True))
            predictor.add_module('Upsample', Upsample(scale_factor=2, mode='nearest'))
            predictor.add_module('Concatenate', ConcatenatingRoute())

        for i in range(1, 4):
            concatenated_channels = (inner_channels + channels)
            in_channels = concatenated_channels if is_not_large_predictor and i == 1 else channels
            predictor.add_module(f'Conv{index + i * 2 - 1}',
                                 Conv2d(in_channels=in_channels, out_channels=inner_channels, kernel_size=1, stride=1,
                                        padding=0))
            predictor.add_module(f'BatchNorm{index + i * 2 - 1}', BatchNorm2d(num_features=inner_channels))
            predictor.add_module(f'Activation{index + i * 2 - 1}', LeakyReLU(negative_slope=.1, inplace=True))

            predictor.add_module(f'Conv{index + i * 2}',
                                 Conv2d(in_channels=inner_channels, out_channels=channels, kernel_size=3, stride=1,
                                        padding=1))
            predictor.add_module(f'BatchNorm{index + i * 2}', BatchNorm2d(num_features=channels))
            predictor.add_module(f'Activation{index + i * 2}', LeakyReLU(negative_slope=.1, inplace=True))

        out_channels = self.bounding_boxes_per_cell * (self.class_count + 5)
        predictor.add_module(f'Conv{index + (i + 1) * 2 - 1}',
                             Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=1, padding=0))

        return predictor

    def recognizing(self, is_recognition=True):
        self._is_recognition = is_recognition
        self.predictor = self._select_predictor()

    def detecting(self):
        self.recognizing(False)

    def forward(self, input: Tensor):
        return self._forward_recognition(input) if self._is_recognition else self._forward_detection(input)

    def _forward_recognition(self, x):
        x = self.backbone(x)
        out = self.predictor(x)
        self.recognition_prediction = out
        return out

    def _forward_detection(self, x):
        x = self.backbone.Section1(x)
        x = self.backbone.Pool1(x)
        x = self.backbone.Section2(x)
        x = self.backbone.Pool2(x)
        x = self.backbone.Section3(x)
        x = self.backbone.Pool3(x)
        x, from_layer_10 = self._forward_section4(x)
        x = self.backbone.Pool4(x)
        x, from_layer_17 = self._forward_section5(x)
        x = self.backbone.Pool5(x)
        x = self.backbone.Section6(x)
        large_predictions, medium_predictions, small_predictions = self._forward_predictors(x, from_layer_17,
                                                                                            from_layer_10)

        return [large_predictions, medium_predictions, small_predictions]

    def _forward_predictors(self, x, from_layer_17, from_layer_10):
        large_predictions, from_layer_37 = self._predict_large_objects(x)
        medium_predictions, from_layer_45 = self._predict_medium_objects(from_layer_37, from_layer_17)
        small_predictions = self._predict_small_objects(from_layer_45, from_layer_10)

        return large_predictions, medium_predictions, small_predictions

    def _forward_section5(self, x):
        residual = self.backbone.Section5.initial_block(x)
        x = self.backbone.Section5(x)

        return x, residual

    def _forward_section4(self, x):
        residual = self.backbone.Section4.initial_block(x)
        x = self.backbone.Section4(x)

        return x, residual

    def _predict_small_objects(self, input, to_concat):
        x = self.predictor.small_predictor.Conv48(input)
        x = self.predictor.small_predictor.BatchNorm48(x)
        x = self.predictor.small_predictor.Activation48(x)
        x = self.predictor.small_predictor.Upsample(x)
        x = self.predictor.small_predictor.Concatenate(x, to_concat)
        x = self.predictor.small_predictor.Conv49(x)
        x = self.predictor.small_predictor.BatchNorm49(x)
        x = self.predictor.small_predictor.Activation49(x)
        x = self.predictor.small_predictor.Conv50(x)
        x = self.predictor.small_predictor.BatchNorm50(x)
        x = self.predictor.small_predictor.Activation50(x)
        x = self.predictor.small_predictor.Conv51(x)
        x = self.predictor.small_predictor.BatchNorm51(x)
        x = self.predictor.small_predictor.Activation51(x)
        x = self.predictor.small_predictor.Conv52(x)
        x = self.predictor.small_predictor.BatchNorm52(x)
        x = self.predictor.small_predictor.Activation52(x)
        x = self.predictor.small_predictor.Conv53(x)
        x = self.predictor.small_predictor.BatchNorm53(x)
        x = self.predictor.small_predictor.Activation53(x)
        x = self.predictor.small_predictor.Conv54(x)
        x = self.predictor.small_predictor.BatchNorm54(x)
        x = self.predictor.small_predictor.Activation54(x)
        x = self.predictor.small_predictor.Conv55(x)
        self.small_predictions = x

        return x

    def _predict_medium_objects(self, input, to_concat):
        x = self.predictor.medium_predictor.Conv40(input)
        x = self.predictor.medium_predictor.BatchNorm40(x)
        x = self.predictor.medium_predictor.Activation40(x)
        x = self.predictor.medium_predictor.Upsample(x)
        x = self.predictor.medium_predictor.Concatenate(x, to_concat)
        x = self.predictor.medium_predictor.Conv41(x)
        x = self.predictor.medium_predictor.BatchNorm41(x)
        x = self.predictor.medium_predictor.Activation41(x)
        x = self.predictor.medium_predictor.Conv42(x)
        x = self.predictor.medium_predictor.BatchNorm42(x)
        x = self.predictor.medium_predictor.Activation42(x)
        x = self.predictor.medium_predictor.Conv43(x)
        x = self.predictor.medium_predictor.BatchNorm43(x)
        x = self.predictor.medium_predictor.Activation43(x)
        x = self.predictor.medium_predictor.Conv44(x)
        x = self.predictor.medium_predictor.BatchNorm44(x)
        x = self.predictor.medium_predictor.Activation44(x)
        x = self.predictor.medium_predictor.Conv45(x)
        x = self.predictor.medium_predictor.BatchNorm45(x)
        x = self.predictor.medium_predictor.Activation45(x)
        from_layer_45 = x
        x = self.predictor.medium_predictor.Conv46(x)
        x = self.predictor.medium_predictor.BatchNorm46(x)
        x = self.predictor.medium_predictor.Activation46(x)
        x = self.predictor.medium_predictor.Conv47(x)
        self.medium_predictions = x

        return x, from_layer_45

    def _predict_large_objects(self, x):
        x = self.predictor.large_predictor.Conv33(x)
        x = self.predictor.large_predictor.BatchNorm33(x)
        x = self.predictor.large_predictor.Activation33(x)
        x = self.predictor.large_predictor.Conv34(x)
        x = self.predictor.large_predictor.BatchNorm34(x)
        x = self.predictor.large_predictor.Activation34(x)
        x = self.predictor.large_predictor.Conv35(x)
        x = self.predictor.large_predictor.BatchNorm35(x)
        x = self.predictor.large_predictor.Activation35(x)
        x = self.predictor.large_predictor.Conv36(x)
        x = self.predictor.large_predictor.BatchNorm36(x)
        x = self.predictor.large_predictor.Activation36(x)
        x = self.predictor.large_predictor.Conv37(x)
        x = self.predictor.large_predictor.BatchNorm37(x)
        x = self.predictor.large_predictor.Activation37(x)
        from_layer_37 = x
        x = self.predictor.large_predictor.Conv38(x)
        x = self.predictor.large_predictor.BatchNorm38(x)
        x = self.predictor.large_predictor.Activation38(x)
        x = self.predictor.large_predictor.Conv39(x)
        self.large_predictions = x

        return x, from_layer_37

    def is_recognizing(self):
        return self._is_recognition

    def loss(self, target):
        if self._is_recognition:
            loss = CrossEntropyLoss(reduction='sum')(self.recognition_prediction, target)

            self.recognition_prediction = None
        else:
            if self.large_predictions is None or self.medium_predictions is None or self.small_predictions is None:
                raise RuntimeError('No predictions has been made')

            loss = self.large_predictor_loss(self.large_predictions, target)
            loss += self.medium_predictor_loss(self.medium_predictions, target)
            loss += self.small_predictor_loss(self.small_predictions, target)

            self.large_predictions = None
            self.medium_predictions = None
            self.small_predictions = None
        return loss

    def cuda(self, device: Optional[Union[int, device]] = ...):
        cuda_device = super().cuda(device)
        self.use_cuda = True
        if not self._is_recognition:
            self.large_predictor_loss.set_cuda(True)
            self.medium_predictor_loss.set_cuda(True)
            self.small_predictor_loss.set_cuda(True)
        return cuda_device

    def cpu(self):
        cpu_device = super().cpu()
        self.use_cuda = False
        if not self._is_recognition:
            self.large_predictor_loss.set_cuda(False)
            self.medium_predictor_loss.set_cuda(False)
            self.small_predictor_loss.set_cuda(False)
        return cpu_device
