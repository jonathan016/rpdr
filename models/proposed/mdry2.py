from typing import Optional, Union

from torch import Tensor, LongTensor, device, linspace as torch_linspace, exp as torch_exp, sigmoid as torch_sigmoid, \
    max as torch_max
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, LeakyReLU, MaxPool2d, Upsample, Softmax, CrossEntropyLoss

from models import GlobalAvgPool2d, YOLOLossSpecification, YOLOLoss, ConcatenatingRoute, YOLOv3Block, nms
from .rpyolov3 import YOLOv3PredictorConfiguration


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

    def __init__(self, class_count: int = 120,
                 large_predictor_config: YOLOv3PredictorConfiguration = YOLOv3PredictorConfiguration(
                     anchor_boxes=[112, 158, 193, 86, 205, 247],
                     spec=YOLOLossSpecification(version=3, num_classes=120, max_object=15, is_multilabel=False)),
                 medium_predictor_config: YOLOv3PredictorConfiguration = YOLOv3PredictorConfiguration(
                     anchor_boxes=[59, 21, 84, 97, 95, 46],
                     spec=YOLOLossSpecification(version=3, num_classes=120, max_object=15, is_multilabel=False)),
                 small_predictor_config: YOLOv3PredictorConfiguration = YOLOv3PredictorConfiguration(
                     anchor_boxes=[19, 37, 35, 57, 52, 78],
                     spec=YOLOLossSpecification(version=3, num_classes=120, max_object=15, is_multilabel=False)),
                 bounding_boxes_per_cell: int = 3):
        super().__init__()

        self._is_recognition = True
        self.class_count = class_count
        self.bounding_boxes_per_cell = bounding_boxes_per_cell

        self.use_cuda = False

        num_backbone_sections = 6
        downsample_ratio = 2 ** num_backbone_sections
        self.large_predictor_loss = YOLOLoss(
            version=3, anchor_boxes=large_predictor_config.anchor_boxes, use_cuda=self.use_cuda,
            spec=large_predictor_config.spec, v3type=YOLOLoss.V3LARGE, downsample_ratio=downsample_ratio // (2 ** 1))
        self.medium_predictor_loss = YOLOLoss(
            version=3, anchor_boxes=medium_predictor_config.anchor_boxes, use_cuda=self.use_cuda,
            spec=medium_predictor_config.spec, v3type=YOLOLoss.V3MEDIUM, downsample_ratio=downsample_ratio // (2 ** 2))
        self.small_predictor_loss = YOLOLoss(
            version=3, anchor_boxes=small_predictor_config.anchor_boxes, use_cuda=self.use_cuda,
            spec=small_predictor_config.spec, v3type=YOLOLoss.V3SMALL, downsample_ratio=downsample_ratio // (2 ** 3))

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
        large_predictor = self._create_yolov3_predictor(32, 1024, YOLOLoss.V3LARGE)
        medium_predictor = self._create_yolov3_predictor(40, 512, YOLOLoss.V3MEDIUM)
        small_predictor = self._create_yolov3_predictor(48, 256, YOLOLoss.V3SMALL)

        predictor_container.add_module('large_predictor', large_predictor)
        predictor_container.add_module('medium_predictor', medium_predictor)
        predictor_container.add_module('small_predictor', small_predictor)

        return predictor_container

    def _create_yolov3_predictor(self, index, channels, type):
        predictor = Sequential()
        inner_channels = int(channels / 2)
        is_not_large_predictor = type != YOLOLoss.V3LARGE

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

        if self.use_cuda:
            self.predictor.cuda()
        else:
            self.predictor.cpu()

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

    def loss(self, target, divide_by_mask=False, class_loss_reduction='mean', location_confidence_loss_reduction='sum'):
        if self._is_recognition:
            loss = CrossEntropyLoss(reduction='sum')(self.recognition_prediction, target)

            self.recognition_prediction = None
        else:
            if self.large_predictions is None or self.medium_predictions is None or self.small_predictions is None:
                raise RuntimeError('No predictions has been made')

            kwargs = {
                'divide_by_mask': divide_by_mask,
                'class_loss_reduction': class_loss_reduction,
                'location_confidence_loss_reduction': location_confidence_loss_reduction
            }
            large_predictor_loss = self.large_predictor_loss(self.large_predictions, target, **kwargs)
            medium_predictor_loss = self.medium_predictor_loss(self.medium_predictions, target, **kwargs)
            small_predictor_loss = self.small_predictor_loss(self.small_predictions, target, **kwargs)

            loss = large_predictor_loss + medium_predictor_loss + small_predictor_loss

            self.large_predictions = None
            self.medium_predictions = None
            self.small_predictions = None
        return loss

    def eval(self):
        ev = super().eval()
        self.large_predictor_loss.eval()
        self.medium_predictor_loss.eval()
        self.small_predictor_loss.eval()
        return ev

    def train(self, mode: bool = True):
        tr = super().train(mode)
        self.large_predictor_loss.train(mode)
        self.medium_predictor_loss.train(mode)
        self.small_predictor_loss.train(mode)
        return tr

    def cuda(self, dev: Optional[Union[int, device]] = None):
        cuda_device = super().cuda(dev)
        self.use_cuda = True
        if not self._is_recognition:
            self.large_predictor_loss.cuda()
            self.medium_predictor_loss.cuda()
            self.small_predictor_loss.cuda()
        self.backbone.cuda()
        self.predictor.cuda()
        return cuda_device

    def cpu(self):
        cpu_device = super().cpu()
        self.use_cuda = False
        if not self._is_recognition:
            self.large_predictor_loss.cpu()
            self.medium_predictor_loss.cpu()
            self.small_predictor_loss.cpu()
        self.backbone.cpu()
        self.predictor.cpu()
        return cpu_device

    def detect_objects(self, image_as_tensor, confidence_threshold, nms_threshold):
        predictions = self.forward(image_as_tensor)
        yolov3_loss_functions = [
            self.large_predictor_loss.layer, self.medium_predictor_loss.layer, self.small_predictor_loss.layer]
        predictions_losses = zip(predictions, yolov3_loss_functions)

        batch_size = predictions[0].size(0)
        all_boxes = [[] for _ in range(batch_size)]
        for prediction, loss_layer in predictions_losses:
            num_anchors = loss_layer.num_anchors

            if prediction.dim() == 3:
                prediction = prediction.unsqueeze(0)

            assert prediction.size(1) == (5 + self.class_count) * num_anchors

            feature_map_height = prediction.size(2)
            feature_map_width = prediction.size(3)
            number_of_pixels = feature_map_height * feature_map_width
            view_size = batch_size * num_anchors * number_of_pixels

            output = prediction.view(batch_size * num_anchors, 5 + self.class_count, number_of_pixels).transpose(
                0, 1).contiguous().view(5 + self.class_count, view_size)
            output = output.cuda() if self.use_cuda else output

            grid_x = torch_linspace(0, feature_map_width - 1, feature_map_width).repeat(feature_map_height, 1).repeat(
                batch_size * num_anchors, 1, 1).view(view_size)
            grid_x = grid_x.cuda() if self.use_cuda else grid_x

            grid_y = torch_linspace(0, feature_map_height - 1, feature_map_height).repeat(
                feature_map_width, 1).t().repeat(batch_size * num_anchors, 1, 1).view(view_size)
            grid_y = grid_y.cuda() if self.use_cuda else grid_y

            anchor_w = Tensor(loss_layer.anchors).view(num_anchors, loss_layer.anchor_step).index_select(
                1, LongTensor([0])).repeat(batch_size, 1).repeat(1, 1, number_of_pixels).view(view_size)
            anchor_w = anchor_w.cuda() if self.use_cuda else anchor_w

            anchor_h = Tensor(loss_layer.anchors).view(num_anchors, loss_layer.anchor_step).index_select(
                1, LongTensor([1])).repeat(batch_size, 1).repeat(1, 1, number_of_pixels).view(view_size)
            anchor_h = anchor_h.cuda() if self.use_cuda else anchor_h

            class_scores = Softmax()(output[5:5 + self.class_count].transpose(0, 1)).data
            max_class_scores, top_classes = torch_max(class_scores, 1)

            objectness_confidences = torch_sigmoid(output[4])
            max_class_scores = max_class_scores.view(-1)
            top_classes = top_classes.view(-1)

            confidences = objectness_confidences * max_class_scores
            objectness_confidences = objectness_confidences[confidences > confidence_threshold]
            x_predictions = (torch_sigmoid(output[0]) + grid_x)[confidences > confidence_threshold] / feature_map_width
            y_predictions = (torch_sigmoid(output[1]) + grid_y)[confidences > confidence_threshold] / feature_map_height
            w_predictions = (torch_exp(output[2]) * anchor_w)[confidences > confidence_threshold] / feature_map_width
            h_predictions = (torch_exp(output[3]) * anchor_h)[confidences > confidence_threshold] / feature_map_height
            class_scores = class_scores[confidences > confidence_threshold].view(-1, self.class_count)
            max_class_scores = max_class_scores[confidences > confidence_threshold]
            top_classes = top_classes[confidences > confidence_threshold]

            for b_index in range(batch_size):
                for index in range(x_predictions.size(0)):
                    objectness = objectness_confidences[index].item()
                    cx = x_predictions[index].item()
                    cy = y_predictions[index].item()
                    w = w_predictions[index].item()
                    h = h_predictions[index].item()
                    max_class_score = max_class_scores[index].item()
                    top_class = top_classes[index].item()

                    box = [cx, cy, w, h, objectness, max_class_score, top_class]

                    possible_classes = (class_scores[index] * objectness > confidence_threshold).nonzero()[:, 0]
                    possible_classes = possible_classes[possible_classes != top_class]

                    for cls in possible_classes:
                        box.append(class_scores[index][cls].item())
                        box.append(cls.item())
                    all_boxes[b_index].append(box)

        detections = []
        image_width, image_height = image_as_tensor.size(3), image_as_tensor.size(2)
        for b_index in range(batch_size):
            batch_detections = []
            boxes = nms(all_boxes[b_index], nms_threshold)
            for box in boxes:
                x1 = max(box[0] - box[2] / 2.0, 0) * image_width
                y1 = max(box[1] - box[3] / 2.0, 0) * image_height
                x2 = min(box[0] + box[2] / 2.0, 1) * image_width
                y2 = min(box[1] + box[3] / 2.0, 1) * image_height
                objectness = box[4]

                for j in range((len(box) - 5) // 2):
                    cls_conf = box[5 + 2 * j]
                    cls_id = box[6 + 2 * j]
                    prob = objectness * cls_conf
                    batch_detections.append([cls_id, prob, x1, y1, x2, y2])
            detections.append(batch_detections)

        return detections
