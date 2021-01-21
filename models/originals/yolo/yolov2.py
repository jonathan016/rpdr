from typing import Optional, Union

from torch import Tensor, LongTensor, device, exp as torch_exp, sigmoid as torch_sigmoid, linspace as torch_linspace, \
    max as torch_max
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, LeakyReLU, MaxPool2d, CrossEntropyLoss, Softmax

from .external_modules import Identity, GlobalAvgPool2d, nms
from .internal_modules import DetectionAdditionalLayers
from .loss_modules import YOLOLossSpecification, YOLOLoss


class YOLOv2(Module):
    """Implementation of YOLOv2 model.

    Initializing this model would require information regarding available classes in the dataset and how many bounding
    boxes to generate per grid cell as specified in YOLOv2 publication since this model is a direct implementation of
    YOLOv2.

    Two types are available: recognizing and detecting (include recognizing). To set the model to recognizing task,
    simply call method ``model.recognizing()``, and to set the model to detection task call ``model.detecting()``.
    These invocations will respectively select the model's structure, so no further modifications are required during
    training or inference. However, note that this class is designed to be used for recognition first then detection,
    hence changing from detection to recognition after successful training for recognition then detection would
    require users to ``load_dict`` the model's weight again since the detection layers would be removed.

    The flow of data can be split into two groups, where ``S`` denotes Section, ``P`` denotes Pool, ``pred`` denotes
    predictor, ``lX`` denotes layer_X, ``pass`` denotes passthrough, and ``AX`` denotes activation from layer X:
        - Recognition::

                S1-P1-S2-P2-S3-P3-S4-P4-S5-P5-S6-pred
        - Detection::

                S1-P1-S2-P2-S3-P3-S4-P4-S5-P5-S6-l19-l20-pass-l21-pred
                                         |_____(A13)_____|
    where A13 is the passthroughed input to passthrough layer, as specified by YOLOv2 config file and its publication.

    To calculate loss, simply call ``model.loss(target).backward()``, since the prediction values are kept after
    forward pass. Recognizing state means the model will use ``CrossEntropyLoss`` as specified in PJReddie's
    implementation (https://github.com/pjreddie/darknet) on ``softmax_layer.c``'s ``forward_softmax_layer`` function, in
    ``softmax_x_ent_cpu`` method call from ``blas.c``, where ``error[i] = (t) ? -log(p) : 0;``. In detecting state,
    the model will use ``YOLOLoss`` layer - in this case, the ``version=2`` ``YOLOLoss`` layer.

    For using this model in CUDA devices, if possible please **only use the** ``.cuda()`` **method** as it automatically
    handles ``YOLOLoss`` transfer to CUDA device. If it is not possible, after calling ``.to()`` method, invoke the
    following: ::

            >>> model.to(cuda_device)   # Primary invocation
            >>> # Following invocations
            >>> model.use_cuda = True
            >>> if model.is_recognizing() is False:
            >>>     model.loss_function.set_cuda(True)

    Arguments:
        class_count: Specifies the number of classes to be detected and recognized by the model.
        anchor_boxes: Specifies the anchor boxes to be used in prediction, where each entry is a single floating
            number. If not specified, the anchor boxes are set to YOLOv2's anchor boxes.
        bounding_boxes_per_cell: Also known as B in YOLO publications, this argument specifies how many bounding
            boxes will be generated per grid cell.
        spec: The specification of loss calculation as specified in ``YOLOLoss`` layer. This is later forwarded to
            the ``YOLOLoss``'s layer, which is selected by the passed ``version`` parameter to ``YOLOLoss`` object.
    """

    def __init__(self, class_count: int, anchor_boxes: list = None, bounding_boxes_per_cell: int = 5,
                 spec: YOLOLossSpecification = None):
        super().__init__()

        self._is_recognition = True
        self.class_count = class_count
        self.bounding_boxes_per_cell = bounding_boxes_per_cell

        self.backbone = self._create_backbone()
        self.predictor = self._select_predictor()

        self.loss_function = CrossEntropyLoss(reduction='sum')
        self.anchor_boxes = anchor_boxes
        self.spec = spec

        self.use_cuda = False

        self.predictions = None

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
            # No softmax layer is added since CrossEntropyLoss, the used classification loss function, encapsulates
            # softmax layer in its loss calculation. See https://discuss.pytorch.org/t/vgg-output-layer-no-softmax/9273
            # for more detail. To get the output with highest value, simply call ``torch.max(output)``, or to get the
            # index of the output with highest value, call ``output.argmax()``.
        else:
            out_channels = self.bounding_boxes_per_cell * (self.class_count + 5)
            predictor.add_module(f'ConvPredictor',
                                 Conv2d(in_channels=1024, out_channels=out_channels, kernel_size=1, padding=0))

        return predictor

    def recognizing(self, is_recognition=True):
        self._is_recognition = is_recognition

        if 'DetectionAdditionalLayers' in self.backbone.__dict__['_modules'].keys():
            self.backbone.__delattr__('DetectionAdditionalLayers')
        elif not is_recognition:
            self.backbone.add_module('DetectionAdditionalLayers',
                                     DetectionAdditionalLayers(self.backbone.Section5.Conv13.out_channels))

        if is_recognition:
            self.loss_function = CrossEntropyLoss(reduction='sum')
        else:
            self.loss_function = YOLOLoss(
                version=2, anchor_boxes=self.anchor_boxes, use_cuda=self.use_cuda, spec=self.spec)

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
        out = self.backbone(x)
        out = self.predictor(out)
        self.predictions = out
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
        self.predictions = out
        return out

    def is_recognizing(self):
        return self._is_recognition

    def loss(self, target, divide_by_mask=False, class_loss_reduction='mean', location_confidence_loss_reduction='sum'):
        if self._is_recognition:
            return self.loss_function(self.predictions, target)

        kwargs = {
            'divide_by_mask': divide_by_mask,
            'class_loss_reduction': class_loss_reduction,
            'location_confidence_loss_reduction': location_confidence_loss_reduction
        }
        return self.loss_function(self.predictions, target, **kwargs)

    def eval(self):
        ev = super().eval()
        self.loss_function.eval()
        return ev

    def train(self, mode: bool = True):
        tr = super().train(mode)
        self.loss_function.train(mode)
        return tr

    def cuda(self, dev: Optional[Union[int, device]] = None):
        cuda_device = super().cuda(dev)
        self.use_cuda = True
        if not self._is_recognition:
            self.loss_function.cuda()
        self.backbone.cuda()
        self.predictor.cuda()
        return cuda_device

    def cpu(self):
        cpu_device = super().cpu()
        self.use_cuda = False
        if not self._is_recognition:
            self.loss_function.cpu()
        self.backbone.cpu()
        self.predictor.cpu()
        return cpu_device

    def detect_objects(self, image_as_tensor, confidence_threshold, nms_threshold):
        self.forward(image_as_tensor)

        yolov2_loss = self.loss_function.layer
        num_anchors = yolov2_loss.num_anchors

        if self.predictions.dim() == 3:
            self.predictions = self.predictions.unsqueeze(0)

        assert self.predictions.size(1) == (5 + self.class_count) * num_anchors

        batch_size = self.predictions.size(0)
        feature_map_height = self.predictions.size(2)
        feature_map_width = self.predictions.size(3)
        number_of_pixels = feature_map_height * feature_map_width
        view_size = batch_size * num_anchors * number_of_pixels

        output = self.predictions.view(batch_size * num_anchors, 5 + self.class_count, number_of_pixels).transpose(
            0, 1).contiguous().view(5 + self.class_count, view_size)
        output = output.cuda() if self.use_cuda else output

        grid_x = torch_linspace(0, feature_map_width - 1, feature_map_width).repeat(feature_map_height, 1).repeat(
            batch_size * num_anchors, 1, 1).view(view_size)
        grid_x = grid_x.cuda() if self.use_cuda else grid_x

        grid_y = torch_linspace(0, feature_map_height - 1, feature_map_height).repeat(
            feature_map_width, 1).t().repeat(batch_size * num_anchors, 1, 1).view(view_size)
        grid_y = grid_y.cuda() if self.use_cuda else grid_y

        anchor_w = Tensor(yolov2_loss.anchors).view(num_anchors, yolov2_loss.anchor_step).index_select(
            1, LongTensor([0])).repeat(batch_size, 1).repeat(1, 1, number_of_pixels).view(view_size)
        anchor_w = anchor_w.cuda() if self.use_cuda else anchor_w

        anchor_h = Tensor(yolov2_loss.anchors).view(num_anchors, yolov2_loss.anchor_step).index_select(
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

        all_boxes = []
        for b_index in range(batch_size):
            boxes = []
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
                boxes.append(box)
            all_boxes.append(boxes)

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
