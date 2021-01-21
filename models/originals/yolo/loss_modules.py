from math import log as math_log
from typing import Optional, Union

from torch import LongTensor, FloatTensor, zeros as torch_zeros, max as torch_max, reshape as torch_reshape, \
    ones as torch_ones, linspace, Tensor, exp as torch_exp, sigmoid as torch_sigmoid, device
from torch.nn import Module, MSELoss, CrossEntropyLoss, BCELoss

from .internal_modules import intersection_over_union


class YOLOLossSpecification:
    """Encapsulates YOLOLoss module specification.

    Previous works use ``kwargs`` for setting the required values. Since the values are numerous and subject to
    change in later sections, this class is created to encapsulate the specification of YOLOv2 and YOLOv3 loss
    function layers.

    If the required arguments are unavailable in current arguments, one can add the required arguments to ``kwargs``
    and modify ``YOLOv2Loss`` and ``YOLOv3Loss`` implementation details.
    """

    def __init__(self, version: int, num_classes: int = 20, noobject_scale: float = .5, object_scale: float = 5,
                 ignore_threshold: float = .5, iou_threshold: float = .5, anchor_step: int = 2, max_object: int = 50,
                 coordinate_loss_scale=1, class_probability_loss_scale=1, is_multilabel=True, **kwargs):
        self.version = version
        self.num_classes = num_classes
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale
        self.ignore_threshold = ignore_threshold
        self.iou_threshold = iou_threshold
        self.anchor_step = anchor_step
        self.max_object = max_object
        self.coordinate_loss_scale = coordinate_loss_scale
        self.class_probability_loss_scale = class_probability_loss_scale
        self.is_multilabel = is_multilabel
        self.kwargs = kwargs


class YOLOv2Loss(Module):
    """YOLOv2 loss function, also known as ``[region]`` block in YOLOv2 config file.

    This is implemented as shown in https://github.com/CharlesPikachu/YOLO and https://github.com/marvis/pytorch-yolo2.
    Modifications are made for variable names, refactoring to functions, logic of incrementing/setting
    ``seen_images`` value, and the 12800 seen images limit is changed to a variable for more flexible settings. All
    credits to @CharlesPikachu and @marvis.
    """

    def __init__(self, anchors, use_cuda, spec):
        super().__init__()

        self.anchor_step = spec.anchor_step
        assert len(anchors) % spec.anchor_step == 0
        self.anchors = anchors
        self.num_anchors = len(anchors) // self.anchor_step

        self.num_classes = spec.num_classes

        self.use_cuda = use_cuda

        self.noobject_scale = spec.noobject_scale
        self.object_scale = spec.object_scale

        self.ignore_threshold = spec.ignore_threshold
        self.iou_threshold = spec.iou_threshold
        self.max_object = spec.max_object

        self.coord_scale = spec.coordinate_loss_scale
        self.class_scale = spec.class_probability_loss_scale

        self.seen_images = 0

    def _to_cuda(self, obj):
        return obj.cuda() if self.use_cuda else obj

    def _convert_to_cpu(self, gpu_matrix):
        return FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

    def _build_targets(self, predictions, target_data, feature_map_width, feature_map_height):
        batch_size = target_data.size(0)
        number_of_pixels = feature_map_height * feature_map_width
        anchors_over_pixels = self.num_anchors * number_of_pixels

        default_size = (batch_size, self.num_anchors, feature_map_height, feature_map_width)

        _1obj = torch_zeros(*default_size)
        _1noobj = torch_ones(*default_size)

        target_center_x_values = torch_zeros(*default_size)
        target_center_y_values = torch_zeros(*default_size)
        target_width_values = torch_zeros(*default_size)
        target_height_values = torch_zeros(*default_size)
        target_confidence_score_values = torch_zeros(*default_size)
        target_class_values = torch_zeros(*default_size)

        for image_index in range(batch_size):
            start_index = image_index * anchors_over_pixels
            end_index = (image_index + 1) * anchors_over_pixels
            predicted_bounding_boxes = predictions[start_index:end_index].t()
            ious = torch_zeros(anchors_over_pixels)

            for t in range(self.max_object):
                if target_data[image_index][t * 5 + 1] == -1:
                    break

                ground_truth_center_x = target_data[image_index][t * 5 + 1] * feature_map_width
                ground_truth_center_y = target_data[image_index][t * 5 + 2] * feature_map_height
                ground_truth_width = target_data[image_index][t * 5 + 3] * feature_map_width
                ground_truth_height = target_data[image_index][t * 5 + 4] * feature_map_height
                ground_truth_bounding_boxes = FloatTensor(
                    [ground_truth_center_x, ground_truth_center_y, ground_truth_width, ground_truth_height])
                ground_truth_bounding_boxes = ground_truth_bounding_boxes.repeat(anchors_over_pixels, 1).t()
                ious = torch_max(ious, intersection_over_union(
                    True, predicted_bounding_boxes, ground_truth_bounding_boxes, is_corner_coordinates=False))
            # https://github.com/marvis/pytorch-yolo2/issues/121#issuecomment-436388664
            _1noobj[image_index][torch_reshape(ious, (
                self.num_anchors, feature_map_height, feature_map_width)) > self.ignore_threshold] = 0

        for image_index in range(batch_size):
            for t in range(self.max_object):
                if target_data[image_index][t * 5 + 1] == -1:
                    break

                anchor_index, ground_truth_width, ground_truth_height = self._find_most_matching_anchor(
                    feature_map_width, feature_map_height, image_index, t, target_data)

                ground_truth_center_x_pixel, ground_truth_center_y_pixel, ground_truth_bounding_box = \
                    self._compose_ground_truth_data(feature_map_width, feature_map_height, ground_truth_height,
                                                    ground_truth_width, image_index, t, target_data)

                predicted_bounding_box = predictions[
                    image_index * anchors_over_pixels + anchor_index * number_of_pixels + ground_truth_center_y_pixel *
                    feature_map_width + ground_truth_center_x_pixel]

                iou = intersection_over_union(
                    False, ground_truth_bounding_box, predicted_bounding_box, is_corner_coordinates=False)

                _1obj[image_index][anchor_index][ground_truth_center_y_pixel][ground_truth_center_x_pixel] = 1
                _1noobj[image_index][anchor_index][ground_truth_center_y_pixel][ground_truth_center_x_pixel] = 0

                target_center_x_values, target_center_y_values, target_width_values, target_height_values, \
                target_confidence_score_values, target_class_values = self._set_target_values(
                    feature_map_width, feature_map_height, image_index, t, target_data, anchor_index, iou,
                    ground_truth_center_x_pixel, ground_truth_center_y_pixel, ground_truth_height, ground_truth_width,
                    target_center_x_values, target_center_y_values, target_class_values,
                    target_confidence_score_values, target_height_values, target_width_values)

        return _1obj, _1noobj, target_center_x_values, target_center_y_values, target_width_values, \
               target_height_values, target_confidence_score_values, target_class_values

    def _set_target_values(self, feature_map_width, feature_map_height, image_index, t, target, anchor_index, iou,
                           ground_truth_center_x_pixel, ground_truth_center_y_pixel, ground_truth_height,
                           ground_truth_width, target_center_x_values, target_center_y_values, target_class_values,
                           target_confidence_score_values, target_height_values, target_width_values):
        image = target[image_index]

        target_center_x_values[image_index][anchor_index][ground_truth_center_y_pixel][
            ground_truth_center_x_pixel] = image[t * 5 + 1] * feature_map_width - ground_truth_center_x_pixel
        target_center_y_values[image_index][anchor_index][ground_truth_center_y_pixel][
            ground_truth_center_x_pixel] = image[t * 5 + 2] * feature_map_height - ground_truth_center_y_pixel
        target_width_values[image_index][anchor_index][ground_truth_center_y_pixel][
            ground_truth_center_x_pixel] = math_log(ground_truth_width / self.anchors[self.anchor_step * anchor_index])
        target_height_values[image_index][anchor_index][ground_truth_center_y_pixel][
            ground_truth_center_x_pixel] = math_log(
            ground_truth_height / self.anchors[self.anchor_step * anchor_index + 1])

        target_confidence_score_values[image_index][anchor_index][ground_truth_center_y_pixel][
            ground_truth_center_x_pixel] = iou

        target_class_values[image_index][anchor_index][ground_truth_center_y_pixel][
            ground_truth_center_x_pixel] = image[t * 5]

        return target_center_x_values, target_center_y_values, target_width_values, target_height_values, \
               target_confidence_score_values, target_class_values

    def _compose_ground_truth_data(self, feature_map_width, feature_map_height, ground_truth_height, ground_truth_width,
                                   image_index, t, target):
        center_x = target[image_index][t * 5 + 1] * feature_map_width
        center_y = target[image_index][t * 5 + 2] * feature_map_height
        pixelized_center_x = int(center_x)
        pixelized_center_y = int(center_y)

        bounding_box_specifications = [center_x, center_y, ground_truth_width, ground_truth_height]

        return pixelized_center_x, pixelized_center_y, bounding_box_specifications

    def _find_most_matching_anchor(self, feature_map_width, feature_map_height, image_index, t, target):
        best_iou = 0.0
        best_n = -1
        ground_truth_width = target[image_index][t * 5 + 3] * feature_map_width
        ground_truth_height = target[image_index][t * 5 + 4] * feature_map_height
        ground_truth_bounding_box = [0, 0, ground_truth_width, ground_truth_height]

        for i in range(self.num_anchors):
            anchor_width = self.anchors[self.anchor_step * i]
            anchor_height = self.anchors[self.anchor_step * i + 1]
            anchor_box = [0, 0, anchor_width, anchor_height]
            iou = intersection_over_union(False, anchor_box, ground_truth_bounding_box, is_corner_coordinates=False)

            if iou > best_iou:
                best_iou = iou
                best_n = i

        return best_n, ground_truth_width, ground_truth_height

    def forward(self, predictions, target, divide_by_mask=False, class_loss_reduction='mean',
                location_confidence_loss_reduction='sum'):
        batch_size = predictions.data.size(0)
        feature_map_height = predictions.data.size(2)
        feature_map_width = predictions.data.size(3)

        if self.training:
            self.seen_images += batch_size

        predicted_center_x_values, predicted_center_y_values, predicted_width_values, predicted_height_values, \
        predicted_confidence_score_values, predicted_class_probability_values = self._parse_predictions(
            predictions, batch_size, feature_map_width, feature_map_height)

        predicted_bounding_boxes = self._get_predicted_bounding_boxes(
            predicted_center_x_values, predicted_center_y_values, predicted_width_values, predicted_height_values,
            batch_size, feature_map_width, feature_map_height)

        _1obj, _1noobj, target_center_x_values, target_center_y_values, target_width_values, target_height_values, \
        target_confidence_score_values, target_class_values = self._build_targets(
            predicted_bounding_boxes, target.data, feature_map_width, feature_map_height)
        class_probabilities_1obj = (_1obj == 1)

        target_center_x_values = self._to_cuda(target_center_x_values)
        target_center_y_values = self._to_cuda(target_center_y_values)
        target_width_values = self._to_cuda(target_width_values)
        target_height_values = self._to_cuda(target_height_values)
        target_confidence_score_values = self._to_cuda(target_confidence_score_values)
        # https://github.com/marvis/pytorch-yolo2/issues/121#issuecomment-489566355
        target_class_values = self._to_cuda(target_class_values[class_probabilities_1obj == 1].view(-1).long())

        _1obj = self._to_cuda(_1obj)
        _1noobj = self._to_cuda(_1noobj)
        class_probabilities_1obj = self._to_cuda(class_probabilities_1obj.view(-1, 1).repeat(1, self.num_classes))
        # https://github.com/marvis/pytorch-yolo2/issues/121#issuecomment-489566355
        predicted_class_probability_values = predicted_class_probability_values[class_probabilities_1obj == 1].view(
            -1, self.num_classes)

        predicted = {
            'x': predicted_center_x_values,
            'y': predicted_center_y_values,
            'w': predicted_width_values,
            'h': predicted_height_values,
            'C': predicted_confidence_score_values,
            'p(c)': predicted_class_probability_values
        }
        target = {
            'x': target_center_x_values,
            'y': target_center_y_values,
            'w': target_width_values,
            'h': target_height_values,
            'C': target_confidence_score_values,
            'p(c)': target_class_values
        }

        return self._calculate_loss(
            predicted, target, _1obj, _1noobj, divide_by_mask, class_loss_reduction, location_confidence_loss_reduction)

    def _calculate_loss(self, predicted, target, _1obj, _1noobj, divide_by_mask, class_loss_reduction,
                        location_confidence_loss_reduction):
        mse_loss = MSELoss(reduction=location_confidence_loss_reduction)

        if divide_by_mask:
            total_objects = _1obj.sum()
            loss_x = self.coord_scale * (mse_loss(predicted['x'] * _1obj, target['x'] * _1obj) / total_objects) / 2.0
            loss_y = self.coord_scale * (mse_loss(predicted['y'] * _1obj, target['y'] * _1obj) / total_objects) / 2.0
            loss_w = self.coord_scale * (mse_loss(predicted['w'] * _1obj, target['w'] * _1obj) / total_objects) / 2.0
            loss_h = self.coord_scale * (mse_loss(predicted['h'] * _1obj, target['h'] * _1obj) / total_objects) / 2.0
            coordinates_loss = loss_x + loss_y + loss_w + loss_h

            object_loss = self.object_scale * (mse_loss(
                predicted['C'] * _1obj, target['C'] * _1obj) / total_objects) / 2.0
            no_object_loss = self.noobject_scale * (mse_loss(
                predicted['C'] * _1noobj, target['C'] * _1noobj) / _1noobj.sum()) / 2.0
            confidence_score_loss = object_loss + no_object_loss

            try:
                class_loss = self.class_scale * CrossEntropyLoss(reduction=class_loss_reduction)(
                    predicted['p(c)'], target['p(c)']) / total_objects
            except:
                class_loss = 0
        else:
            loss_x = self.coord_scale * mse_loss(predicted['x'] * _1obj, target['x'] * _1obj) / 2.0
            loss_y = self.coord_scale * mse_loss(predicted['y'] * _1obj, target['y'] * _1obj) / 2.0
            loss_w = self.coord_scale * mse_loss(predicted['w'] * _1obj, target['w'] * _1obj) / 2.0
            loss_h = self.coord_scale * mse_loss(predicted['h'] * _1obj, target['h'] * _1obj) / 2.0
            coordinates_loss = loss_x + loss_y + loss_w + loss_h

            object_loss = self.object_scale * mse_loss(predicted['C'] * _1obj, target['C'] * _1obj) / 2.0
            no_object_loss = self.noobject_scale * mse_loss(predicted['C'] * _1noobj, target['C'] * _1noobj) / 2.0
            confidence_score_loss = object_loss + no_object_loss

            try:
                class_loss = self.class_scale * CrossEntropyLoss(reduction=class_loss_reduction)(
                    predicted['p(c)'], target['p(c)'])
            except:
                class_loss = 0

        return coordinates_loss + confidence_score_loss + class_loss

    def _get_predicted_bounding_boxes(self, predicted_center_x_values, predicted_center_y_values,
                                      predicted_width_values, predicted_height_values, batch_size, feature_map_width,
                                      feature_map_height):
        default_shape = batch_size * self.num_anchors * feature_map_height * feature_map_width

        pred_boxes = self._to_cuda(FloatTensor(4, default_shape))

        grid_x = linspace(0, feature_map_width - 1, feature_map_width).repeat(feature_map_height, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(default_shape)
        grid_x = self._to_cuda(grid_x)

        grid_y = linspace(0, feature_map_height - 1, feature_map_height).repeat(feature_map_width, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(default_shape)
        grid_y = self._to_cuda(grid_y)

        anchor_w = self._to_cuda(Tensor(self.anchors)).view(self.num_anchors, 2).index_select(1, self._to_cuda(
            LongTensor([0])))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, feature_map_height * feature_map_width)
        anchor_w = anchor_w.view(default_shape)

        anchor_h = self._to_cuda(Tensor(self.anchors)).view(self.num_anchors, 2).index_select(1, self._to_cuda(
            LongTensor([1])))
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, feature_map_height * feature_map_width)
        anchor_h = anchor_h.view(default_shape)

        # https://github.com/marvis/pytorch-yolo2/issues/131#issuecomment-460989919
        pred_boxes[0] = torch_reshape(predicted_center_x_values.data, (1, default_shape)) + grid_x
        pred_boxes[1] = torch_reshape(predicted_center_y_values.data, (1, default_shape)) + grid_y
        pred_boxes[2] = torch_reshape(torch_exp(predicted_width_values.data), (1, default_shape)) * anchor_w
        pred_boxes[3] = torch_reshape(torch_exp(predicted_height_values.data), (1, default_shape)) * anchor_h

        return self._convert_to_cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, 4))

    def _parse_predictions(self, predictions, batch_size, feature_map_width, feature_map_height):
        output = predictions.view(batch_size, self.num_anchors, (5 + self.num_classes), feature_map_height,
                                  feature_map_width)

        default_size = (batch_size, self.num_anchors, feature_map_height, feature_map_width)

        predicted_center_x_values = output.index_select(2, self._to_cuda(LongTensor([0])))
        predicted_center_x_values = torch_sigmoid(predicted_center_x_values.view(*default_size))

        predicted_center_y_values = output.index_select(2, self._to_cuda(LongTensor([1])))
        predicted_center_y_values = torch_sigmoid(predicted_center_y_values.view(*default_size))

        predicted_width_values = output.index_select(2, self._to_cuda(LongTensor([2])))
        predicted_width_values = predicted_width_values.view(*default_size)

        predicted_height_values = output.index_select(2, self._to_cuda(LongTensor([3])))
        predicted_height_values = predicted_height_values.view(*default_size)

        predicted_confidence_score_values = output.index_select(2, self._to_cuda(LongTensor([4])))
        predicted_confidence_score_values = torch_sigmoid(predicted_confidence_score_values.view(*default_size))

        linear_space = self._to_cuda(linspace(5, 5 + self.num_classes - 1, self.num_classes).long())
        predicted_class_probability_values = output.index_select(2, linear_space)
        predicted_class_probability_values = predicted_class_probability_values.view(
            batch_size * self.num_anchors, self.num_classes, feature_map_height * feature_map_width)
        predicted_class_probability_values = predicted_class_probability_values.transpose(1, 2).contiguous()
        predicted_class_probability_values = predicted_class_probability_values.view(
            batch_size * self.num_anchors * feature_map_height * feature_map_width, self.num_classes)

        return predicted_center_x_values, predicted_center_y_values, predicted_width_values, predicted_height_values, \
               predicted_confidence_score_values, predicted_class_probability_values

    def cuda(self, dev: Optional[Union[int, device]] = None):
        cuda_device = super().cuda(dev)
        self.use_cuda = True
        return cuda_device

    def cpu(self):
        cpu_device = super().cpu()
        self.use_cuda = False
        return cpu_device


class YOLOv3Loss(Module):
    """YOLOv3 loss function, also known as ``[yolo]`` block in YOLOv3 config file.

    This is implemented as shown in https://github.com/CharlesPikachu/YOLO and https://github.com/marvis/pytorch-yolo2.
    Modifications are made for variable names, refactoring to functions, logic of incrementing/setting
    ``seen_images`` value, and the 12800 seen images limit is changed to a variable for more flexible settings. All
    credits to @CharlesPikachu and @marvis.
    """

    def __init__(self, anchors, downsample_ratio, use_cuda, spec):
        super().__init__()

        self.anchor_step = spec.anchor_step
        assert len(anchors) % spec.anchor_step == 0
        self.anchors = [a / downsample_ratio for a in anchors]
        self.num_anchors = len(anchors) // self.anchor_step

        self.num_classes = spec.num_classes

        self.use_cuda = use_cuda

        self.noobject_scale = spec.noobject_scale
        self.object_scale = spec.object_scale

        self.ignore_threshold = spec.ignore_threshold
        self.iou_threshold = spec.iou_threshold
        self.max_object = spec.max_object

        self.coord_scale = spec.coordinate_loss_scale
        self.class_scale = spec.class_probability_loss_scale

        self.seen_images = 0

        self.is_multilabel = spec.is_multilabel

    def _to_cuda(self, obj):
        return obj.cuda() if self.use_cuda else obj

    def _convert_to_cpu(self, gpu_matrix):
        return FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

    def _build_targets(self, predictions, target_data, feature_map_width, feature_map_height):
        batch_size = target_data.size(0)
        number_of_pixels = feature_map_height * feature_map_width
        anchors_over_pixels = self.num_anchors * number_of_pixels

        default_size = (batch_size, self.num_anchors, feature_map_height, feature_map_width)

        _1obj = torch_zeros(*default_size)
        _1noobj = torch_ones(*default_size)

        target_center_x_values = torch_zeros(*default_size)
        target_center_y_values = torch_zeros(*default_size)
        target_width_values = torch_zeros(*default_size)
        target_height_values = torch_zeros(*default_size)
        target_confidence_score_values = torch_zeros(*default_size)
        target_class_values = torch_zeros(*default_size)

        for image_index in range(batch_size):
            start_index = image_index * anchors_over_pixels
            end_index = (image_index + 1) * anchors_over_pixels
            predicted_bounding_boxes = predictions[start_index:end_index].t()
            ious = torch_zeros(anchors_over_pixels)

            for t in range(self.max_object):
                if target_data[image_index][t * 5 + 1] == -1:
                    break

                ground_truth_center_x = target_data[image_index][t * 5 + 1] * feature_map_width
                ground_truth_center_y = target_data[image_index][t * 5 + 2] * feature_map_height
                ground_truth_width = target_data[image_index][t * 5 + 3] * feature_map_width
                ground_truth_height = target_data[image_index][t * 5 + 4] * feature_map_height
                ground_truth_bounding_boxes = FloatTensor(
                    [ground_truth_center_x, ground_truth_center_y, ground_truth_width, ground_truth_height])
                ground_truth_bounding_boxes = ground_truth_bounding_boxes.repeat(anchors_over_pixels, 1).t()
                ious = torch_max(ious, intersection_over_union(
                    True, predicted_bounding_boxes, ground_truth_bounding_boxes, is_corner_coordinates=False))
            # https://github.com/marvis/pytorch-yolo2/issues/121#issuecomment-436388664
            _1noobj[image_index][torch_reshape(ious, (
                self.num_anchors, feature_map_height, feature_map_width)) > self.ignore_threshold] = 0

        for image_index in range(batch_size):
            for t in range(self.max_object):
                if target_data[image_index][t * 5 + 1] == -1:
                    break

                anchor_index, ground_truth_width, ground_truth_height = self._find_most_matching_anchor(
                    feature_map_width, feature_map_height, image_index, t, target_data)

                ground_truth_center_x_pixel, ground_truth_center_y_pixel, ground_truth_bounding_box = \
                    self._compose_ground_truth_data(feature_map_width, feature_map_height, ground_truth_height,
                                                    ground_truth_width, image_index, t, target_data)

                predicted_bounding_box = predictions[
                    image_index * anchors_over_pixels + anchor_index * number_of_pixels + ground_truth_center_y_pixel *
                    feature_map_width + ground_truth_center_x_pixel]

                iou = intersection_over_union(
                    False, ground_truth_bounding_box, predicted_bounding_box, is_corner_coordinates=False)

                _1obj[image_index][anchor_index][ground_truth_center_y_pixel][ground_truth_center_x_pixel] = 1
                _1noobj[image_index][anchor_index][ground_truth_center_y_pixel][ground_truth_center_x_pixel] = 0

                target_center_x_values, target_center_y_values, target_width_values, target_height_values, \
                target_confidence_score_values, target_class_values = self._set_target_values(
                    feature_map_width, feature_map_height, image_index, t, target_data, anchor_index, iou,
                    ground_truth_center_x_pixel, ground_truth_center_y_pixel, ground_truth_height, ground_truth_width,
                    target_center_x_values, target_center_y_values, target_class_values, target_confidence_score_values,
                    target_height_values, target_width_values)

        return _1obj, _1noobj, target_center_x_values, target_center_y_values, target_width_values, \
               target_height_values, target_confidence_score_values, target_class_values

    def _set_target_values(self, feature_map_width, feature_map_height, image_index, t, target, anchor_index, iou,
                           ground_truth_center_x_pixel, ground_truth_center_y_pixel, ground_truth_height,
                           ground_truth_width, target_center_x_values, target_center_y_values, target_class_values,
                           target_confidence_score_values, target_height_values, target_width_values):
        image = target[image_index]

        target_center_x_values[image_index][anchor_index][ground_truth_center_y_pixel][
            ground_truth_center_x_pixel] = image[t * 5 + 1] * feature_map_width - ground_truth_center_x_pixel
        target_center_y_values[image_index][anchor_index][ground_truth_center_y_pixel][
            ground_truth_center_x_pixel] = image[t * 5 + 2] * feature_map_height - ground_truth_center_y_pixel
        target_width_values[image_index][anchor_index][ground_truth_center_y_pixel][
            ground_truth_center_x_pixel] = math_log(ground_truth_width / self.anchors[self.anchor_step * anchor_index])
        target_height_values[image_index][anchor_index][ground_truth_center_y_pixel][
            ground_truth_center_x_pixel] = math_log(
            ground_truth_height / self.anchors[self.anchor_step * anchor_index + 1])

        target_confidence_score_values[image_index][anchor_index][ground_truth_center_y_pixel][
            ground_truth_center_x_pixel] = iou

        target_class_values[image_index][anchor_index][ground_truth_center_y_pixel][
            ground_truth_center_x_pixel] = image[t * 5]

        return target_center_x_values, target_center_y_values, target_width_values, target_height_values, \
               target_confidence_score_values, target_class_values

    def _compose_ground_truth_data(self, feature_map_width, feature_map_height, ground_truth_height, ground_truth_width,
                                   image_index, t, target):
        center_x = target[image_index][t * 5 + 1] * feature_map_width
        center_y = target[image_index][t * 5 + 2] * feature_map_height
        pixelized_center_x = int(center_x)
        pixelized_center_y = int(center_y)

        bounding_box_specifications = [center_x, center_y, ground_truth_width, ground_truth_height]

        return pixelized_center_x, pixelized_center_y, bounding_box_specifications

    def _find_most_matching_anchor(self, feature_map_width, feature_map_height, image_index, t, target):
        best_iou = 0.0
        best_n = -1
        ground_truth_width = target[image_index][t * 5 + 3] * feature_map_width
        ground_truth_height = target[image_index][t * 5 + 4] * feature_map_height
        ground_truth_bounding_box = [0, 0, ground_truth_width, ground_truth_height]

        for i in range(self.num_anchors):
            anchor_width = self.anchors[self.anchor_step * i]
            anchor_height = self.anchors[self.anchor_step * i + 1]
            anchor_box = [0, 0, anchor_width, anchor_height]
            iou = intersection_over_union(False, anchor_box, ground_truth_bounding_box, is_corner_coordinates=False)

            if iou > best_iou:
                best_iou = iou
                best_n = i

        return best_n, ground_truth_width, ground_truth_height

    def forward(self, predictions, target, divide_by_mask=False, class_loss_reduction='mean',
                location_confidence_loss_reduction='sum'):
        batch_size = predictions.data.size(0)
        feature_map_height = predictions.data.size(2)
        feature_map_width = predictions.data.size(3)

        if self.training:
            self.seen_images += batch_size

        predicted_center_x_values, predicted_center_y_values, predicted_width_values, predicted_height_values, \
        predicted_confidence_score_values, predicted_class_probability_values = self._parse_predictions(
            predictions, batch_size, feature_map_width, feature_map_height)

        predicted_bounding_boxes = self._get_predicted_bounding_boxes(
            predicted_center_x_values, predicted_center_y_values, predicted_width_values, predicted_height_values,
            batch_size, feature_map_width, feature_map_height)

        _1obj, _1noobj, target_center_x_values, target_center_y_values, target_width_values, target_height_values, \
        target_confidence_score_values, target_class_values = self._build_targets(
            predicted_bounding_boxes, target.data, feature_map_width, feature_map_height)
        class_probabilities_1obj = (_1obj == 1)

        target_center_x_values = self._to_cuda(target_center_x_values)
        target_center_y_values = self._to_cuda(target_center_y_values)
        target_width_values = self._to_cuda(target_width_values)
        target_height_values = self._to_cuda(target_height_values)
        target_confidence_score_values = self._to_cuda(target_confidence_score_values)
        # https://github.com/marvis/pytorch-yolo2/issues/121#issuecomment-489566355
        target_class_values = target_class_values[class_probabilities_1obj == 1].view(-1)
        target_class_values = target_class_values if self.is_multilabel else target_class_values.long()
        target_class_values = torch_sigmoid(target_class_values) if self.is_multilabel else target_class_values
        target_class_values = self._to_cuda(target_class_values)

        _1obj = self._to_cuda(_1obj)
        _1noobj = self._to_cuda(_1noobj)
        class_probabilities_1obj = self._to_cuda(class_probabilities_1obj.view(-1, 1).repeat(1, self.num_classes))
        # https://github.com/marvis/pytorch-yolo2/issues/121#issuecomment-489566355
        predicted_class_probability_values = predicted_class_probability_values[class_probabilities_1obj == 1].view(
            -1, self.num_classes)

        predicted = {
            'x': predicted_center_x_values,
            'y': predicted_center_y_values,
            'w': predicted_width_values,
            'h': predicted_height_values,
            'C': predicted_confidence_score_values,
            'p(c)': predicted_class_probability_values
        }
        target = {
            'x': target_center_x_values,
            'y': target_center_y_values,
            'w': target_width_values,
            'h': target_height_values,
            'C': target_confidence_score_values,
            'p(c)': target_class_values
        }

        return self._calculate_loss(
            predicted, target, _1obj, _1noobj, divide_by_mask, class_loss_reduction, location_confidence_loss_reduction)

    def _calculate_loss(self, predicted, target, _1obj, _1noobj, divide_by_mask, class_loss_reduction,
                        location_confidence_loss_reduction):
        mse_loss = MSELoss(reduction=location_confidence_loss_reduction)

        if divide_by_mask:
            total_objects = _1obj.sum()
            loss_x = self.coord_scale * (mse_loss(predicted['x'] * _1obj, target['x'] * _1obj) / total_objects) / 2.0
            loss_y = self.coord_scale * (mse_loss(predicted['y'] * _1obj, target['y'] * _1obj) / total_objects) / 2.0
            loss_w = self.coord_scale * (mse_loss(predicted['w'] * _1obj, target['w'] * _1obj) / total_objects) / 2.0
            loss_h = self.coord_scale * (mse_loss(predicted['h'] * _1obj, target['h'] * _1obj) / total_objects) / 2.0
            coordinates_loss = loss_x + loss_y + loss_w + loss_h

            object_loss = self.object_scale * (
                    mse_loss(predicted['C'] * _1obj, target['C'] * _1obj) / total_objects) / 2.0
            no_object_loss = self.noobject_scale * (mse_loss(
                predicted['C'] * _1noobj, target['C'] * _1noobj) / _1noobj.sum()) / 2.0
            confidence_score_loss = object_loss + no_object_loss

            try:
                if self.is_multilabel:
                    class_loss = BCELoss(reduction=class_loss_reduction)(predicted['p(c'], self._to_cuda(torch_zeros(
                        predicted['p(c'].shape)).index_fill_(1, target['p(c'].data.cpu().long(), 1.0))
                else:
                    class_loss = CrossEntropyLoss(reduction=class_loss_reduction)(predicted['p(c)'], target['p(c)'])
                class_loss = self.class_scale * class_loss / total_objects
            except:
                class_loss = 0
        else:
            loss_x = self.coord_scale * mse_loss(predicted['x'] * _1obj, target['x'] * _1obj) / 2.0
            loss_y = self.coord_scale * mse_loss(predicted['y'] * _1obj, target['y'] * _1obj) / 2.0
            loss_w = self.coord_scale * mse_loss(predicted['w'] * _1obj, target['w'] * _1obj) / 2.0
            loss_h = self.coord_scale * mse_loss(predicted['h'] * _1obj, target['h'] * _1obj) / 2.0
            coordinates_loss = loss_x + loss_y + loss_w + loss_h

            object_loss = self.object_scale * mse_loss(predicted['C'] * _1obj, target['C'] * _1obj) / 2.0
            no_object_loss = self.noobject_scale * mse_loss(predicted['C'] * _1noobj, target['C'] * _1noobj) / 2.0
            confidence_score_loss = object_loss + no_object_loss

            try:
                if self.is_multilabel:
                    class_loss = BCELoss(reduction=class_loss_reduction)(predicted['p(c'], self._to_cuda(torch_zeros(
                        predicted['p(c'].shape)).index_fill_(1, target['p(c'].data.cpu().long(), 1.0))
                else:
                    class_loss = CrossEntropyLoss(reduction=class_loss_reduction)(predicted['p(c)'], target['p(c)'])
                class_loss = self.class_scale * class_loss
            except:
                class_loss = 0

        # Divided by 3 (number of predictors across scales - Large, Medium, Small) to give equivalent weight for each
        # predictor.
        return (coordinates_loss + confidence_score_loss + class_loss) / 3

    def _get_predicted_bounding_boxes(self, predicted_center_x_values, predicted_center_y_values,
                                      predicted_width_values, predicted_height_values, batch_size, feature_map_width,
                                      feature_map_height):
        default_shape = batch_size * self.num_anchors * feature_map_height * feature_map_width

        pred_boxes = self._to_cuda(FloatTensor(4, default_shape))

        grid_x = linspace(0, feature_map_width - 1, feature_map_width).repeat(feature_map_height, 1).repeat(
            batch_size * self.num_anchors, 1, 1)
        grid_x = self._to_cuda(grid_x.view(default_shape))

        grid_y = linspace(0, feature_map_height - 1, feature_map_height).repeat(feature_map_width, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(default_shape)
        grid_y = self._to_cuda(grid_y)

        anchor_w = self._to_cuda(Tensor(self.anchors)).view(self.num_anchors, 2).index_select(1, self._to_cuda(
            LongTensor([0])))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, feature_map_height * feature_map_width)
        anchor_w = anchor_w.view(default_shape)

        anchor_h = self._to_cuda(Tensor(self.anchors)).view(self.num_anchors, 2).index_select(1, self._to_cuda(
            LongTensor([1])))
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, feature_map_height * feature_map_width)
        anchor_h = anchor_h.view(default_shape)

        # https://github.com/marvis/pytorch-yolo2/issues/131#issuecomment-460989919
        pred_boxes[0] = torch_reshape(predicted_center_x_values.data, (1, default_shape)) + grid_x
        pred_boxes[1] = torch_reshape(predicted_center_y_values.data, (1, default_shape)) + grid_y
        pred_boxes[2] = torch_reshape(torch_exp(predicted_width_values.data), (1, default_shape)) * anchor_w
        pred_boxes[3] = torch_reshape(torch_exp(predicted_height_values.data), (1, default_shape)) * anchor_h

        return self._convert_to_cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, 4))

    def _parse_predictions(self, predictions, batch_size, feature_map_width, feature_map_height):
        output = predictions.view(
            batch_size, self.num_anchors, (5 + self.num_classes), feature_map_height, feature_map_width)

        default_size = (batch_size, self.num_anchors, feature_map_height, feature_map_width)

        predicted_center_x_values = output.index_select(2, self._to_cuda(LongTensor([0])))
        predicted_center_x_values = torch_sigmoid(predicted_center_x_values.view(*default_size))

        predicted_center_y_values = output.index_select(2, self._to_cuda(LongTensor([1])))
        predicted_center_y_values = torch_sigmoid(predicted_center_y_values.view(*default_size))

        predicted_width_values = output.index_select(2, self._to_cuda(LongTensor([2])))
        predicted_width_values = predicted_width_values.view(*default_size)

        predicted_height_values = output.index_select(2, self._to_cuda(LongTensor([3])))
        predicted_height_values = predicted_height_values.view(*default_size)

        predicted_confidence_score_values = output.index_select(2, self._to_cuda(LongTensor([4])))
        predicted_confidence_score_values = torch_sigmoid(predicted_confidence_score_values.view(*default_size))

        linear_space = self._to_cuda(linspace(5, 5 + self.num_classes - 1, self.num_classes).long())
        predicted_class_probability_values = output.index_select(2, linear_space)
        predicted_class_probability_values = predicted_class_probability_values.view(
            batch_size * self.num_anchors, self.num_classes, feature_map_height * feature_map_width)
        predicted_class_probability_values = predicted_class_probability_values.transpose(1, 2).contiguous()
        predicted_class_probability_values = predicted_class_probability_values.view(
            batch_size * self.num_anchors * feature_map_height * feature_map_width, self.num_classes)

        return predicted_center_x_values, predicted_center_y_values, predicted_width_values, predicted_height_values, \
               predicted_confidence_score_values, predicted_class_probability_values

    def cuda(self, dev: Optional[Union[int, device]] = None):
        cuda_device = super().cuda(dev)
        self.use_cuda = True
        return cuda_device

    def cpu(self):
        cpu_device = super().cpu()
        self.use_cuda = False
        return cpu_device


class YOLOLoss(Module):
    """A module for easier YOLO loss function settings and selection.

    This module can be appended to extensions of ``Module`` as the loss function. Common usage in a ``Module``
    extension would be::

        def __init__(self, version, anchor_boxes, spec):
            self.loss = YOLOLoss(version=version, anchor_boxes=anchor_boxes, use_cuda=next(self.parameters()).is_cuda,
             spec=spec)

    Arguments:
        version: YOLO loss function version to use. Currently supports only YOLOv2 and YOLOv3 loss functions.
        anchor_boxes: Both YOLOv2 and YOLOv3 use anchor boxes in detection. Set the values of anchor boxes in a list
            without grouping each anchor box's values. E.g., for 1 anchor box, set anchor_boxes value to [x,
            y]. If not specified, the default anchor boxes for each corresponding YOLO version will be set.
        use_cuda: Several computations require transfer of tensor to CUDA device, and if this isn't handled by
            use_cuda, the layer will raise error.
        spec: Specification of YOLO loss function. See more detail at YOLOLossSpecification class.
        v3type: Loss function type for YOLOv3 network. Useful for determining which anchor boxes configuration to
            return as default anchor boxes, should the anchor_boxes parameter is not specified. Must be of either
            YOLOLoss.V3LARGE, YOLOLoss.V3MEDIUM, or YOLOLoss.V3SMALL.
    """
    V3LARGE = 'Large'
    V3MEDIUM = 'Medium'
    V3SMALL = 'Small'

    def __init__(self, version: int, anchor_boxes: list, use_cuda: bool, spec: YOLOLossSpecification, v3type=V3LARGE,
                 downsample_ratio: int = None):
        super().__init__()

        assert version is not None
        assert use_cuda is not None
        assert spec is not None

        self.version = version
        self.anchor_boxes = anchor_boxes if anchor_boxes else self.__get_default_anchor_boxes(v3type)

        if version == 2:
            self.layer = YOLOv2Loss(self.anchor_boxes, use_cuda, spec)
        elif version == 3:
            assert downsample_ratio is not None
            self.layer = YOLOv3Loss(self.anchor_boxes, downsample_ratio, use_cuda, spec)

    def __get_default_anchor_boxes(self, v3type):
        if self.version == 2:
            # Anchor boxes from yolov2-voc.cfg
            return [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
        elif self.version == 3:
            if v3type == self.V3LARGE:
                return [10, 13, 16, 30, 33, 23]
            elif v3type == self.V3MEDIUM:
                return [30, 61, 62, 45, 59, 119]
            elif v3type == self.V3SMALL:
                return [116, 90, 156, 198, 373, 326]

    def set_anchors(self, anchors, downsample_ratio=None):
        assert len(anchors) % self.layer.anchor_step == 0
        self.layer.anchors = anchors if downsample_ratio is None else [a / downsample_ratio for a in anchors]
        self.layer.num_anchors = len(anchors) // self.layer.anchor_step

    def set_cuda(self, use_cuda):
        self.layer.use_cuda = use_cuda

    def set_seen_images(self, seen_images):
        self.layer.seen_images = seen_images

    def cuda(self, dev: Optional[Union[int, device]] = None):
        cuda_device = super().cuda(dev)
        self.layer.cuda()
        return cuda_device

    def cpu(self):
        cpu_device = super().cpu()
        self.layer.cpu()
        return cpu_device

    def forward(self, predictions: Tensor, target: Tensor, **kwargs):
        return self.layer(predictions, target, **kwargs)
