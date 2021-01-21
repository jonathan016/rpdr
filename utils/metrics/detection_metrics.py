import numpy as np
from shapely.geometry import box as shapely_box
from torch import Tensor, LongTensor, cat as torch_cat, float as torch_float, uint8 as torch_uint8, \
    sort as torch_sort, zeros as torch_zeros, max as torch_max, cumsum as torch_cumsum, sum as torch_sum, \
    tensor as torch_tensor

from models.originals.ssd.external_modules import find_jaccard_overlap

_PRETTY_PRINT_DISCRETE = """
Discrete Metrics:
=================
Mean Average Precision (mAP): {0}
------------
Per classes:
------------
Recalls:\n{1}
Precisions:\n{2}
Average Precisions:\n{3}"""

_PRETTY_PRINT_CONTINUOUS = """
Continuous Metrics:
===================
Overall Recall:\t{0}
Overall Precision:\t{1}"""


class ValueContainer:
    SSD = 'ssd'
    YOLO = 'yolo'

    def __init__(self, type):
        assert type == self.SSD or type == self.YOLO
        if type == self.SSD:
            self.boxes = []
            self.labels = []
            self.scores = []
        else:
            self.values = []

    def __getitem__(self, attribute):
        assert attribute in ['boxes', 'labels', 'scores', 'values']
        return self.__getattribute__(attribute)


class DetectionMetric:
    SSD = 'ssd'
    YOLO = 'yolo'

    def __init__(self, type, num_classes):
        assert type == self.SSD or type == self.YOLO
        self.type = type
        self.num_classes = num_classes

        self.average_precisions = None
        self.mean_average_precision = None
        self.recalls = None
        self.precisions = None
        self.overall_recall = None
        self.overall_precision = None
        self.continuous_recalls = None
        self.continuous_precisions = None

    @property
    def discrete_metrics(self):
        pretty_print = _PRETTY_PRINT_DISCRETE.format(
            self.mean_average_precision, self.recalls, self.precisions, self.average_precisions)

        return pretty_print, self.mean_average_precision, self.recalls, self.precisions, self.average_precisions

    @property
    def continuous_metrics(self):
        pretty_print = _PRETTY_PRINT_CONTINUOUS.format(self.overall_recall, self.overall_precision)

        return pretty_print, self.overall_recall, self.overall_precision, self.continuous_recalls, \
               self.continuous_precisions

    def calculate_metrics(self, predictions: ValueContainer, targets: ValueContainer, iou_threshold=0.5,
                          ssd_is_cuda=False):
        if self.type == self.SSD:
            self.average_precisions, self.recalls, self.precisions = self._ssd_discrete_metrics(
                predictions, targets, iou_threshold=iou_threshold, is_cuda=ssd_is_cuda)
            self.mean_average_precision = sum(self.average_precisions) / len(self.average_precisions)
            self.overall_recall, self.overall_precision, self.continuous_recalls, self.continuous_precisions = \
                self._ssd_continuous_metrics(predictions, targets, is_cuda=ssd_is_cuda)
        else:
            self.average_precisions, self.recalls, self.precisions = self._yolo_discrete_metrics(
                predictions, targets, iou_threshold)
            self.mean_average_precision = sum(self.average_precisions) / len(self.average_precisions)
            self.overall_recall, self.overall_precision, self.continuous_recalls, self.continuous_precisions = \
                self._yolo_continuous_metrics(predictions, targets)

    def get_average_precisions(self, predictions: ValueContainer, targets: ValueContainer, iou_threshold=0.5,
                               ssd_is_cuda=False):
        if self.type == self.SSD:
            average_precisions = self._ssd_discrete_metrics(predictions, targets, iou_threshold, ssd_is_cuda)[0]
        else:
            average_precisions = self._yolo_discrete_metrics(predictions, targets, iou_threshold)[0]

        return average_precisions

    def get_mean_average_precision(self, predictions: ValueContainer, targets: ValueContainer, iou_threshold=0.5,
                                   ssd_is_cuda=False):
        average_precisions = self.get_average_precisions(predictions, targets, iou_threshold, ssd_is_cuda)

        if self.type == self.SSD:
            mean_average_precision = sum(average_precisions) / len(average_precisions)
        else:
            mean_average_precision = sum(average_precisions) / len(average_precisions)

        return mean_average_precision

    def _ssd_discrete_metrics(self, predictions, targets, iou_threshold=0.5, is_cuda=False):
        def __to_cuda(obj):
            if is_cuda:
                obj = obj.cuda()
            return obj

        predicted_boxes = predictions['boxes']
        predicted_labels = predictions['labels']
        predicted_class_scores = predictions['scores']

        target_boxes = targets['boxes']
        target_labels = targets['labels']

        assert len(predicted_boxes) == len(predicted_labels) == len(predicted_class_scores) == len(
            target_boxes) == len(target_labels)

        target_images = list()
        for i in range(len(target_labels)):
            target_images.extend([i] * target_labels[i].size(0))
        target_images = __to_cuda(LongTensor(target_images))
        target_boxes = torch_cat(target_boxes, dim=0)
        target_labels = torch_cat(target_labels, dim=0)

        assert target_images.size(0) == target_boxes.size(0) == target_labels.size(0)

        predicted_images = list()
        for i in range(len(predicted_labels)):
            predicted_images.extend([i] * predicted_labels[i].size(0))
        predicted_images = __to_cuda(LongTensor(predicted_images))
        predicted_boxes = torch_cat(predicted_boxes, dim=0)
        predicted_labels = torch_cat(predicted_labels, dim=0)
        predicted_class_scores = torch_cat(predicted_class_scores, dim=0)

        assert predicted_images.size(0) == predicted_boxes.size(0) == predicted_labels.size(
            0) == predicted_class_scores.size(0)

        average_precisions = torch_zeros(self.num_classes, dtype=torch_float)
        recalls = torch_zeros(self.num_classes, dtype=torch_float)
        precisions = torch_zeros(self.num_classes, dtype=torch_float)
        for c in range(self.num_classes):
            target_class_images = target_images[target_labels == c]
            target_class_boxes = target_boxes[target_labels == c]

            total_objects = target_class_boxes.size(0)

            target_class_boxes_detected = __to_cuda(torch_zeros(total_objects, dtype=torch_uint8))

            class_c_predicted_images = predicted_images[predicted_labels == c]
            class_c_predicted_boxes = predicted_boxes[predicted_labels == c]
            class_c_predicted_class_scores = predicted_class_scores[predicted_labels == c]
            class_c_num_detections = class_c_predicted_boxes.size(0)
            if class_c_num_detections == 0:
                continue

            class_c_predicted_class_scores, sort_ind = torch_sort(class_c_predicted_class_scores, dim=0,
                                                                  descending=True)
            class_c_predicted_images = class_c_predicted_images[sort_ind]
            class_c_predicted_boxes = class_c_predicted_boxes[sort_ind]

            true_positives = __to_cuda(torch_zeros(class_c_num_detections, dtype=torch_float))
            false_positives = __to_cuda(torch_zeros(class_c_num_detections, dtype=torch_float))
            for d in range(class_c_num_detections):
                this_detection_box = class_c_predicted_boxes[d].unsqueeze(0)
                this_image = class_c_predicted_images[d]

                object_boxes = target_class_boxes[target_class_images == this_image]
                if object_boxes.size(0) == 0:
                    false_positives[d] = 1
                    continue

                overlaps = find_jaccard_overlap(this_detection_box, object_boxes)
                max_overlap, ind = torch_max(overlaps.squeeze(0), dim=0)

                original_ind = LongTensor(range(target_class_boxes.size(0)))[target_class_images == this_image][ind]

                if max_overlap.item() > iou_threshold:
                    if target_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        target_class_boxes_detected[original_ind] = 1
                    else:
                        false_positives[d] = 1
                else:
                    false_positives[d] = 1

            cumul_true_positives = torch_cumsum(true_positives, dim=0)
            cumul_false_positives = torch_cumsum(false_positives, dim=0)
            cumul_precision = cumul_true_positives / (cumul_true_positives + cumul_false_positives + 1e-10)
            cumul_recall = cumul_true_positives / total_objects

            recall_thresholds = [x / 10 for x in range(11)]
            interpolated_precisions = __to_cuda(torch_zeros((len(recall_thresholds)), dtype=torch_float))
            for i, threshold in enumerate(recall_thresholds):
                recalls_above_threshold = cumul_recall >= threshold
                if recalls_above_threshold.any():
                    interpolated_precisions[i] = cumul_precision[recalls_above_threshold].max()
                else:
                    interpolated_precisions[i] = 0.
            average_precisions[c] = interpolated_precisions.mean()

            total_true_positives = torch_sum(true_positives)
            recalls[c] = total_true_positives / max(float(total_objects), 1e-10)
            precisions[c] = total_true_positives / max(
                total_true_positives + torch_sum(false_positives), torch_tensor(1e-10))
        return average_precisions.tolist(), recalls.tolist(), precisions.tolist()

    def _ssd_continuous_metrics(self, predictions, targets, is_cuda=False):
        def __to_cuda(obj):
            if is_cuda:
                obj = obj.cuda()
            return obj

        predicted_boxes = predictions['boxes']
        target_boxes = targets['boxes']

        assert len(predicted_boxes) == len(target_boxes)
        total_images = len(target_boxes)

        image_ground_truths = LongTensor([target.size(0) for target in target_boxes])
        image_predictions = LongTensor([prediction.size(0) for prediction in predicted_boxes])
        continuous_recalls = torch_zeros(total_images, dtype=torch_float)
        continuous_precisions = torch_zeros(total_images, dtype=torch_float)
        image_dimensions = __to_cuda(LongTensor([300, 300, 300, 300]))
        for image_index in range(total_images):
            if len(target_boxes[image_index]) == 0:
                continue
            image_predicted_boxes = (predicted_boxes[image_index] * image_dimensions).tolist()
            image_target_boxes = (target_boxes[image_index] * image_dimensions).tolist()

            image_predicted_boxes = [shapely_box(*box) for box in image_predicted_boxes]
            image_target_boxes = [shapely_box(*box) for box in image_target_boxes]

            total_predictions = len(image_predicted_boxes)
            total_targets = len(image_target_boxes)
            if total_predictions == 0 or total_targets == 0:
                continue

            ground_truth_union = image_target_boxes[0]
            for image_target_box in image_target_boxes[1:]:
                ground_truth_union = ground_truth_union.union(image_target_box)

            prediction_union = image_predicted_boxes[0]
            for image_predicted_box in image_predicted_boxes[1:]:
                prediction_union = prediction_union.union(image_predicted_box)

            prediction_ground_truth_intersection = prediction_union.intersection(ground_truth_union)

            ground_truth_union = ground_truth_union.area
            prediction_union = prediction_union.area
            prediction_ground_truth_intersection = prediction_ground_truth_intersection.area

            continuous_recalls[image_index] = torch_tensor(
                prediction_ground_truth_intersection / max(ground_truth_union, 1e-10))
            continuous_precisions[image_index] = torch_tensor(
                prediction_ground_truth_intersection / max(prediction_union, 1e-10))

        overall_recall = (image_ground_truths * continuous_recalls).sum() / max(
            image_ground_truths.sum(), torch_tensor(1e-10))
        overall_precision = (image_predictions * continuous_precisions).sum() / max(
            image_predictions.sum(), torch_tensor(1e-10))

        return overall_recall.item(), overall_precision.item(), continuous_recalls.tolist(), \
               continuous_precisions.tolist()

    def _yolo_discrete_metrics(self, predictions, targets, iou_threshold=0.5):
        predictions = np.array(predictions['values'])
        targets = np.array(targets['values'])

        average_precisions = []
        recalls = []
        precisions = []
        for class_id in range(self.num_classes):
            total_predicted_images = len(predictions)
            class_detections = [list(filter(
                lambda x: x[0] == class_id, predictions[image_index])) for image_index in range(total_predicted_images)]

            class_targets = []
            for i in range(len(targets)):
                result = []
                for target in targets[i]:
                    if target[0] == class_id:
                        result.append(target)
                class_targets.append(result)
            class_targets = np.array(class_targets)
            have_been_detected = [False] * sum([len(ct) for ct in class_targets])

            total_class_ground_truth = len(class_targets)
            total_class_detection = len(class_detections)
            true_positives = np.zeros(total_class_detection)
            false_positives = np.zeros(total_class_detection)

            for image_index, detection in enumerate(class_detections):
                if len(detection) == 0:
                    continue

                if len(class_targets[image_index]) == 0:
                    false_positives[image_index] = 1.
                    continue

                confidences = np.array([float(x[1]) for x in detection])
                bounding_boxes = np.array([[float(coord) for coord in x[2:]] for x in detection])

                sorted_indices = np.argsort(-confidences)
                bounding_boxes = bounding_boxes[sorted_indices, :]
                detected_bounding_boxes = bounding_boxes.astype(float)

                ground_truth_bounding_boxes = np.array([class_target[1:] for class_target in class_targets[
                    image_index]]).astype(float)

                maximum_overlap = -np.inf
                maximum_overlap_index = np.inf

                for detected_bounding_box in detected_bounding_boxes:
                    if ground_truth_bounding_boxes.size > 0:
                        intersection_x_min = np.maximum(ground_truth_bounding_boxes[:, 0], detected_bounding_box[0])
                        intersection_y_min = np.maximum(ground_truth_bounding_boxes[:, 1], detected_bounding_box[1])
                        intersection_x_max = np.minimum(ground_truth_bounding_boxes[:, 2], detected_bounding_box[2])
                        intersection_y_max = np.minimum(ground_truth_bounding_boxes[:, 3], detected_bounding_box[3])
                        intersection_width = np.maximum(intersection_x_max - intersection_x_min + 1., 0.)
                        intersection_height = np.maximum(intersection_y_max - intersection_y_min + 1., 0.)

                        intersection_area = intersection_width * intersection_height

                        union_x_min = detected_bounding_box[2] - detected_bounding_box[0] + 1.
                        union_y_min = detected_bounding_box[3] - detected_bounding_box[1] + 1.
                        union_x_max = ground_truth_bounding_boxes[:, 2] - ground_truth_bounding_boxes[:, 0] + 1.
                        union_y_max = ground_truth_bounding_boxes[:, 3] - ground_truth_bounding_boxes[:, 1] + 1.

                        union_area = union_x_min * union_y_min + union_x_max * union_y_max - intersection_area

                        overlaps = intersection_area / union_area
                        maximum_overlap = np.max(overlaps)
                        maximum_overlap_index = np.argmax(overlaps)

                    if maximum_overlap > iou_threshold:
                        if not have_been_detected[maximum_overlap_index]:
                            true_positives[image_index] = 1.
                            have_been_detected[maximum_overlap_index] = True
                        else:
                            false_positives[image_index] = 1.
                    else:
                        false_positives[image_index] = 1.

            cumul_true_positives = np.cumsum(true_positives)
            cumul_false_positives = np.cumsum(false_positives)

            class_recalls = cumul_true_positives / max(float(total_class_ground_truth), np.finfo(np.float64).eps)
            class_precisions = cumul_true_positives / np.maximum(
                cumul_true_positives + cumul_false_positives, np.finfo(np.float64).eps)

            class_average_precision = 0.
            recall_thresholds = [x / 10 for x in range(11)]
            for threshold in recall_thresholds:
                if np.sum(class_recalls >= threshold) == 0:
                    interpolated_precision = 0
                else:
                    interpolated_precision = np.max(class_precisions[class_recalls >= threshold])
                class_average_precision += interpolated_precision
            class_average_precision /= len(recall_thresholds)

            average_precisions.append(class_average_precision)
            total_true_positives = np.sum(true_positives)
            recalls.append(total_true_positives / np.maximum(total_class_ground_truth, np.finfo(np.float64).eps))
            precisions.append(total_true_positives / np.maximum(total_true_positives + np.sum(false_positives),
                                                                np.finfo(np.float64).eps))
        return average_precisions, recalls, precisions

    def _yolo_continuous_metrics(self, predictions, targets):
        predicted_boxes = predictions['values']
        target_boxes = targets['values']

        predicted_boxes = [Tensor([box[2:] for box in image]) for image in predicted_boxes]
        target_boxes = [Tensor([box[1:] for box in image]) for image in target_boxes]

        assert len(predicted_boxes) == len(target_boxes)
        total_images = len(target_boxes)

        image_ground_truths = LongTensor([target.size(0) for target in target_boxes])
        image_predictions = LongTensor([prediction.size(0) for prediction in predicted_boxes])
        continuous_recalls = torch_zeros(total_images, dtype=torch_float)
        continuous_precisions = torch_zeros(total_images, dtype=torch_float)
        for image_index in range(total_images):
            image_predicted_boxes = (predicted_boxes[image_index]).tolist()
            image_target_boxes = (target_boxes[image_index]).tolist()

            image_predicted_boxes = [shapely_box(*box) for box in image_predicted_boxes]
            image_target_boxes = [shapely_box(*box) for box in image_target_boxes]

            total_predictions = len(image_predicted_boxes)
            total_targets = len(image_target_boxes)
            if total_predictions == 0 or total_targets == 0:
                continue

            ground_truth_union = image_target_boxes[0]
            for image_target_box in image_target_boxes[1:]:
                ground_truth_union = ground_truth_union.union(image_target_box)

            prediction_union = image_predicted_boxes[0]
            for image_predicted_box in image_predicted_boxes[1:]:
                prediction_union = prediction_union.union(image_predicted_box)

            prediction_ground_truth_intersection = prediction_union.intersection(ground_truth_union)

            ground_truth_union = ground_truth_union.area
            prediction_union = prediction_union.area
            prediction_ground_truth_intersection = prediction_ground_truth_intersection.area

            continuous_recalls[image_index] = torch_tensor(
                prediction_ground_truth_intersection / max(ground_truth_union, 1e-10))
            continuous_precisions[image_index] = torch_tensor(
                prediction_ground_truth_intersection / max(prediction_union, 1e-10))

        overall_recall = (image_ground_truths * continuous_recalls).sum() / max(
            image_ground_truths.sum(), torch_tensor(1e-10))
        overall_precision = (image_predictions * continuous_precisions).sum() / max(
            image_predictions.sum(), torch_tensor(1e-10))

        return overall_recall.item(), overall_precision.item(), continuous_recalls.tolist(), \
               continuous_precisions.tolist()


class DetectionByBoxCenterPointMetric(DetectionMetric):
    def __init__(self, type, num_classes):
        super().__init__(type, num_classes)

    def _ssd_discrete_metrics(self, predictions, targets, is_cuda=False, *unused_args, **unused_kwargs):
        def __to_cuda(obj):
            if is_cuda:
                obj = obj.cuda()
            return obj

        predicted_boxes = predictions['boxes']
        predicted_labels = predictions['labels']
        predicted_class_scores = predictions['scores']

        target_boxes = targets['boxes']
        target_labels = targets['labels']

        assert len(predicted_boxes) == len(predicted_labels) == len(predicted_class_scores) == len(
            target_boxes) == len(target_labels)

        target_images = list()
        for i in range(len(target_labels)):
            target_images.extend([i] * target_labels[i].size(0))
        target_images = __to_cuda(LongTensor(target_images))
        target_boxes = torch_cat(target_boxes, dim=0)
        target_labels = torch_cat(target_labels, dim=0)

        assert target_images.size(0) == target_boxes.size(0) == target_labels.size(0)

        predicted_images = list()
        for i in range(len(predicted_labels)):
            predicted_images.extend([i] * predicted_labels[i].size(0))
        predicted_images = __to_cuda(LongTensor(predicted_images))
        predicted_boxes = torch_cat(predicted_boxes, dim=0)
        predicted_labels = torch_cat(predicted_labels, dim=0)
        predicted_class_scores = torch_cat(predicted_class_scores, dim=0)

        assert predicted_images.size(0) == predicted_boxes.size(0) == predicted_labels.size(
            0) == predicted_class_scores.size(0)

        average_precisions = torch_zeros(self.num_classes, dtype=torch_float)
        recalls = torch_zeros(self.num_classes, dtype=torch_float)
        precisions = torch_zeros(self.num_classes, dtype=torch_float)
        for c in range(self.num_classes):
            target_class_images = target_images[target_labels == c]
            target_class_boxes = target_boxes[target_labels == c]

            total_objects = target_class_boxes.size(0)

            target_class_boxes_detected = __to_cuda(torch_zeros(total_objects, dtype=torch_uint8))

            class_c_predicted_images = predicted_images[predicted_labels == c]
            class_c_predicted_boxes = predicted_boxes[predicted_labels == c]
            class_c_predicted_class_scores = predicted_class_scores[predicted_labels == c]
            class_c_num_detections = class_c_predicted_boxes.size(0)
            if class_c_num_detections == 0:
                continue

            class_c_predicted_class_scores, sort_ind = torch_sort(class_c_predicted_class_scores, dim=0,
                                                                  descending=True)
            class_c_predicted_images = class_c_predicted_images[sort_ind]
            class_c_predicted_boxes = class_c_predicted_boxes[sort_ind]

            true_positives = __to_cuda(torch_zeros(class_c_num_detections, dtype=torch_float))
            false_positives = __to_cuda(torch_zeros(class_c_num_detections, dtype=torch_float))
            for d in range(class_c_num_detections):
                this_detection_box = shapely_box(*class_c_predicted_boxes[d].data)
                this_image = class_c_predicted_images[d]

                object_boxes = target_class_boxes[target_class_images == this_image]
                if object_boxes.size(0) == 0:
                    false_positives[d] = 1
                    continue

                ground_truth_contains_prediction_center = [
                    shapely_box(*box.data).contains(this_detection_box.centroid) for box in object_boxes]
                for ind, prediction_center_in_ground_truth in enumerate(ground_truth_contains_prediction_center):
                    original_ind = LongTensor(range(target_class_boxes.size(0)))[target_class_images == this_image][ind]

                    if prediction_center_in_ground_truth:
                        if target_class_boxes_detected[original_ind] == 0:
                            true_positives[d] = 1
                            target_class_boxes_detected[original_ind] = 1
                        else:
                            false_positives[d] = 1
                    else:
                        false_positives[d] = 1

            cumul_true_positives = torch_cumsum(true_positives, dim=0)
            cumul_false_positives = torch_cumsum(false_positives, dim=0)
            cumul_precision = cumul_true_positives / (cumul_true_positives + cumul_false_positives + 1e-10)
            cumul_recall = cumul_true_positives / total_objects

            recall_thresholds = [x / 10 for x in range(11)]
            interpolated_precisions = __to_cuda(torch_zeros((len(recall_thresholds)), dtype=torch_float))
            for i, threshold in enumerate(recall_thresholds):
                recalls_above_threshold = cumul_recall >= threshold
                if recalls_above_threshold.any():
                    interpolated_precisions[i] = cumul_precision[recalls_above_threshold].max()
                else:
                    interpolated_precisions[i] = 0.
            average_precisions[c] = interpolated_precisions.mean()

            total_true_positives = torch_sum(true_positives)
            recalls[c] = total_true_positives / max(float(total_objects), 1e-10)
            precisions[c] = total_true_positives / max(
                total_true_positives + torch_sum(false_positives), torch_tensor(1e-10))
        return average_precisions.tolist(), recalls.tolist(), precisions.tolist()

    def _yolo_discrete_metrics(self, predictions, targets, *unused_args, **unused_kwargs):
        predictions = np.array(predictions['values'])
        targets = np.array(targets['values'])

        average_precisions = []
        recalls = []
        precisions = []
        for class_id in range(self.num_classes):
            total_predicted_images = len(predictions)
            class_detections = [list(filter(
                lambda x: x[0] == class_id, predictions[image_index])) for image_index in range(total_predicted_images)]

            class_targets = []
            for i in range(len(targets)):
                result = []
                for target in targets[i]:
                    if target[0] == class_id:
                        result.append(target)
                class_targets.append(result)
            class_targets = np.array(class_targets)
            have_been_detected = [False] * sum([len(ct) for ct in class_targets])

            total_class_ground_truth = len(class_targets)
            total_class_detection = len(class_detections)
            true_positives = np.zeros(total_class_detection)
            false_positives = np.zeros(total_class_detection)

            for image_index, detection in enumerate(class_detections):
                if len(detection) == 0:
                    continue

                if len(class_targets[image_index]) == 0:
                    false_positives[image_index] = 1.
                    continue

                confidences = np.array([float(x[1]) for x in detection])
                bounding_boxes = np.array([[float(coord) for coord in x[2:]] for x in detection])

                sorted_indices = np.argsort(-confidences)
                bounding_boxes = bounding_boxes[sorted_indices, :]
                detected_bounding_boxes = bounding_boxes.astype(float)

                ground_truth_bounding_boxes = np.array([class_target[1:] for class_target in class_targets[
                    image_index]]).astype(float)

                for detected_bounding_box in detected_bounding_boxes:
                    prediction_box_center = shapely_box(*detected_bounding_box).centroid
                    ground_truth_contains_prediction_center = [
                        shapely_box(*box).contains(prediction_box_center) for box in ground_truth_bounding_boxes]

                    for ind, contains in enumerate(ground_truth_contains_prediction_center):
                        if contains:
                            if not have_been_detected[ind]:
                                true_positives[image_index] = 1.
                                have_been_detected[ind] = True
                            else:
                                false_positives[image_index] = 1.
                        else:
                            false_positives[image_index] = 1.

            cumul_true_positives = np.cumsum(true_positives)
            cumul_false_positives = np.cumsum(false_positives)

            class_recalls = cumul_true_positives / max(float(total_class_ground_truth), np.finfo(np.float64).eps)
            class_precisions = cumul_true_positives / np.maximum(
                cumul_true_positives + cumul_false_positives, np.finfo(np.float64).eps)

            class_average_precision = 0.
            recall_thresholds = [x / 10 for x in range(11)]
            for threshold in recall_thresholds:
                if np.sum(class_recalls >= threshold) == 0:
                    interpolated_precision = 0
                else:
                    interpolated_precision = np.max(class_precisions[class_recalls >= threshold])
                class_average_precision += interpolated_precision
            class_average_precision /= len(recall_thresholds)

            average_precisions.append(class_average_precision)
            total_true_positives = np.sum(true_positives)
            recalls.append(total_true_positives / np.maximum(total_class_ground_truth, np.finfo(np.float64).eps))
            precisions.append(total_true_positives / np.maximum(total_true_positives + np.sum(false_positives),
                                                                np.finfo(np.float64).eps))
        return average_precisions, recalls, precisions
