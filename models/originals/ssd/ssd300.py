from math import sqrt as math_sqrt
from typing import Optional, Union

from torch import FloatTensor, LongTensor, device as torch_device, cat as torch_cat, zeros as torch_zeros, \
    max as torch_max, uint8 as torch_uint8, BoolTensor
from torch.nn import Module, Parameter, functional as F
from torch.nn.init import constant_ as torch_nn_constant

from .external_modules import cxcy_to_xy, gcxgcy_to_cxcy, find_jaccard_overlap, VGGBase, AuxiliaryConvolutions, \
    PredictionConvolutions
from .internal_modules import PriorBoxesConfig
from .loss_modules import MultiBoxLoss


class SSD300(Module):
    """SSD300 rewrite in PyTorch.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Some
    modifications are made. All credits to @sgrvinod.

    The main modification made in this implementation is for the loss calculation, in which rather than creating the
    loss object outside this model's implementation, the loss object is automatically created when instantiating this
    class. To calculate loss, simply call ``model.loss(target).backward()``, since the prediction values are kept after
    forward pass.

    However, SSD accepts two target values, which are the **ground truth locations** and the **ground truth classes
    scores** as specified in ``GroZiDetectionDataset._format_ssd()`` method. Therefore, to calculate the loss,
    one can do the following: ::

            >>> model = SSD300(**constructor_args)
            >>> for data, *target in ssd_dataset:
            >>>     predicted_locations, predicted_classes_scores = model(data)
            >>>     loss = model.loss(*target)
            >>>     loss.backward()

    Arguments:
        num_classes: The number of possible known objects to detect and recognize.
        load_pretrained_base: A flag denoting whether to load pretrained base network (VGG-16) weights or not.
        prior_boxes: The configuration of number of prior boxes per detection layer in SSD.
        aspect_ratios: The configuration of aspect ratios to be used in the prior boxes construction.
    """

    def __init__(self, num_classes, load_pretrained_base=True, prior_boxes=PriorBoxesConfig(),
                 aspect_ratios=PriorBoxesConfig()):
        super(SSD300, self).__init__()

        self.is_cuda = False

        self.num_classes = num_classes

        self.base = VGGBase(load_pretrained=load_pretrained_base)
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(num_classes, prior_boxes)

        self.rescale_factors = Parameter(FloatTensor(1, 512, 1, 1))
        torch_nn_constant(self.rescale_factors, 20)

        self.aspect_ratios = aspect_ratios.value()
        self.priors_cxcy = self.create_prior_boxes(self.aspect_ratios)

        self.predicted_locations = None
        self.predicted_classes_scores = None
        self.loss_function = MultiBoxLoss(self.priors_cxcy)

    def _get_device(self):
        return torch_device("cuda" if self.is_cuda else "cpu")

    def _to_cuda(self, obj):
        if self.is_cuda:
            return obj.cuda()
        return obj

    def set_cuda(self, is_cuda: bool = False):
        self.is_cuda = is_cuda
        self.loss_function.set_cuda(is_cuda)

    def forward(self, image):
        conv4_3_feats, conv7_feats = self._base_predict(image)
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_convs(conv7_feats)

        self.predicted_locations, self.predicted_classes_scores = self.pred_convs(
            conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats)

        return self.predicted_locations, self.predicted_classes_scores

    def _base_predict(self, image):
        conv4_3_feats, conv7_feats = self.base(image)
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()
        conv4_3_feats = conv4_3_feats / norm
        conv4_3_feats = conv4_3_feats * self.rescale_factors

        return conv4_3_feats, conv7_feats

    def create_prior_boxes(self, aspect_ratios):
        feature_map_dimensions = {'conv4_3': 38,
                                  'conv7': 19,
                                  'conv8_2': 10,
                                  'conv9_2': 5,
                                  'conv10_2': 3,
                                  'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        feature_maps = list(feature_map_dimensions.keys())

        prior_boxes = []

        for k, fmap in enumerate(feature_maps):
            for i in range(feature_map_dimensions[fmap]):
                for j in range(feature_map_dimensions[fmap]):
                    cx = (j + 0.5) / feature_map_dimensions[fmap]
                    cy = (i + 0.5) / feature_map_dimensions[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append(
                            [cx, cy, obj_scales[fmap] * math_sqrt(ratio), obj_scales[fmap] / math_sqrt(ratio)])

                        if ratio == 1.:
                            try:
                                additional_scale = math_sqrt(obj_scales[fmap] * obj_scales[feature_maps[k + 1]])
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = self._to_cuda(FloatTensor(prior_boxes))
        prior_boxes.clamp_(0, 1)

        return prior_boxes

    def loss(self, *target):
        target_locations = target[0]
        target_classes_scores = target[1]

        return self.loss_function(
            self.predicted_locations, self.predicted_classes_scores, target_locations, target_classes_scores)

    def detect_objects(self, image_as_tensor, min_score, max_overlap, top_k):
        predicted_locs, predicted_scores = self.forward(image_as_tensor)
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)

        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))

            image_boxes = list()
            image_labels = list()
            image_scores = list()

            for c in range(self.num_classes - 1):
                class_scores = predicted_scores[i][:, c]
                score_above_min_score = class_scores > min_score
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]
                class_decoded_locs = decoded_locs[score_above_min_score]

                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
                class_decoded_locs = class_decoded_locs[sort_ind]

                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)

                suppress = self._to_cuda(torch_zeros((n_above_min_score), dtype=torch_uint8))
                for box in range(class_decoded_locs.size(0)):
                    if suppress[box] == 1:
                        continue

                    suppress = torch_max(suppress, (overlap[box] > max_overlap).type(torch_uint8))
                    suppress[box] = 0

                kept_indices = self._to_cuda(suppress.type(BoolTensor).logical_not())
                locs = class_decoded_locs[kept_indices].tolist()
                for loc_index, loc in enumerate(locs):
                    locs[loc_index] = [max(loc[0], 0.), max(loc[1], 0.), min(loc[2], 1.), min(loc[3], 1.)]
                image_boxes.append(self._to_cuda(FloatTensor(locs)))
                image_labels.append(self._to_cuda(LongTensor(kept_indices.sum().item() * [c])))
                image_scores.append(self._to_cuda(class_scores[kept_indices]))

            if len(image_boxes) == 0:
                image_boxes.append(self._to_cuda(FloatTensor([[0., 0., 0., 0.]])))
                image_labels.append(self._to_cuda(LongTensor([120])))
                image_scores.append(self._to_cuda(FloatTensor([0.])))

            image_boxes = self._to_cuda(torch_cat(image_boxes, dim=0))
            image_labels = self._to_cuda(torch_cat(image_labels, dim=0))
            image_scores = self._to_cuda(torch_cat(image_scores, dim=0))
            n_objects = image_scores.size(0)

            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]
                image_boxes = image_boxes[sort_ind][:top_k]
                image_labels = image_labels[sort_ind][:top_k]

            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores

    def cuda(self, dev: Optional[Union[int, torch_device]] = None):
        self.set_cuda(True)
        self.loss_function.cuda(dev)
        self.priors_cxcy = self.create_prior_boxes(self.aspect_ratios).cuda(dev)
        self.loss_function.set_priors(self.priors_cxcy)
        return super().cuda(dev)

    def cpu(self):
        self.set_cuda(False)
        self.loss_function.cpu()
        self.priors_cxcy = self.priors_cxcy.cpu()
        self.loss_function.set_priors(self.priors_cxcy)
        return super().cpu()
