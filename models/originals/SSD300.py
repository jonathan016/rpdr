from torch import FloatTensor, LongTensor, device as torch_device, cat as torch_cat, zeros as torch_zeros, \
    max as torch_max, uint8 as torch_uint8
from torch.nn import Module, Parameter, functional as F
from torch.nn.init import constant_ as torch_nn_constant
from math import sqrt as math_sqrt

from models.modules.external_modules import VGGBase, AuxiliaryConvolutions, PredictionConvolutions, cxcy_to_xy, \
    gcxgcy_to_cxcy, find_jaccard_overlap


class SSD300(Module):
    """SSD300 rewrite in PyTorch.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Modifications
    are made for variable names only. All credits to @sgrvinod.
    """

    def __init__(self, num_classes, load_pretrained_base=True, prior_boxes=PredictionConvolutions.PriorBoxesConfig()):
        super(SSD300, self).__init__()

        self.is_cuda = False

        self.num_classes = num_classes

        self.base = VGGBase(load_pretrained=load_pretrained_base)
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(num_classes, prior_boxes)

        self.rescale_factors = Parameter(FloatTensor(1, 512, 1, 1))
        torch_nn_constant(self.rescale_factors, 20)

        self.priors_cxcy = self.create_prior_boxes()

    def _get_device(self):
        return torch_device("cuda" if self.is_cuda else "cpu")

    def set_cuda(self, is_cuda: bool = False):
        self.is_cuda = is_cuda

    def forward(self, image):
        conv4_3_feats, conv7_feats = self._base_predict(image)
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_convs(conv7_feats)

        locations, classes_scores = self.pred_convs(
            conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats)

        return locations, classes_scores

    def _base_predict(self, image):
        conv4_3_feats, conv7_feats = self.base(image)
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()
        conv4_3_feats = conv4_3_feats / norm
        conv4_3_feats = conv4_3_feats * self.rescale_factors
        return conv4_3_feats, conv7_feats

    def create_prior_boxes(self):
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

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

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

        prior_boxes = FloatTensor(prior_boxes).to(self._get_device())
        prior_boxes.clamp_(0, 1)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
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

            # max_scores, best_label = predicted_scores[i].max(dim=1)

            for c in range(1, self.num_classes):
                class_scores = predicted_scores[i][:, c]
                score_above_min_score = class_scores > min_score
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0: continue
                class_scores = class_scores[score_above_min_score]
                class_decoded_locs = decoded_locs[score_above_min_score]

                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
                class_decoded_locs = class_decoded_locs[sort_ind]

                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)

                suppress = torch_zeros((n_above_min_score), dtype=torch_uint8).to(self._get_device())
                for box in range(class_decoded_locs.size(0)):
                    if suppress[box] == 1: continue

                    suppress = torch_max(suppress, overlap[box] > max_overlap)
                    suppress[box] = 0

                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(LongTensor((1 - suppress).sum().item() * [c]).to(self._get_device()))
                image_scores.append(class_scores[1 - suppress])

            if len(image_boxes) == 0:
                image_boxes.append(FloatTensor([[0., 0., 1., 1.]]).to(self._get_device()))
                image_labels.append(LongTensor([0]).to(self._get_device()))
                image_scores.append(FloatTensor([0.]).to(self._get_device()))

            image_boxes = torch_cat(image_boxes, dim=0)
            image_labels = torch_cat(image_labels, dim=0)
            image_scores = torch_cat(image_scores, dim=0)
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
