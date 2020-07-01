from torch import LongTensor, device as torch_device, zeros as torch_zeros, float as torch_float, long as torch_long, \
    cat as torch_cat, log as torch_log
from torch.nn import Module, L1Loss, CrossEntropyLoss

from .external_modules import cxcy_to_xy, find_jaccard_overlap


class MultiBoxLoss(Module):
    """MultiBox loss as used by SSD.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Some 
    modifications are made. All credits to @sgrvinod.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()

        self.is_cuda = False

        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = L1Loss()
        self.cross_entropy = CrossEntropyLoss(reduction='none')

    def _get_device(self):
        return torch_device("cuda" if self.is_cuda else "cpu")

    def set_cuda(self, is_cuda: bool = False):
        self.is_cuda = is_cuda

    def forward(self, predicted_locations, predicted_scores, boxes, labels):
        batch_size = predicted_locations.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locations.size(1) == predicted_scores.size(1)

        ground_truth_locations = torch_zeros((batch_size, n_priors, 4), dtype=torch_float).to(self._get_device())
        ground_truth_classes = torch_zeros((batch_size, n_priors), dtype=torch_long).to(self._get_device())

        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)

            _, prior_for_each_object = overlap.max(dim=1)
            object_for_each_prior[prior_for_each_object] = LongTensor(range(n_objects)).to(self._get_device())
            overlap_for_each_prior[prior_for_each_object] = 1.

            label_for_each_prior = labels[i][object_for_each_prior]
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0

            ground_truth_classes[i] = label_for_each_prior
            ground_truth_locations[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)

        positive_priors = ground_truth_classes != 0

        localization_loss = self.smooth_l1(
            predicted_locations[positive_priors], ground_truth_locations[positive_priors])

        n_positives = positive_priors.sum(dim=1)
        n_hard_negatives = self.neg_pos_ratio * n_positives

        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), ground_truth_classes.view(-1)).view(
            batch_size, n_priors)
        conf_loss_pos = conf_loss_all[positive_priors]

        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[positive_priors] = 0.
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)
        hardness_ranks = LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(self._get_device())
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]

        confidence_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()

        return confidence_loss + self.alpha * localization_loss


def xy_to_cxcy(xy):
    """Calculation of center-size coordinates calculation from boundary coordinates as used in SSD rewrite in PyTorch.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Some 
    modifications are made. All credits to @sgrvinod.
    """

    return torch_cat([(xy[:, 2:] + xy[:, :2]) / 2, xy[:, 2:] - xy[:, :2]], 1)


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """Encodes bounding boxes to the corresponding prior boxes, both in center-size coordinates form, as used in SSD
    rewrite in PyTorch.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Some 
    modifications are made. All credits to @sgrvinod.
    """

    # https://github.com/weiliu89/caffe/issues/155
    return torch_cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),
                      torch_log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)
