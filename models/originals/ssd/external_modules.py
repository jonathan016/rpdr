from collections import OrderedDict

from torch import arange as torch_arange, cat as torch_cat, exp as torch_exp, max as torch_max, min as torch_min, \
    clamp as torch_clamp, load as torch_load
from torch.nn import Module, Conv2d, MaxPool2d, functional as F
from torch.nn.init import xavier_uniform_ as torch_nn_xavier_uniform, constant_ as torch_nn_constant
from torchvision.models import vgg16 as torchvision_vgg16

from models.originals.ssd.internal_modules import PriorBoxesConfig


def _decimate(tensor, m):
    """Decimate a tensor by factor m to convert fully connected layers to equivalent convolutional layers as used in
    SSD rewrite in PyTorch.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Some
    modifications are made. All credits to @sgrvinod.
    """

    assert tensor.dim() == len(m)

    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d, index=torch_arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor


def cxcy_to_xy(cxcy):
    """Calculation of boundary coordinates calculation from center-size coordinates as used in SSD rewrite in PyTorch.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Some
    modifications are made. All credits to @sgrvinod.
    """

    return torch_cat([cxcy[:, :2] - (cxcy[:, 2:] / 2), cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """Decodes bounding boxes from the corresponding prior boxes, both in center-size coordinates form, as used in SSD
    rewrite in PyTorch.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Some 
    modifications are made. All credits to @sgrvinod.
    """

    return torch_cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],
                      torch_exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)


def _find_intersection(set1, set2):
    """Calculation of intersection of every box combination between two sets of boxes that are in boundary
    coordinates as used in SSD rewrite in PyTorch.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Some 
    modifications are made. All credits to @sgrvinod.
    """

    lower_bounds = torch_max(set1[:, :2].unsqueeze(1), set2[:, :2].unsqueeze(0))
    upper_bounds = torch_min(set1[:, 2:].unsqueeze(1), set2[:, 2:].unsqueeze(0))

    intersection_dims = torch_clamp(upper_bounds - lower_bounds, min=0)

    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]


def find_jaccard_overlap(set1, set2):
    """Calculation of Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary
    coordinates as used in SSD rewrite in PyTorch.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Some 
    modifications are made. All credits to @sgrvinod.
    """

    intersection = _find_intersection(set1, set2)

    set1_areas = (set1[:, 2] - set1[:, 0]) * (set1[:, 3] - set1[:, 1])
    set2_areas = (set2[:, 2] - set2[:, 0]) * (set2[:, 3] - set2[:, 1])

    union = set1_areas.unsqueeze(1) + set2_areas.unsqueeze(0) - intersection

    return intersection / union


class VGGBase(Module):
    """VGG base model as used by SSD for extracting features.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Some 
    modifications are made. All credits to @sgrvinod.
    """

    def __init__(self, load_pretrained=True):
        super(VGGBase, self).__init__()

        self.conv1_1 = Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4_1 = Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv6 = Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = Conv2d(1024, 1024, kernel_size=1)

        if load_pretrained is not False: self._load_pretrained_layers(load_pretrained)

    def forward(self, image):
        out = F.relu(self.conv1_1(image))
        out = F.relu(self.conv1_2(out))
        out = self.pool1(out)

        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = self.pool2(out)

        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out))
        out = F.relu(self.conv3_3(out))
        out = self.pool3(out)

        out = F.relu(self.conv4_1(out))
        out = F.relu(self.conv4_2(out))
        out = F.relu(self.conv4_3(out))
        conv4_3_features = out
        out = self.pool4(out)

        out = F.relu(self.conv5_1(out))
        out = F.relu(self.conv5_2(out))
        out = F.relu(self.conv5_3(out))
        out = self.pool5(out)

        out = F.relu(self.conv6(out))

        conv7_features = F.relu(self.conv7(out))

        return conv4_3_features, conv7_features

    def _load_pretrained_layers(self, load_pretrained):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        pretrained_state_dict = self._load_pretrained_base(load_pretrained)
        pretrained_param_names = list(pretrained_state_dict.keys())

        for i, param in enumerate(param_names[:-4]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']
        state_dict['conv6.weight'] = _decimate(conv_fc6_weight, m=[4, None, 3, 3])
        state_dict['conv6.bias'] = _decimate(conv_fc6_bias, m=[4])

        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']
        state_dict['conv7.weight'] = _decimate(conv_fc7_weight, m=[4, 4, None, None])
        state_dict['conv7.bias'] = _decimate(conv_fc7_bias, m=[4])

        self.load_state_dict(state_dict)

    def _load_pretrained_base(self, load_pretrained):
        if type(load_pretrained) == bool:
            return torchvision_vgg16(pretrained=load_pretrained).state_dict()
        elif type(load_pretrained) == OrderedDict:
            return load_pretrained
        else:
            return torch_load(load_pretrained)


class AuxiliaryConvolutions(Module):
    """Auxiliary convolutions as used by SSD.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Some 
    modifications are made. All credits to @sgrvinod.
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        self.conv8_1 = Conv2d(1024, 256, kernel_size=1, padding=0)
        self.conv8_2 = Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv9_1 = Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv10_1 = Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = Conv2d(128, 256, kernel_size=3, padding=0)

        self.conv11_1 = Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = Conv2d(128, 256, kernel_size=3, padding=0)

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, Conv2d):
                torch_nn_xavier_uniform(c.weight)
                torch_nn_constant(c.bias, 0.)

    def forward(self, conv7_feats):
        out = F.relu(self.conv8_1(conv7_feats))
        out = F.relu(self.conv8_2(out))
        conv8_2_features = out

        out = F.relu(self.conv9_1(out))
        out = F.relu(self.conv9_2(out))
        conv9_2_features = out

        out = F.relu(self.conv10_1(out))
        out = F.relu(self.conv10_2(out))
        conv10_2_features = out

        out = F.relu(self.conv11_1(out))
        conv11_2_features = F.relu(self.conv11_2(out))

        return conv8_2_features, conv9_2_features, conv10_2_features, conv11_2_features


class PredictionConvolutions(Module):
    """Final predictor convolutions as used by SSD after the auxiliary convolutions.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Some 
    modifications are made. All credits to @sgrvinod.
    """

    def __init__(self, num_classes, prior_boxes: PriorBoxesConfig):
        super(PredictionConvolutions, self).__init__()

        self.num_classes = num_classes

        prior_boxes = prior_boxes.value()

        self.loc_conv4_3 = Conv2d(512, prior_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = Conv2d(1024, prior_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = Conv2d(512, prior_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = Conv2d(256, prior_boxes['conv9_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv10_2 = Conv2d(256, prior_boxes['conv10_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv11_2 = Conv2d(256, prior_boxes['conv11_2'] * 4, kernel_size=3, padding=1)

        self.cl_conv4_3 = Conv2d(512, prior_boxes['conv4_3'] * num_classes, kernel_size=3, padding=1)
        self.cl_conv7 = Conv2d(1024, prior_boxes['conv7'] * num_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = Conv2d(512, prior_boxes['conv8_2'] * num_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = Conv2d(256, prior_boxes['conv9_2'] * num_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = Conv2d(256, prior_boxes['conv10_2'] * num_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = Conv2d(256, prior_boxes['conv11_2'] * num_classes, kernel_size=3, padding=1)

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, Conv2d):
                torch_nn_xavier_uniform(c.weight)
                torch_nn_constant(c.bias, 0.)

    def forward(self, conv4_3_features, conv7_features, conv8_2_features, conv9_2_features, conv10_2_features,
                conv11_2_features):
        batch_size = conv4_3_features.size(0)

        locations = self._predict_locations(
            batch_size, conv4_3_features, conv7_features, conv8_2_features, conv9_2_features, conv10_2_features,
            conv11_2_features)
        locations = torch_cat(list(locations), dim=1)

        classes_scores = self._predict_classes(
            batch_size, conv4_3_features, conv7_features, conv8_2_features, conv9_2_features, conv10_2_features,
            conv11_2_features)
        classes_scores = torch_cat(list(classes_scores), dim=1)

        return locations, classes_scores

    def __layer_predict_classes(self, layer, features, batch_size):
        out = layer(features)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(batch_size, -1, self.num_classes)

    def _predict_classes(self, batch_size, conv4_3_features, conv7_features, conv8_2_features, conv9_2_features,
                         conv10_2_features, conv11_2_features):
        c_conv4_3 = self.__layer_predict_classes(self.cl_conv4_3, conv4_3_features, batch_size)
        c_conv7 = self.__layer_predict_classes(self.cl_conv7, conv7_features, batch_size)
        c_conv8_2 = self.__layer_predict_classes(self.cl_conv8_2, conv8_2_features, batch_size)
        c_conv9_2 = self.__layer_predict_classes(self.cl_conv9_2, conv9_2_features, batch_size)
        c_conv10_2 = self.__layer_predict_classes(self.cl_conv10_2, conv10_2_features, batch_size)
        c_conv11_2 = self.__layer_predict_classes(self.cl_conv11_2, conv11_2_features, batch_size)

        return c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2

    def __layer_predict_locations(self, layer, features, batch_size):
        out = layer(features)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(batch_size, -1, 4)

    def _predict_locations(self, batch_size, conv4_3_features, conv7_features, conv8_2_features, conv9_2_features,
                           conv10_2_features, conv11_2_features):
        l_conv4_3 = self.__layer_predict_locations(self.loc_conv4_3, conv4_3_features, batch_size)
        l_conv7 = self.__layer_predict_locations(self.loc_conv7, conv7_features, batch_size)
        l_conv8_2 = self.__layer_predict_locations(self.loc_conv8_2, conv8_2_features, batch_size)
        l_conv9_2 = self.__layer_predict_locations(self.loc_conv9_2, conv9_2_features, batch_size)
        l_conv10_2 = self.__layer_predict_locations(self.loc_conv10_2, conv10_2_features, batch_size)
        l_conv11_2 = self.__layer_predict_locations(self.loc_conv11_2, conv11_2_features, batch_size)

        return l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2
