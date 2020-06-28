from torch import Tensor, cat as torch_cat, arange as torch_arange, max as torch_max, min as torch_min, \
    clamp as torch_clamp, exp as torch_exp
from torch.nn import Module, Conv2d, MaxPool2d, functional as F
from torch.nn.functional import avg_pool2d as torch_avg_pool2d
from torch.nn.init import xavier_uniform_ as torch_nn_xavier_uniform, constant_ as torch_nn_constant
from torchvision.models import vgg16 as torchvision_vgg16


class Identity(Module):
    """Identity activation due to unavailability in this project's PyTorch version.

    This is implemented as shown in https://github.com/pytorch/pytorch/pull/19249 by @MilesCranmer. All credits to
    @MilesCranmer.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


class Reorg(Module):
    """Reorg layer as used in YOLOv2 rewrite in PyTorch.

    This is implemented as shown in https://github.com/marvis/pytorch-yolo2. Modifications are made for variable
    names only. All credits to @marvis.
    """

    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, input_feature_map: Tensor):
        assert (input_feature_map.data.dim() == 4)
        batch = input_feature_map.data.size(0)
        channels = input_feature_map.data.size(1)
        height = input_feature_map.data.size(2)
        width = input_feature_map.data.size(3)

        assert (height % self.stride == 0)
        assert (width % self.stride == 0)
        width_stride = self.stride
        height_stride = self.stride

        output_height = int(height / height_stride)
        output_width = int(width / width_stride)
        output_feature_map = input_feature_map.view(batch, channels, output_height, height_stride, output_width,
                                                    width_stride).transpose(3, 4).contiguous()
        output_feature_map = output_feature_map.view(batch, channels, output_height * output_width,
                                                     height_stride * width_stride).transpose(2, 3).contiguous()
        output_feature_map = output_feature_map.view(batch, channels, height_stride * width_stride, output_height,
                                                     output_width).transpose(1, 2).contiguous()
        output_feature_map = output_feature_map.view(batch, height_stride * width_stride * channels, output_height,
                                                     output_width)

        return output_feature_map


class GlobalAvgPool2d(Module):
    """GlobalAvgPool2d layer as used in YOLOv2 rewrite in PyTorch.

    This is implemented as shown in https://github.com/marvis/pytorch-yolo2. Modifications are made for variable
    names only. All credits to @marvis.
    """

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, input: Tensor):
        batch = input.data.size(0)
        channels = input.data.size(1)
        height = input.data.size(2)
        width = input.data.size(3)

        output = torch_avg_pool2d(input, (height, width))
        output = output.view(batch, channels)

        return output


def _decimate(tensor, m):
    """Decimate a tensor by factor m to convert fully connected layers to equivalent convolutional layers as used in
    SSD rewrite in PyTorch.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Modifications
    are made for variable names only. All credits to @sgrvinod.
    """

    assert tensor.dim() == len(m)

    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d, index=torch_arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor


def cxcy_to_xy(cxcy):
    """Calculation of boundary coordinates calculation from center-size coordinates as used in SSD rewrite in PyTorch.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Modifications
    are made for variable names only. All credits to @sgrvinod.
    """

    return torch_cat([cxcy[:, :2] - (cxcy[:, 2:] / 2), cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """Decodes bounding boxes from the corresponding prior boxes, both in center-size coordinates form, as used in SSD
    rewrite in PyTorch.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Modifications
    are made for variable names only. All credits to @sgrvinod.
    """

    return torch_cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],
                      torch_exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)


def _find_intersection(set1, set2):
    """Calculation of intersection of every box combination between two sets of boxes that are in boundary
    coordinates as used in SSD rewrite in PyTorch.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Modifications
    are made for variable names only. All credits to @sgrvinod.
    """

    lower_bounds = torch_max(set1[:, :2].unsqueeze(1), set2[:, :2].unsqueeze(0))
    upper_bounds = torch_min(set1[:, 2:].unsqueeze(1), set2[:, 2:].unsqueeze(0))

    intersection_dims = torch_clamp(upper_bounds - lower_bounds, min=0)

    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]


def find_jaccard_overlap(set1, set2):
    """Calculation of Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary
    coordinates as used in SSD rewrite in PyTorch.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Modifications
    are made for variable names only. All credits to @sgrvinod.
    """

    intersection = _find_intersection(set1, set2)

    set1_areas = (set1[:, 2] - set1[:, 0]) * (set1[:, 3] - set1[:, 1])
    set2_areas = (set2[:, 2] - set2[:, 0]) * (set2[:, 3] - set2[:, 1])

    union = set1_areas.unsqueeze(1) + set2_areas.unsqueeze(0) - intersection

    return intersection / union


class VGGBase(Module):
    """VGG base model as used by SSD for extracting features.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Modifications
    are made for variable names only. All credits to @sgrvinod.
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

        if load_pretrained: self._load_pretrained_layers()

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

    def _load_pretrained_layers(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        pretrained_state_dict = torchvision_vgg16(pretrained=True).state_dict()
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


class AuxiliaryConvolutions(Module):
    """Auxiliary convolutions as used by SSD.

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Modifications
    are made for variable names only. All credits to @sgrvinod.
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

    This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Modifications
    are made for variable names only. All credits to @sgrvinod.
    """

    class PriorBoxesConfig:
        def __init__(self, config: dict = None):
            self.__config = {'conv4_3': 4,
                             'conv7': 6,
                             'conv8_2': 6,
                             'conv9_2': 6,
                             'conv10_2': 4,
                             'conv11_2': 4}

            if config: self.set_config(config)

        def set_config(self, config: dict):
            assert config.keys() == self.__config.keys()

            self.__config = config

        def value(self):
            return self.__config

        def conv4_3(self, num_boxes):
            self.__config['conv4_3'] = num_boxes

            return self

        def conv7(self, num_boxes):
            self.__config['conv7'] = num_boxes

            return self

        def conv8_2(self, num_boxes):
            self.__config['conv8_2'] = num_boxes

            return self

        def conv9_2(self, num_boxes):
            self.__config['conv9_2'] = num_boxes

            return self

        def conv10_2(self, num_boxes):
            self.__config['conv10_2'] = num_boxes

            return self

        def conv11_2(self, num_boxes):
            self.__config['conv11_2'] = num_boxes

            return self

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
