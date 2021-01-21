import sys
from argparse import ArgumentParser

import torch
from torch.nn import Module, Linear, Conv2d, functional as torch_fn, LeakyReLU, ReLU, MaxPool2d
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, Normalize


class GlobalAvgPool2d(Module):
    """GlobalAvgPool2d layer as used in YOLOv2 rewrite in PyTorch.

    This is implemented as shown in https://github.com/marvis/pytorch-yolo2. Modifications are made for variable
    names only. All credits to @marvis.
    """

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, input: torch.Tensor):
        batch = input.data.size(0)
        channels = input.data.size(1)
        height = input.data.size(2)
        width = input.data.size(3)

        output = torch.nn.functional.avg_pool2d(input, (height, width))
        output = output.view(batch, channels)

        return output


class Hydra(Module):
    def __init__(self, total_class: int, in_channel: int):
        super().__init__()
        for i in range(total_class):
            self.add_module(str(i), Linear(in_channel, 1))

    def forward(self, input: torch.Tensor):
        batch_size = input.size(0)
        out = torch.Tensor([[] for _ in range(batch_size)]).to(next(self.parameters()).device)
        for i, head in enumerate(self.modules()):
            if i == 0:  # Skip self's module
                continue
            out = torch.cat((out, head(input)), dim=1)
        return out


class VGG16Hydra(Module):
    def __init__(self, total_class, **vgg_kwargs):
        super().__init__()
        self.vgg_feature_extractor = vgg16(**vgg_kwargs).features
        self.global_average_pooling = GlobalAvgPool2d()
        self.hydra = Hydra(total_class, self.vgg_feature_extractor[-3].out_channels)

    def forward(self, x):
        x = self.vgg_feature_extractor(x)
        x = self.global_average_pooling(x)
        x = self.hydra(x)

        return x

    def cuda(self, device=None):
        self.vgg_feature_extractor = self.vgg_feature_extractor.cuda(device)
        self.hydra = self.hydra.cuda(device)
        return super().cuda(device)

    def cpu(self):
        self.vgg_feature_extractor = self.vgg_feature_extractor.cpu()
        self.hydra = self.hydra.cpu()
        return super().cpu()


class Hydra1Conv(Module):
    class Head(Module):
        def __init__(self):
            super().__init__()
            self.add_module('conv', Conv2d(512, 512, (3, 3)))
            self.add_module('pool', GlobalAvgPool2d())
            self.add_module('linear', Linear(512, 1))

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = self.linear(x)
            return x

    def __init__(self, total_class: int, in_channel: int):
        super().__init__()
        for i in range(total_class):
            self.add_module(str(i), self.Head())

    def forward(self, input: torch.Tensor):
        batch_size = input.size(0)
        out = torch.Tensor([[] for _ in range(batch_size)]).to(next(self.parameters()).device)
        for head in self.modules():
            if type(head) is not self.Head:
                continue
            out = torch.cat((out, head(input)), dim=1)
        return out


class VGG16Hydra1Conv(Module):
    def __init__(self, total_class, **vgg_kwargs):
        super().__init__()
        self.vgg_feature_extractor = vgg16(**vgg_kwargs).features
        self.hydra = Hydra1Conv(total_class, self.vgg_feature_extractor[-3].out_channels)

    def forward(self, x):
        x = self.vgg_feature_extractor(x)
        x = self.hydra(x)

        return x

    def cuda(self, device=None):
        self.vgg_feature_extractor = self.vgg_feature_extractor.cuda(device)
        self.hydra = self.hydra.cuda(device)
        return super().cuda(device)

    def cpu(self):
        self.vgg_feature_extractor = self.vgg_feature_extractor.cpu()
        self.hydra = self.hydra.cpu()
        return super().cpu()


class Hydra2Conv(Module):
    class Head(Module):
        def __init__(self):
            super().__init__()
            self.add_module('conv1', Conv2d(512, 512, (3, 3)))
            self.add_module('conv2', Conv2d(512, 512, (1, 1)))
            self.add_module('pool', GlobalAvgPool2d())
            self.add_module('linear', Linear(512, 1))

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pool(x)
            x = self.linear(x)
            return x

    def __init__(self, total_class: int, in_channel: int):
        super().__init__()
        for i in range(total_class):
            self.add_module(str(i), self.Head())

    def forward(self, input: torch.Tensor):
        batch_size = input.size(0)
        out = torch.Tensor([[] for _ in range(batch_size)]).to(next(self.parameters()).device)
        for head in self.modules():
            if type(head) is not self.Head:
                continue
            out = torch.cat((out, head(input)), dim=1)
        return out


class VGG16Hydra2Conv(Module):
    def __init__(self, total_class, **vgg_kwargs):
        super().__init__()
        self.vgg_feature_extractor = vgg16(**vgg_kwargs).features
        self.hydra = Hydra2Conv(total_class, self.vgg_feature_extractor[-3].out_channels)

    def forward(self, x):
        x = self.vgg_feature_extractor(x)
        x = self.hydra(x)

        return x

    def cuda(self, device=None):
        self.vgg_feature_extractor = self.vgg_feature_extractor.cuda(device)
        self.hydra = self.hydra.cuda(device)
        return super().cuda(device)

    def cpu(self):
        self.vgg_feature_extractor = self.vgg_feature_extractor.cpu()
        self.hydra = self.hydra.cpu()
        return super().cpu()


class VGG16Decimate(Module):
    def __init__(self, total_class, vgg_model, pretrained_state_dict):
        super().__init__()
        self.vgg_feature_extractor = vgg_model.features
        self.global_average_pooling = GlobalAvgPool2d()
        self.classifier_1 = Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.classifier_2 = Conv2d(1024, 1024, kernel_size=1)
        self.classifier_3 = Conv2d(1024, total_class, kernel_size=1)

        state_dict = self.state_dict()

        classifier_1_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)
        classifier_1_bias = pretrained_state_dict['classifier.0.bias']
        state_dict['classifier_1.weight'] = self._decimate(classifier_1_weight, m=[4, None, 3, 3])
        state_dict['classifier_2.bias'] = self._decimate(classifier_1_bias, m=[4])

        classifier_2_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)
        classifier_2_bias = pretrained_state_dict['classifier.3.bias']
        state_dict['classifier_2.weight'] = self._decimate(classifier_2_weight, m=[4, 4, None, None])
        state_dict['classifier_2.bias'] = self._decimate(classifier_2_bias, m=[4])

        self.load_state_dict(state_dict)

    def _decimate(self, tensor, m):
        """Decimate a tensor by factor m to convert fully connected layers to equivalent convolutional layers as used in
        SSD rewrite in PyTorch.

        This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Some
        modifications are made. All credits to @sgrvinod.
        """

        assert tensor.dim() == len(m)

        for d in range(tensor.dim()):
            if m[d] is not None:
                index = torch.arange(start=0, end=tensor.size(d), step=m[d]).long().to(next(self.parameters()).device)
                tensor = tensor.index_select(dim=d, index=index)

        return tensor

    def forward(self, x):
        x = self.vgg_feature_extractor(x)
        x = torch_fn.relu(self.classifier_1(x))
        x = torch_fn.relu(self.classifier_2(x))
        x = self.classifier_3(x)
        x = self.global_average_pooling(x)
        return x

    def cuda(self, device=None):
        self.vgg_feature_extractor = self.vgg_feature_extractor.cuda(device)
        self.classifier_1 = self.classifier_1.cuda(device)
        self.classifier_2 = self.classifier_2.cuda(device)
        self.classifier_3 = self.classifier_3.cuda(device)
        return super().cuda(device)

    def cpu(self):
        self.vgg_feature_extractor = self.vgg_feature_extractor.cpu()
        self.classifier_1 = self.classifier_1.cpu()
        self.classifier_2 = self.classifier_2.cpu()
        self.classifier_3 = self.classifier_3.cpu()
        return super().cpu()


class VGG16DecimateLeaky(Module):
    def __init__(self, total_class, vgg_model, pretrained_state_dict):
        super().__init__()
        self.vgg_feature_extractor = vgg_model.features

        names = [name for name, module in self.vgg_feature_extractor.named_modules() if type(module) is ReLU]
        for name in names:
            self.vgg_feature_extractor.__setattr__(name, LeakyReLU(negative_slope=.1, inplace=True))

        self.global_average_pooling = GlobalAvgPool2d()
        self.classifier_1 = Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.classifier_2 = Conv2d(1024, 1024, kernel_size=1)
        self.classifier_3 = Conv2d(1024, total_class, kernel_size=1)

        state_dict = self.state_dict()

        classifier_1_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)
        classifier_1_bias = pretrained_state_dict['classifier.0.bias']
        state_dict['classifier_1.weight'] = self._decimate(classifier_1_weight, m=[4, None, 3, 3])
        state_dict['classifier_2.bias'] = self._decimate(classifier_1_bias, m=[4])

        classifier_2_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)
        classifier_2_bias = pretrained_state_dict['classifier.3.bias']
        state_dict['classifier_2.weight'] = self._decimate(classifier_2_weight, m=[4, 4, None, None])
        state_dict['classifier_2.bias'] = self._decimate(classifier_2_bias, m=[4])

        self.load_state_dict(state_dict)

    def _decimate(self, tensor, m):
        """Decimate a tensor by factor m to convert fully connected layers to equivalent convolutional layers as used in
        SSD rewrite in PyTorch.

        This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Some
        modifications are made. All credits to @sgrvinod.
        """

        assert tensor.dim() == len(m)

        for d in range(tensor.dim()):
            if m[d] is not None:
                index = torch.arange(start=0, end=tensor.size(d), step=m[d]).long().to(next(self.parameters()).device)
                tensor = tensor.index_select(dim=d, index=index)

        return tensor

    def forward(self, x):
        x = self.vgg_feature_extractor(x)
        x = torch_fn.leaky_relu(self.classifier_1(x), negative_slope=.1, inplace=True)
        x = torch_fn.leaky_relu(self.classifier_2(x), negative_slope=.1, inplace=True)
        x = self.classifier_3(x)
        x = self.global_average_pooling(x)
        return x

    def cuda(self, device=None):
        self.vgg_feature_extractor = self.vgg_feature_extractor.cuda(device)
        self.classifier_1 = self.classifier_1.cuda(device)
        self.classifier_2 = self.classifier_2.cuda(device)
        self.classifier_3 = self.classifier_3.cuda(device)
        return super().cuda(device)

    def cpu(self):
        self.vgg_feature_extractor = self.vgg_feature_extractor.cpu()
        self.classifier_1 = self.classifier_1.cpu()
        self.classifier_2 = self.classifier_2.cpu()
        self.classifier_3 = self.classifier_3.cpu()
        return super().cpu()


class VGG16DecimatePool31(Module):
    def __init__(self, total_class, vgg_model, pretrained_state_dict):
        super().__init__()
        self.vgg_feature_extractor = vgg_model.features
        self.vgg_feature_extractor.__setattr__('30', MaxPool2d(kernel_size=3, stride=1, padding=1))

        self.global_average_pooling = GlobalAvgPool2d()
        self.classifier_1 = Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.classifier_2 = Conv2d(1024, 1024, kernel_size=1)
        self.classifier_3 = Conv2d(1024, total_class, kernel_size=1)

        state_dict = self.state_dict()

        classifier_1_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)
        classifier_1_bias = pretrained_state_dict['classifier.0.bias']
        state_dict['classifier_1.weight'] = self._decimate(classifier_1_weight, m=[4, None, 3, 3])
        state_dict['classifier_2.bias'] = self._decimate(classifier_1_bias, m=[4])

        classifier_2_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)
        classifier_2_bias = pretrained_state_dict['classifier.3.bias']
        state_dict['classifier_2.weight'] = self._decimate(classifier_2_weight, m=[4, 4, None, None])
        state_dict['classifier_2.bias'] = self._decimate(classifier_2_bias, m=[4])

        self.load_state_dict(state_dict)

    def _decimate(self, tensor, m):
        """Decimate a tensor by factor m to convert fully connected layers to equivalent convolutional layers as used in
        SSD rewrite in PyTorch.

        This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Some
        modifications are made. All credits to @sgrvinod.
        """

        assert tensor.dim() == len(m)

        for d in range(tensor.dim()):
            if m[d] is not None:
                index = torch.arange(start=0, end=tensor.size(d), step=m[d]).long().to(next(self.parameters()).device)
                tensor = tensor.index_select(dim=d, index=index)

        return tensor

    def forward(self, x):
        x = self.vgg_feature_extractor(x)
        x = torch_fn.leaky_relu(self.classifier_1(x), negative_slope=.1, inplace=True)
        x = torch_fn.leaky_relu(self.classifier_2(x), negative_slope=.1, inplace=True)
        x = self.classifier_3(x)
        x = self.global_average_pooling(x)
        return x

    def cuda(self, device=None):
        self.vgg_feature_extractor = self.vgg_feature_extractor.cuda(device)
        self.classifier_1 = self.classifier_1.cuda(device)
        self.classifier_2 = self.classifier_2.cuda(device)
        self.classifier_3 = self.classifier_3.cuda(device)
        return super().cuda(device)

    def cpu(self):
        self.vgg_feature_extractor = self.vgg_feature_extractor.cpu()
        self.classifier_1 = self.classifier_1.cpu()
        self.classifier_2 = self.classifier_2.cpu()
        self.classifier_3 = self.classifier_3.cpu()
        return super().cpu()


class VGG16DecimatePool31Ceil(Module):
    def __init__(self, total_class, vgg_model, pretrained_state_dict):
        super().__init__()
        self.vgg_feature_extractor = vgg_model.features
        self.vgg_feature_extractor.__setattr__('16', MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        self.vgg_feature_extractor.__setattr__('30', MaxPool2d(kernel_size=3, stride=1, padding=1))

        self.global_average_pooling = GlobalAvgPool2d()
        self.classifier_1 = Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.classifier_2 = Conv2d(1024, 1024, kernel_size=1)
        self.classifier_3 = Conv2d(1024, total_class, kernel_size=1)

        state_dict = self.state_dict()

        classifier_1_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)
        classifier_1_bias = pretrained_state_dict['classifier.0.bias']
        state_dict['classifier_1.weight'] = self._decimate(classifier_1_weight, m=[4, None, 3, 3])
        state_dict['classifier_2.bias'] = self._decimate(classifier_1_bias, m=[4])

        classifier_2_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)
        classifier_2_bias = pretrained_state_dict['classifier.3.bias']
        state_dict['classifier_2.weight'] = self._decimate(classifier_2_weight, m=[4, 4, None, None])
        state_dict['classifier_2.bias'] = self._decimate(classifier_2_bias, m=[4])

        self.load_state_dict(state_dict)

    def _decimate(self, tensor, m):
        """Decimate a tensor by factor m to convert fully connected layers to equivalent convolutional layers as used in
        SSD rewrite in PyTorch.

        This is implemented as shown in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. Some
        modifications are made. All credits to @sgrvinod.
        """

        assert tensor.dim() == len(m)

        for d in range(tensor.dim()):
            if m[d] is not None:
                index = torch.arange(start=0, end=tensor.size(d), step=m[d]).long().to(next(self.parameters()).device)
                tensor = tensor.index_select(dim=d, index=index)

        return tensor

    def forward(self, x):
        x = self.vgg_feature_extractor(x)
        x = torch_fn.leaky_relu(self.classifier_1(x), negative_slope=.1, inplace=True)
        x = torch_fn.leaky_relu(self.classifier_2(x), negative_slope=.1, inplace=True)
        x = self.classifier_3(x)
        x = self.global_average_pooling(x)
        return x

    def cuda(self, device=None):
        self.vgg_feature_extractor = self.vgg_feature_extractor.cuda(device)
        self.classifier_1 = self.classifier_1.cuda(device)
        self.classifier_2 = self.classifier_2.cuda(device)
        self.classifier_3 = self.classifier_3.cuda(device)
        return super().cuda(device)

    def cpu(self):
        self.vgg_feature_extractor = self.vgg_feature_extractor.cpu()
        self.classifier_1 = self.classifier_1.cpu()
        self.classifier_2 = self.classifier_2.cpu()
        self.classifier_3 = self.classifier_3.cpu()
        return super().cpu()


_structure = {
    'vgg16': vgg16(False),  # 1_300-3_300, 10_300
    'vgg16-hydra': VGG16Hydra(120, pretrained=False),  # 4_300, 5_300
    'vgg16-hydra-1conv': VGG16Hydra1Conv(120, pretrained=False),  # 8_300
    'vgg16-hydra-2conv': VGG16Hydra2Conv(120, pretrained=False),  # 9_300
    'vgg16-decimate': VGG16Decimate(120, vgg16(False), vgg16(False).state_dict()),  # 11_300-17_300
    'vgg16-decimate-leaky': VGG16DecimateLeaky(120, vgg16(False), vgg16(False).state_dict()),  # 18_300
    'vgg16-decimate-pool31': VGG16DecimatePool31(120, vgg16(False), vgg16(False).state_dict()),  # 19_300
    'vgg16-decimate-pool31-ceil': VGG16DecimatePool31Ceil(120, vgg16(False), vgg16(False).state_dict()),  # 20_300
    'd19': Module(),
    'd53': Module()
}

if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()

    parser.add_argument('-t', required=True, choices=list(_structure.keys()), help='Model type')
    parser.add_argument('-s', required=True, help='Series of the model to be used')
    parser.add_argument('-nc', action='store_false', help='Boolean flag to not use CUDA')
    parser.add_argument('-e', default='./rpdr-config-results/data/in_situ_jpgs', help='Evaluation data root')
    parser.add_argument('-ei', default='./rpdr/val_test/recog_val_test.json', help='Evaluation indices path')
    parser.add_argument('-ef', default='./rpdr/val_test/recog_val_test_classes_files.json',
                        help='Evaluation classes-files path')
    parser.add_argument('-m', default='./rpdr', help='Location of where to find this project\'s modules')
    parser.add_argument('-d', default='./', help='Location of where to save predictions')

    args = parser.parse_args()

    _type = args.t
    series = str(args.s)
    use_cuda = bool(args.nc)
    eval_root = args.e
    eval_indices = args.ei
    eval_files = args.ef
    modules_path = args.m

    # Allows for importing modules of this project
    sys.path.append(modules_path)
    from models.proposed import RPYOLOv2, RPYOLOv3
    from utils.datasets import RecognitionDataset
    from utils.metrics import infer

    _structure['vgg16'].classifier[6] = Linear(4096, 120)
    _structure['d19'] = RPYOLOv2()
    _structure['d53'] = RPYOLOv3()

    # Construct model
    model = _structure[_type]
    if _type.startswith('d'):
        model.recognizing()
    model.load_state_dict(torch.load(f'{series}_best_model.pth.tar', map_location='cpu')['state_dict'])

    if use_cuda and torch.cuda.is_available():
        model = model.cuda()

    # Construct dataset
    image_resolution = (300, 300)
    if series.endswith('224'): image_resolution = (224, 224)
    if series.endswith('256'): image_resolution = (256, 256)
    if series.endswith('448'): image_resolution = (448, 448)

    transform = Compose([
        Resize(image_resolution),
        Grayscale(num_output_channels=3),
        ToTensor(),
        Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    data = DataLoader(RecognitionDataset(
        eval_root, eval_indices, eval_files, RecognitionDataset.TEST, transform=transform), batch_size=1,
        shuffle=False)

    # Inference
    outputs, predictions, ground_truths = infer(model, data, use_cuda)

    # Save inference results
    torch.save({
        'outputs': outputs,
        'predictions': predictions,
        'ground_truths': ground_truths
    }, f'{series}_results.pth.tar')
