from torch import cat as torch_concat_tensor
from torch.nn import Module, Conv2d, Sequential, BatchNorm2d, LeakyReLU

from .external_modules import Reorg


class Passthrough(Module):
    def __init__(self, inner_conv_in_channels=512):
        super().__init__()
        self.inner_conv = Conv2d(in_channels=inner_conv_in_channels, out_channels=64, kernel_size=1, stride=1,
                                 padding=0)
        self.reorg = Reorg(stride=2)

    def forward(self, passthroughed_input, normal_flow_input):
        output = self.inner_conv(passthroughed_input)
        output = self.reorg(output)

        tensor = torch_concat_tensor((output, normal_flow_input), dim=1)
        return tensor


class DetectionAdditionalLayers(Module):
    def __init__(self, passthrough_in_channels=512):
        super().__init__()

        self.layer_19 = Sequential()
        self.layer_19.add_module(f'Conv19', Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1))
        self.layer_19.add_module(f'BatchNorm19', BatchNorm2d(num_features=1024))
        self.layer_19.add_module(f'Activation19', LeakyReLU(negative_slope=.1, inplace=True))

        self.layer_20 = Sequential()
        self.layer_20.add_module(f'Conv20', Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1))
        self.layer_20.add_module(f'BatchNorm20', BatchNorm2d(num_features=1024))
        self.layer_20.add_module(f'Activation20', LeakyReLU(negative_slope=.1, inplace=True))

        self.passthrough: Passthrough = Passthrough(passthrough_in_channels)

        self.layer_21 = Sequential()
        passthrough_out_channels = self.passthrough.inner_conv.out_channels * int(pow(self.passthrough.reorg.stride, 2))
        in_channels = self.layer_20.Conv20.out_channels + passthrough_out_channels
        self.layer_21.add_module(f'Conv21',
                                 Conv2d(in_channels=in_channels, out_channels=1024, kernel_size=3, padding=1))
        self.layer_21.add_module(f'BatchNorm21', BatchNorm2d(num_features=1024))
        self.layer_21.add_module(f'Activation21', LeakyReLU(negative_slope=.1, inplace=True))

    def forward(self, passthroughed_input, to_concat):
        to_concat = self.layer_19(to_concat)
        to_concat = self.layer_20(to_concat)
        passthrough = self.passthrough(passthroughed_input, to_concat)
        output = self.layer_21(passthrough)

        return output
