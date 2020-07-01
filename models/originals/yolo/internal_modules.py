from torch import cat as torch_concat_tensor, Tensor
from torch.nn import Module, Conv2d, Sequential, BatchNorm2d, LeakyReLU

from .external_modules import Reorg, _bbox_ious, _bbox_iou


def intersection_over_union(many, first, second, is_corner_coordinates):
    """Handles intersection over union (IoU) calculation for YOLO models.

    This method hides the compuutation of IoU for either many predictions and ground truths or one prediction and
    ground truth.
    """
    if many:
        return _bbox_ious(first, second, is_corner_coordinates)
    return _bbox_iou(first, second, is_corner_coordinates)


class ConcatenatingRoute(Module):
    """Implements ``[route]`` layer from YOLO config files for two layers.

    The ``[route]`` layer from YOLO config files can have two different ``layers`` specification. For only one number
    in ``layers``, it can be assumed as a skip connection. For two numbers in ``layers``, this means the output of these
    layer indexes will be concatenated by the specified order of numbers in ``layers``, then will be used for further
    calculations.
    """

    def __init__(self):
        super().__init__()

    def forward(self, first, second):
        return torch_concat_tensor((first, second), dim=1)


class Passthrough(Module):
    """Implements passthrough operation as specified in YOLOv2 config file.

    The passthrough layer on the publication seems to be similar to residual layer in ResNet, but it is quite
    different as shown in the config file, where there are convolution, reorg, and concatenating feature map
    operations. This class encapsulates such operation.
    """

    def __init__(self, inner_conv_in_channels=512):
        super().__init__()
        self.inner_conv = Conv2d(in_channels=inner_conv_in_channels, out_channels=64, kernel_size=1, stride=1,
                                 padding=0)
        self.reorg = Reorg(stride=2)
        self.route = ConcatenatingRoute()

    def forward(self, passthroughed_input, normal_flow_input):
        output = self.inner_conv(passthroughed_input)
        output = self.reorg(output)
        output = self.route(output, normal_flow_input)
        return output


class DetectionAdditionalLayers(Module):
    """Encapsulates the additional detection layers in YOLOv2.

    YOLOv2 requires adding several more layers for better features when detecting objects. This class encapsulates
    those layers, which include three convolutional layers (layer 19-21) with one passthrough layer between layer 20
    and 21. Note that this class do not  encapsulate the predictor. The predictor is selected at ``YOLOv2`` class.

    As specified in the publication, detecting with the layers encapsulated within this class requires two input: the
    passthroughed input (from layer 13) and the normal flow input (from layer 18).
    """

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


class TwoConvsBeforeResidual(Module):
    """Encapsulates two convolutional layers before residual operation/skip connection in YOLOv3.

    In YOLOv3 config file, commonly found structure is something like this::

            [convolutional]
            stride=1
            ...
            [convolutional]
            stride=1
            ...
            [shortcut]
            from=-3

    To ease model structure and forward pass, such architecture is encapsulated to this class, hence the name
    ``TwoConvsBeforeResidual`` to denote what it actually means. Note that residual and shortcut is the same in this
    class' implementation. The flow of data can be seen as below: ::

            input - Conv1 - Conv2 - (+) - output
                 \__________________/
                      (residual)

    where the ``(+)`` denotes add operation: adding the values without creating new axis or dimension.
    """

    def __init__(self, index=0, in_channels=None, out_channels=None):
        super().__init__()

        if in_channels is None:
            raise ValueError('in_channels must be specified')
        if out_channels is None:
            raise ValueError('out_channels must be specified')

        self.convs = Sequential()

        self.convs.add_module(f'Conv{index}',
                              Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                     padding=0))
        self.convs.add_module(f'BatchNorm{index}', BatchNorm2d(num_features=out_channels))
        self.convs.add_module(f'Activation{index}', LeakyReLU(negative_slope=.1, inplace=True))
        self.convs.add_module(f'Conv{index + 1}',
                              Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=3, stride=1,
                                     padding=1))
        self.convs.add_module(f'BatchNorm{index + 1}', BatchNorm2d(num_features=in_channels))
        self.convs.add_module(f'Activation{index + 1}', LeakyReLU(negative_slope=.1, inplace=True))

    def forward(self, input: Tensor):
        residual = input
        output = residual + self.convs(input)
        return output


class YOLOv3Block(Module):
    """Encapsulates typical YOLOv3 block.

    In YOLOv3 config file, for each downsampling block, a similar pattern is found such as below::

            [convolutional]
            stride=2
            ...
            [convolutional]
            stride=1
            ...
            [convolutional]
            stride=1
            ...
            [shortcut]
            from=-3

    To ease model structure and forward pass, such architecture is encapsulated to this class, hence the name
    ``YOLOv3Block`` as a symbol of this class' structure representation. For each convolutional with ``stride=1`` and
    following ``[shortcut]``, this kind of structure is encapsulated in ``TwoConvsBeforeResidual`` class.
    """

    def __init__(self, index=0, in_channels=None, out_channels=None, convs_residual_count=0, initial_block_stride=2):
        super().__init__()

        if in_channels is None:
            raise ValueError('in_channels must be specified')
        if out_channels is None:
            raise ValueError('out_channels must be specified')

        self.index = index
        self._create_body(in_channels, index, out_channels, convs_residual_count, initial_block_stride)

    def _create_body(self, in_channels, index, out_channels, stacked_conv_residual_count, initial_block_stride):
        self.initial_block = Sequential()
        self.initial_block.add_module(f'Conv{index}',
                                      Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                             stride=initial_block_stride, padding=1))
        self.initial_block.add_module(f'BatchNorm{index}', BatchNorm2d(num_features=out_channels))
        self.initial_block.add_module(f'Activation{index}', LeakyReLU(negative_slope=.1, inplace=True))

        modules = []
        for i in range(1, stacked_conv_residual_count + 1):
            modules.append(TwoConvsBeforeResidual(index=index + i, in_channels=out_channels, out_channels=in_channels))
            index += 1
        self.following_blocks = Sequential(*modules)

    def forward(self, input: Tensor):
        output = self.initial_block(input)

        for block in self.following_blocks:
            output = block(output)

        return output
