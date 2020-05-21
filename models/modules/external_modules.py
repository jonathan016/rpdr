from torch.nn import Module
from torch.nn.functional import avg_pool2d as torch_avg_pool2d
from torch import Tensor


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
