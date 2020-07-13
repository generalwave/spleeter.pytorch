import torch
from torch import nn
from torch.nn import functional
from math import floor, ceil


class Conv2dKeras(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding='same', dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv2dKeras, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            0, dilation, groups,
            bias, padding_mode)
        self.keras_mode = padding

    def _padding_size(self, size, idx):
        output = (size[idx] + self.stride[idx] - 1) // self.stride[idx]
        padding = (output - 1) * self.stride[idx] + (self.kernel_size[idx] - 1) * self.dilation[idx] + 1 - size[idx]
        padding = max(0, padding)
        return padding

    def forward(self, x):
        if self.keras_mode == 'same':
            size = x.shape[2:]
            row = self._padding_size(size, 0)
            col = self._padding_size(size, 1)
            x = functional.pad(x, [floor(col / 2), ceil(col / 2), floor(row / 2), ceil(row / 2)])

        return super(Conv2dKeras, self).forward(x)


# 简单支持
class ConvTranspose2dKeras(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros'):
        assert output_padding == 1
        super(ConvTranspose2dKeras, self).__init__(
            in_channels, out_channels, kernel_size, 1,
            dilation * (kernel_size - 1), 0, groups, bias,
            dilation, padding_mode)

        self.keras_kernel_size = kernel_size
        self.keras_stride = stride
        self.keras_padding = padding
        self.keras_output_padding = output_padding
        self.keras_dilation = dilation

    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        assert output_size is None
        output_padding = self.keras_output_padding

        padding = self.keras_dilation * (self.keras_kernel_size - 1) - self.keras_padding

        b, c, h, w = x.shape
        h = h + (h - 1) * (self.keras_stride - 1) + 2 * padding + output_padding
        w = w + (w - 1) * (self.keras_stride - 1) + 2 * padding + output_padding
        hs, he = padding + output_padding, h - padding
        ws, we = padding + output_padding, w - padding
        newx = torch.zeros((b, c, h, w), dtype=x.dtype, device=x.device)
        newx[:, :, hs:he:self.keras_stride, ws:we:self.keras_stride] = x

        return functional.conv_transpose2d(
            newx, self.weight, self.bias, self.stride, self.padding,
            0, self.groups, self.dilation)
