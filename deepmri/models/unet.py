import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """
    PyTorch implementation of a U-Net.
    Refer to https://github.com/facebookresearch/fastMRI/blob/master/fastmri/models/unet.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int = 64,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0
    ):
        """
        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels in the output.
            channels: Number of output channels of the first downsampling layer.
            num_pool_layers: Number of downsampling and upsampling layers.
            drop_prob: Dropout probability.
        """

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        ch = self.channels

        # configure downsampling layers
        self.downsample_layers = nn.ModuleList([ConvBlock(self.in_channels, ch, self.drop_prob)])
        for _ in range(self.num_pool_layers - 1):
            self.downsample_layers.append(ConvBlock(ch, ch * 2, self.drop_prob))
            ch *= 2

        self.conv = ConvBlock(ch, ch * 2, self.drop_prob)

        # configure upsampling layers
        self.up_transpose_conv = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        for _ in range(self.num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.upsample_layers.append(ConvBlock(ch * 2, ch, self.drop_prob))
            ch //= 2
        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.upsample_layers.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, self.drop_prob),
                nn.Conv2d(ch, self.out_channels, kernel_size=1, stride=1),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape `(N, in_channels, H, W)`.
        Returns:
            Output tensor of shape `(N, out_channels, H, W)`.
        """

        output = x
        stack = []

        # apply downsampling layers
        for layer in self.downsample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply upsampling layers
        for up_conv, layer in zip(self.up_transpose_conv, self.upsample_layers):
            output = up_conv(output)
            downsample_output = stack.pop()
            # reflect pad on the right/botton if needed to handle odd input dimensions
            pad = [0, 0, 0, 0]
            if output.shape[-1] != downsample_output.shape[-1]:
                pad[1] = 1  # padding right
            if output.shape[-2] != downsample_output.shape[-2]:
                pad[3] = 1  # padding bottom
            if torch.sum(torch.tensor(pad)) != 0:
                output = F.pad(output, tuple(pad), mode='reflect')
            # apply skip-connection
            output = torch.cat((output, downsample_output), dim=1)
            output = layer(output)

        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers with same padding
    each followed by instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_channels: int, out_channels: int, drop_prob: float):
        """
        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels in the output.
            drop_prob: Dropout probability.
        """

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.InstanceNorm2d(self.out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(self.drop_prob),
            nn.Conv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.InstanceNorm2d(self.out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(self.drop_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape `(N, in_channels, H, W)`.
        Returns:
            Output tensor of shape `(N, out_channels, H, W)`.
        """

        return self.layers(x)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one transposed convolution
    layer followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels in the output.
        """

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                self.in_channels,
                self.out_channels,
                kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(self.out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape `(N, in_channels, H, W)`.
        Returns:
            Output tensor of shape `(N, out_channels, H*2, W*2)`.
        """

        return self.layers(x)
