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
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(self.in_channels, ch, kernel_size=3, padding=1),
                nn.BatchceNorm2d(ch),
                nn.ReLU(inplace=True),
                ConvBlock(ch, ch, ch, self.drop_prob),
            )
        )
        for _ in range(self.num_pool_layers - 1):
            self.downsample_layers.append(ConvBlock(ch, ch * 2, ch * 2, self.drop_prob))
            ch *= 2

        self.conv = ConvBlock(ch, ch * 2, ch, self.drop_prob)

        # configure upsampling layers
        self.up_transpose_conv = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        for _ in range(self.num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch, ch))
            self.upsample_layers.append(ConvBlock(ch * 2, ch, ch // 2, self.drop_prob))
            ch //= 2
        self.up_transpose_conv.append(TransposeConvBlock(ch, ch))
        self.upsample_layers.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, ch, self.drop_prob),
                nn.Conv2d(ch, self.in_channels, kernel_size=1, stride=1),
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1),
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

    def __init__(
        self,
        in_channels: int,
        intra_channels: int,
        out_channels: int,
        drop_prob: float
    ):
        """
        Args:
            in_channels: Number of channels in the input.
            intra_channels: Number of channels in the intra-output.
            out_channels: Number of channels in the output.
            drop_prob: Dropout probability.
        """

        super().__init__()

        self.in_channels = in_channels
        self.intra_channels = intra_channels
        self.out_channels = out_channels
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.intra_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.InstanceNorm2d(self.intra_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(self.drop_prob),
            nn.Conv2d(
                self.intra_channels,
                self.out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.InstanceNorm2d(self.out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(self.drop_prob),
            AttenLayer(self.out_channels),
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


class AttenLayer(nn.Module):
    """
    A Attention Layer that maxes out the features from Frequency-Attention Layer
    and Channel-Attention Layer.
    """

    def __init__(self, in_channels: int):
        """
        Args:
            in_channels: Number of channels in the input.
        """

        super().__init__()

        self.in_channels = in_channels

        self.frequency_atten = FrequencyAttenLayer(self.in_channels)
        self.channel_atten = ChannelAttenLayer(self.in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape `(N, in_channels, H, W)`.
        Returns:
            Output tensor of shape `(N, in_channels, H, W)`.
        """

        return torch.maximum(self.frequency_atten(x), self.channel_atten(x))


class FrequencyAttenLayer(nn.Module):
    """
    A Frequency-Attention Layer that represents frequency-attention weighted features
    by linearly combining representation of all channels for a spatial frequency.
    """

    def __init__(self, in_channels: int):
        """
        Args:
            in_channels: Number of channels in the input.
        """

        super().__init__()

        self.in_channels = in_channels

        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 1, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape `(N, in_channels, H, W)`.
        Returns:
            Output tensor of shape `(N, in_channels, H, W)`.
        """

        return torch.mul(self.layers(x), x)


class ChannelAttenLayer(nn.Module):
    """
    A Channel-Attention Layer that represents channel-attention weighted features.
    """

    def __init__(self, in_channels: int):
        """
        Args:
            in_channels: Number of channels in the input.
        """

        super().__init__()

        self.in_channels = in_channels

        self.layers = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape `(N, in_channels, H, W)`.
        Returns:
            Output tensor of shape `(N, in_channels, H, W)`.
        """

        output = self.layers(torch.norm(x, p=1, dim=(2,3)))
        return torch.mul(output.reshape(-1, self.in_channels, 1, 1), x)
