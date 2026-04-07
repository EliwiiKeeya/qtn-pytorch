# -*- encoding: utf-8 -*-
# @File			: model.py
# @Date			: 2026/04/04 11:56:14
# @Author		: Eliwii_Keeya
import torch
from torch import nn
from core_qnn.quaternion_layers import QuaternionConv


class Patching(nn.Module):
    """
    Patch Embedding module for Quanternion Transformer Network.
    - This module takes an input image and divides it into non-overlapping patches.
    - Each patch is then flattened and projected into a higher-dimensional embedding space.
    - The output is a sequence of patch embeddings that can be fed into the transformer encoder.
    """

    def __init__(self, patch_size: int) -> None:
        """
        Args:
            patch_size (int): The size of each patch (e.g., 16 for 16x16 patches).
        """
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, C, H, W]
        Returns:
            torch.Tensor: [M, C, K, K],
                where M = B * N,
                N = (H // K) * (W // K),
                K = patch_size
        """
        K = self.patch_size
        B, C, H, W = x.shape

        # Check input shape
        if not H % K == 0 and W % K == 0:
            raise ValueError(
                f"Input spatial dimensions must be divisible by the patch size."
                f"Got H={H}, W={W}, patch_size={K}."
            )

        nH = H // K
        nW = W // K
        N = nH * nW

        x = x.unfold(2, K, K).unfold(3, K, K)  # [B, C, nH, nW, K, K]
        x = x.permute(0, 2, 3, 1, 4, 5)       # [B, nH, nW, C, K, K]
        x = x.reshape(B * N, C, K, K)          # [M, C, K, K]
        return x


class BandAdaptiveSelection(nn.Module):
    """
    Band Adaptive Selection Module for Quanternion Transformer Network.
    - This module takes the patch embeddings and performs adaptive selection of the most informative bands.
    - It computes attention scores for each band and selects the top-k bands based on these scores.
    - The output consists of the real part of the patch embeddings and the selected bands for the imaginary part.
    - This module is designed to enhance the representational capacity of the model by focusing on the most relevant features in the patch embeddings.
    - The output is structured to be compatible with quaternion operations in subsequent layers of the transformer encoder.
    - The real part (r) is obtained by applying a convolutional layer to the input patch embeddings.
    - The attention scores (att) are computed by applying global average pooling followed by a convolutional layer and a sigmoid activation.
    - The top-k bands are selected based on the attention scores, and the corresponding features are gathered to form the imaginary part (i, j, k) of the output.
    - The output tensors are returned in a contiguous format to ensure efficient memory access in subsequent operations.
    - This module is crucial for enabling the model to adaptively focus on the most informative features in the patch embeddings, thereby improving the overall performance of the Quanternion Transformer Network.
    - The design of this module allows for flexibility in the number of bands selected and can be easily integrated into the transformer architecture for enhanced feature representation
    """

    def __init__(self, patch_size: int, channels: int):
        """
        Args:
            patch_size (int): The size of each patch.
            in_channels (int): The number of input channels.
        """
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels

        self.conv1 = nn.Conv2d(
            in_channels=self.channels,
            out_channels=1,
            kernel_size=1,
        )
        self.pool = nn.AvgPool2d(
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
            padding_mode='circular'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [M, C, K, K]
        Returns:
            torch.Tensor: [M, 4C, K, K],
                where the first C channels correspond to the real part (r),
                and the remaining 3C channels correspond to the imaginary part (i, j, k).
        """
        # Check input shape
        M, _, H, W = x.shape
        if H != self.patch_size or W != self.patch_size:
            raise ValueError(
                f"Input tensor must have shape [M, {self.channels},"
                f"{self.patch_size}, {self.patch_size}]."
                f"Got {x.shape}."
            )

        # Real Part
        r = self.conv1(x)

        # Imaginary Part
        gap = self.pool(x).view(M, 1, self.channels)
        att = self.conv2(gap).squeeze(1)
        att = self.sigmoid(att)

        _, mask = att.topk(3)
        mask = mask.view(M, 3, 1, 1)
        mask = mask.expand(M, 3, self.patch_size, self.patch_size)

        ijk = x.gather(dim=1, index=mask)
        x = torch.cat([r, ijk], dim=1)

        return x


class QuaternionSelfAttention(nn.Module):
    """
    Quaternion Self-Attention Module for Quanternion Transformer Network.
    - This module implements the self-attention mechanism in the quaternion domain.
    - It takes the input tensor, applies quaternion convolutions to compute the attention scores, and returns the attended output.
    - The attention scores are computed by applying a series of quaternion convolutions and non-linear activations to the input tensor.
    - The output is obtained by multiplying the input tensor with the computed attention scores, allowing the model to focus on the most relevant features in the input.
    - This module is designed to enhance the representational capacity of the model by enabling it to capture complex relationships between features in the quaternion domain.
    - The use of quaternion convolutions allows for more efficient and expressive feature representations, which can lead to improved performance in tasks such as image classification, object detection, and other computer vision applications.
    """

    def __init__(self, channels: int) -> None:
        """
        Args:
            channels (int): The number of channels.
            patch_size (int): The size of each patch.
        """
        super().__init__()
        self.channels = channels

        self.qconv1 = QuaternionConv(
            in_channels=self.channels,
            out_channels=self.channels,
            stride=1,
            kernel_size=3,
            padding=1
        )

        self.qconv2 = QuaternionConv(
            in_channels=self.channels * 2,
            out_channels=self.channels * 2,
            stride=1,
            kernel_size=3,
            padding=1
        )

        self.qconv3 = QuaternionConv(
            in_channels=self.channels * 2,
            out_channels=self.channels,
            stride=1,
            kernel_size=1,
            padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [M, C, K, K]
        Returns:
            torch.Tensor: [M, C, K, K]
        """
        y = self.qconv1(x)
        yy = torch.cat([y, y], dim=1)
        att = self.qconv3(self.qconv2(yy))
        x = x * att
        return x


class QSABlock(nn.Module):
    """
    Quaternion Self-Attention Block for Quanternion Transformer Network.
    - This block consists of a layer normalization, a convolutional layer, a GELU activation, a quaternion self-attention module, and another convolutional layer.
    - The input tensor is first normalized and passed through the convolutional layer to extract features.
    - The output is then activated using the GELU function before being fed into the quaternion self-attention module to compute the attention scores.
    - Finally, the output from the attention module is normalized again and passed through another convolutional layer to produce the final output of the block.
    - This block is designed to capture complex relationships between features in the quaternion domain while maintaining efficient computation through the use of convolutional layers and non-linear activations.
    - The combination of these components allows for enhanced feature representation and improved performance in various computer vision tasks when integrated into the Quanternion Transformer Network architecture.
    """

    def __init__(self, channels: int, patch_size: int) -> None:
        """
        Args:
            channels (int): The number of channels.
            patch_size (int): The size of each patch.
        """
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size

        self.conv1 = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.ln1 = nn.LayerNorm(
            [self.channels, self.patch_size, self.patch_size])
        self.gelu = nn.GELU()
        self.qsa = QuaternionSelfAttention(self.channels)
        self.conv2 = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.ln2 = nn.LayerNorm(
            [self.channels, self.patch_size, self.patch_size])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [M, C, K, K]
        Returns:
            torch.Tensor: [M, C, K, K]
        """
        x = self.ln1(x)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.qsa(x)
        x = self.ln2(x)
        x = self.conv2(x)
        return x


class MLPBlock(nn.Module):
    """
    MLP Block for Quanternion Transformer Network.
    - This block consists of two layer normalization layers and two convolutional layers.
    - The input tensor is first normalized and passed through the first convolutional layer to extract features.
    - The output is then normalized again and passed through the second convolutional layer to produce the final output of the block.
    - This block is designed to enhance the representational capacity of the model by allowing it to capture complex relationships between features in the quaternion domain through the use of convolutional layers and non-linear activations.
    - The combination of these components allows for improved performance in various computer vision tasks when integrated into
    """

    def __init__(self, channels: int, patch_size: int) -> None:
        """
        Args:
            channels (int): The number of channels.
            patch_size (int): The size of each patch.
        """
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size

        self.ln1 = nn.LayerNorm(
            [self.channels, self.patch_size, self.patch_size])
        self.conv1 = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.ln2 = nn.LayerNorm(
            [self.channels, self.patch_size, self.patch_size])
        self.conv2 = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [M, C, K, K]
        Returns:
            torch.Tensor: [M, C, K, K]
        """
        x = self.ln1(x)
        x = self.conv1(x)
        x = self.ln2(x)
        x = self.conv2(x)
        return x


class QuaternionTransformer(nn.Module):
    """
    Quaternion Transformer for processing quaternion data.
    - This module consists of a series of Quaternion Self-Attention blocks and MLP blocks.
    - The input tensor is passed through each block sequentially, with residual connections to enhance feature representation.
    - The number of blocks and the configuration of each block can be customized to suit different tasks and datasets.
    - This module is designed to capture complex relationships between features in the quaternion domain while maintaining efficient computation through the use of convolutional layers and non-linear activations.
    - The combination of these components allows for improved performance in various computer vision tasks when integrated into the Quanternion
    """

    def __init__(self, channels: int, patch_size: int, L: int) -> None:
        """
        Args:
            channels (int): The number of channels.
            patch_size (int): The size of each patch.
            L (int): The number of blocks.
        """
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.L = L

        self.qsa = nn.ModuleList(
            [QSABlock(self.channels, self.patch_size)for _ in range(self.L)])
        self.mlp = nn.ModuleList(
            [MLPBlock(self.channels, self.patch_size)for _ in range(self.L)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [M, C, K, K]
        Returns:
            torch.Tensor: [M, C, K, K]
        """
        for qsa_block, mlp_block in zip(self.qsa, self.mlp):
            x = x + qsa_block(x)
            x = x + mlp_block(x)
        return x


class QuaternionTransformerNetworkTiny(nn.Module):
    """
    Quaternion Transformer Network Tiny (QTN-Tiny) for image classification.
    - This module consists of a series of components including patching, band adaptive selection, downsampling, and quaternion transformer blocks.
    - The input image is first processed through the patching module to create patch embeddings, followed by the band adaptive selection module to select the most informative features.
    - The downsampling modules reduce the spatial dimensions of the feature maps while increasing the channel dimensions, allowing for more efficient processing in the subsequent quaternion transformer blocks.
    - The quaternion transformer blocks capture complex relationships between features in the quaternion domain, enhancing the representational capacity of the model.
    - The final output is a feature representation that can be used for various downstream tasks such as image classification, object detection, and other computer vision applications.
    """
    
    def __init__(self, patch_size: int, channels: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels

        self.patching = Patching(patch_size=self.patch_size)

        self.basm = BandAdaptiveSelection(
            patch_size=self.patch_size, channels=self.channels)

        self.downsample1 = torch.nn.PixelUnshuffle(2)

        self.qtn1 = QuaternionTransformer(
            channels=16, patch_size=self.patch_size // 2, L=3)

        self.proj1 = nn.Sequential(
            nn.LayerNorm([16, self.patch_size // 2, self.patch_size // 2]),
            nn.Conv2d(16, 64, 1)
        )
        self.qtn2 = QuaternionTransformer(
            channels=64, patch_size=self.patch_size // 2, L=3)

        self.downsample2 = torch.nn.Sequential(
            torch.nn.LayerNorm(
                [64, self.patch_size // 2, self.patch_size // 2]),
            torch.nn.PixelUnshuffle(2),
            torch.nn.Conv2d(256, 128, 1)
        )

        self.qtn3 = QuaternionTransformer(
            channels=128, patch_size=self.patch_size // 4, L=5)

        self.proj2 = nn.Sequential(
            nn.LayerNorm([128, self.patch_size // 4, self.patch_size // 4]),
            nn.Conv2d(128, 256, 1)
        )
        self.qtn4 = QuaternionTransformer(
            channels=256, patch_size=self.patch_size // 4, L=2)

        self.ln = nn.LayerNorm(
            [256, self.patch_size // 4, self.patch_size // 4])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, C, H, W]
        Returns:
            torch.Tensor: [M, C, K, K]
        """
        x = self.patching(x)
        x = self.basm(x)
        x = self.downsample1(x)
        x = self.qtn1(x)
        x = self.proj1(x)
        x = self.qtn2(x)
        x = self.downsample2(x)
        x = self.qtn3(x)
        x = self.proj2(x)
        x = self.qtn4(x)
        x = self.ln(x)
        return x
