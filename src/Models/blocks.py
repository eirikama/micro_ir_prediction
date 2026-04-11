import torch
import torch.nn.functional as F
from torch import nn


class InputNorm(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=True, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class AugmentedConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dk: int,
        dv: int,
        Nh: int,
        padding: int = 0,
        shape: int = 0,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert (
            self.dk % self.Nh == 0
        ), "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert (
            self.dv % self.Nh == 0
        ), "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv1d(
            self.in_channels,
            self.out_channels - self.dv,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        self.qkv_conv = nn.Conv1d(
            self.in_channels,
            2 * self.dk + self.dv,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        self.attn_out = nn.Conv1d(self.dv, self.dv, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv_out(x)
        batch, _, width = conv_out.size()

        flat_q, flat_k, flat_v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)

        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        weights = F.softmax(logits, dim=-1)

        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, width))

        attn_out = self.combine_heads_1d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(
        self, x: torch.Tensor, dk: int, dv: int, Nh: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        qkv = self.qkv_conv(x)
        N, _, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_1d(q, Nh)
        k = self.split_heads_1d(k, Nh)
        v = self.split_heads_1d(v, Nh)

        dkh = dk // Nh
        q = q * dkh**-0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, W))

        return flat_q, flat_k, flat_v

    def split_heads_1d(self, x: torch.Tensor, Nh: int) -> torch.Tensor:
        batch, channels, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_1d(self, x: torch.Tensor) -> torch.Tensor:
        batch, Nh, dv, W = x.size()
        ret_shape = (batch, Nh * dv, W)
        return torch.reshape(x, ret_shape)


class SpectralBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int, dk: int = 16, dv: int = 8, Nh: int = 4
    ) -> None:
        super().__init__()
        self.conv = AugmentedConv(in_ch, out_ch, kernel_size, dk=dk, dv=dv, Nh=Nh)
        self.bn = nn.BatchNorm1d(out_ch)
        self.leaky = nn.LeakyReLU(0.1)
        # 1x1 conv to match dimensions for the residual add
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky(self.bn(self.conv(x)) + self.shortcut(x))
