import conformer
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single
import matplotlib.pyplot as plt
from conformer import ConformerBlock


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GaborConv1d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, device="cpu", stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'):
        kernel_size = _single(kernel_size)  # 确保kernel_size是元组
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)

        super(GaborConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode)

        # 初始化Gabor滤波器参数
        self.freq = nn.Parameter(
            (np.pi / 2) * 1.41 ** (-torch.randint(0, 5, (out_channels, in_channels)).float()))
        self.psi = nn.Parameter(np.pi * torch.rand(out_channels, in_channels))
        self.sigma = nn.Parameter(np.pi / self.freq)
        self.t0 = torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0]
        self.device = device

        # 注册buffer来存储时间步长
        # self.register_buffer('t', torch.linspace(-self.t0 + 1, self.t0, self.kernel_size[0]))

    def forward(self, input_signal):
        t = torch.linspace(-self.t0 + 1, self.t0, self.kernel_size[0]).to(self.device)
        # weight = torch.empty(self.weight.shape, requires_grad=False).to(self.device)
        # t = self.t
        weight = torch.empty(self.weight.shape).to(self.device)

        for i in range(self.out_channels):
            for j in range(self.in_channels):
                sigma = self.sigma[i, j].expand_as(t)
                freq = self.freq[i, j].expand_as(t)
                psi = self.psi[i, j].expand_as(t)

                # 计算Gabor滤波器
                g = torch.exp(-0.5 * (t ** 2 / (sigma + 1e-3) ** 2))
                g = g * torch.cos(freq * t + psi)
                g = g / (sigma * np.sqrt(2 * np.pi) + 1e-3)  # 归一化
                weight[i, j] = g
                # self.weight.data[i, j] = g

        return F.conv1d(input_signal, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class GaborNN1D(nn.Module):
    def __init__(self, sample_length=99):
        super(GaborNN1D, self).__init__()
        self.name = 'GaborNN1D'
        self.sample_length = sample_length


        # 计算序列长度变化
        # 输入: (batch_size, 1, 99)
        self.g0 = GaborConv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3, device=device)  # 输出: (batch_size, 32, 99)
        self.bn1 = nn.BatchNorm1d(32)
        self.m1 = nn.MaxPool1d(kernel_size=1)  # 输出: (batch_size, 32, 99)

        self.conv1 = nn.Conv1d(32, 64, kernel_size=5, padding=2)  # 输出: (batch_size, 64, 99)
        self.bn2 = nn.BatchNorm1d(64)
        self.m2 = nn.MaxPool1d(kernel_size=1)  # 输出: (batch_size, 64, 99)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)  # 输出: (batch_size, 128, 99)
        self.bn3 = nn.BatchNorm1d(128)
        self.m3 = nn.MaxPool1d(kernel_size=1)  # 输出: (batch_size, 128, 99)


    def forward(self, x):
        # Gabor卷积层
        x = self.g0(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.m1(x)

        # 第一个卷积层
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.m2(x)

        # 第二个卷积层
        x = self.conv2(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.m3(x)

        return x



class gaborformer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 dim_head=64,
                 heads=8,
                 ff_mult=4,
                 conv_expansion_factor=2,
                 conv_kernel_size=31,
                 attn_dropout=0.,
                 ff_dropout=0.,
                 conv_dropout=0.,
                 conv_causal=False,
                 num_classes=2,
                 sample_length=99):
        super(gaborformer, self).__init__()
        self.gabor = GaborNN1D(sample_length=sample_length)  # 输出: (batch_size, 128, 99)
        self.dim = dim
        self.layers = nn.ModuleList([])
        self.sample_length = sample_length
        # 计算全连接层的输入特征数
        self.flatten_features = self.dim * self.sample_length

        for _ in range(depth):
            self.layers.append(ConformerBlock(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                ff_mult=ff_mult,
                conv_expansion_factor=conv_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                conv_causal=conv_causal,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                conv_dropout = conv_dropout

            ))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.dim * self.sample_length, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.gabor(x)
        c = x
        c1 = c.view(-1, self.flatten_features)
        x = x.permute(0, 2, 1)  # (batch_size, 99, 128)
        for block in self.layers:
            x = block(x)

        # 展平
        x = x.view(-1, 128*self.sample_length)
        c2 = x
        x = self.fc(x)

        return x,c,c1,c2


if __name__ == '__main__':
    batch_size = 2
    sample_length = 988
    dim=128
    model = gaborformer(
        dim=dim,
        depth=3,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.1,
        ff_dropout=0.1,
        conv_dropout=0.1,
        conv_causal=False,
        sample_length=sample_length,
        num_classes= 2
    ).to(device)
    x = torch.randn(batch_size, 1, sample_length).to(device)
    output = model(x)

    # Print the output shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
