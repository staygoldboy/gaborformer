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




class GaborNN1D(nn.Module):
    def __init__(self, sample_length=99):
        super(GaborNN1D, self).__init__()
        self.name = 'GaborNN1D'
        self.sample_length = sample_length


        # 计算序列长度变化
        # 输入: (batch_size, 1, 99)
        self.g0 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3, device=device)  # 输出: (batch_size, 32, 99)
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



class wo_gabor(nn.Module):
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
        super(wo_gabor, self).__init__()
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

        return x


if __name__ == '__main__':
    batch_size = 2
    sample_length = 988
    dim=128
    model = wo_gabor(
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