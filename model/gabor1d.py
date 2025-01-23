import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single
import matplotlib.pyplot as plt

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

    def forward(self, input_signal):
        t = torch.linspace(-self.t0 + 1, self.t0, self.kernel_size[0]).to(self.device)
        weight = torch.empty(self.weight.shape, requires_grad=False).to(self.device)

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
                self.weight.data[i, j] = g

        return F.conv1d(input_signal, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class GaborNN1D(nn.Module):
    def __init__(self, num_classes=2, sample_length=110):
        super(GaborNN1D, self).__init__()
        self.name = 'GaborNN1D'
        self.sample_length = sample_length


        # 计算序列长度变化
        # 输入: (batch_size, 1, 99)
        self.g0 = GaborConv1d(in_channels=1, out_channels=32, kernel_size=7,
                              padding=3, device=device)  # 输出: (batch_size, 32, 99)
        self.bn1 = nn.BatchNorm1d(32)
        self.m1 = nn.MaxPool1d(kernel_size=1)  # 输出: (batch_size, 32, 99)

        self.conv1 = nn.Conv1d(32, 64, kernel_size=5, padding=2)  # 输出: (batch_size, 64, 99)
        self.bn2 = nn.BatchNorm1d(64)
        self.m2 = nn.MaxPool1d(kernel_size=1)  # 输出: (batch_size, 64, 99)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)  # 输出: (batch_size, 128, 99)
        self.bn3 = nn.BatchNorm1d(128)
        self.m3 = nn.MaxPool1d(kernel_size=1)  # 输出: (batch_size, 128, 99)

        # 计算全连接层的输入特征数
        self.flatten_features = 128 * self.sample_length

        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(self.flatten_features, 256)
        self.fc2 = nn.Linear(256, num_classes)

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
        c1 = x

        # 展平
        x = x.view(-1, self.flatten_features)

        # 全连接层
        x = self.dropout(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    np.random.seed(0)
    data = np.load("MODMA_features_labels_selected.npz")
    feature = data['features'][0]
    labels = data['labels']
    feature1 = torch.tensor(feature).unsqueeze(0).unsqueeze(0)  # 扩展维度 (batch_size=1, channels=1)
    model = GaborNN1D(num_classes=2).to(device)
    model.initialize_weights()
    output, c1 = model(feature1.to(device))
    plt.imshow(c1.squeeze(0).detach().cpu().numpy(), interpolation='nearest', cmap='hot')
    plt.show()
