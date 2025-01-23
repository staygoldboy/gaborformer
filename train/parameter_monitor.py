import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from gaborformer import gaborformer
from gaborformer import GaborConv1d
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

'''
热力图视图：

展示所有参数的总变化量
使用颜色深浅表示变化大小
直接标注具体的变化值


相对变化视图：

只显示变化最大的5个通道
使用相对变化百分比而不是绝对值
添加了参考基准值


详细单通道视图：

为每个变化最大的通道创建单独的详细图表
包含实际值、趋势线和变化范围
显示统计信息（均值、标准差、总变化量、趋势斜率）



这样的可视化方式能够：

更容易识别哪些参数发生了显著变化
更清晰地展示微小的变化
提供更多的统计信息来分析参数变化
'''
class AudioNpzDataset(Dataset):
    def __init__(self, audio_data, labels):
        """
        初始化数据集
        audio_data: numpy数组，形状为(n_samples, n_timestamps)
        labels: numpy数组，形状为(n_samples,)
        """
        self.audio_data = torch.FloatTensor(audio_data)
        self.labels = torch.LongTensor(labels)

        # 确保音频数据是二维的 [batch, timestamps]
        if len(self.audio_data.shape) == 1:
            self.audio_data = self.audio_data.unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.audio_data[idx].unsqueeze(0), self.labels[idx]  # 添加通道维度 [1, timestamps]


class GaborParameterMonitor:
    def __init__(self):
        self.parameter_history = defaultdict(list)

    def capture_parameters(self, model, epoch):
        """在每个epoch后捕获Gabor滤波器的参数"""
        for name, module in model.named_modules():
            if isinstance(module, GaborConv1d):
                # 获取当前参数值
                freq = module.freq.detach().cpu().numpy()
                psi = module.psi.detach().cpu().numpy()
                sigma = module.sigma.detach().cpu().numpy()

                # 存储参数
                self.parameter_history[f'{name}_freq'].append((epoch, freq))
                self.parameter_history[f'{name}_psi'].append((epoch, psi))
                self.parameter_history[f'{name}_sigma'].append((epoch, sigma))

    def plot_parameter_changes(self, save_dir='./parameter_plots'):
        """绘制参数变化的趋势图"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        for param_name, history in self.parameter_history.items():
            epochs, values = zip(*history)
            values = np.array(values)
            out_channels, in_channels = values[0].shape

            # 计算每个参数的变化量
            changes = np.abs(values[1:] - values[:-1])
            total_change = np.sum(changes, axis=0)  # 计算总变化量

            # 找出变化最大的几个通道
            flat_changes = total_change.flatten()
            top_indices = np.argsort(flat_changes)[-5:]  # 选择变化最大的5个通道
            top_out_channels = top_indices // in_channels
            top_in_channels = top_indices % in_channels

            # 1. 热力图
            plt.figure(figsize=(10, 8))
            im = plt.imshow(total_change, cmap='YlOrRd', aspect='auto')
            plt.xlabel('Input Channel')
            plt.ylabel('Output Channel')
            plt.title(f'Total Changes in {param_name}')
            plt.colorbar(im)

            # 为热力图添加数值标注
            for i in range(out_channels):
                for j in range(in_channels):
                    text = plt.text(j, i, f'{total_change[i, j]:.2e}',
                                    ha="center", va="center", color="black",
                                    fontsize=8)

            plt.tight_layout()
            # 保存热力图
            plt.savefig(os.path.join(save_dir, f'{param_name}_heatmap.png'),
                        bbox_inches='tight', dpi=300)
            plt.close()

            # 2. 时间序列图
            plt.figure(figsize=(12, 6))
            for idx in range(len(top_indices)):
                out_ch = top_out_channels[idx]
                in_ch = top_in_channels[idx]
                # 计算这个通道的相对变化
                values_channel = values[:, out_ch, in_ch]
                baseline = values_channel[0]
                relative_changes = (values_channel - baseline) / np.abs(baseline) * 100
                # 绘制相对变化百分比
                plt.plot(epochs, relative_changes,
                         label=f'out_{out_ch}_in_{in_ch} (Reference value: {baseline:.2e})',
                         marker='o', markersize=4)

            plt.xlabel('Epoch')
            plt.ylabel('Relative Change (%)')
            plt.title(f'Top 5 Changing Parameters in {param_name}')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            # 保存时间序列图
            plt.savefig(os.path.join(save_dir, f'{param_name}_timeseries.png'),
                        bbox_inches='tight', dpi=300)
            plt.close()

            # 3. 为每个变化最大的通道创建详细的单独图表
            for idx in range(len(top_indices)):
                out_ch = top_out_channels[idx]
                in_ch = top_in_channels[idx]

                plt.figure(figsize=(10, 6))
                values_channel = values[:, out_ch, in_ch]

                # 主曲线
                plt.plot(epochs, values_channel, 'b-', label='Actual Value', zorder=3)

                # 添加变化趋势
                z = np.polyfit(epochs, values_channel, 1)
                p = np.poly1d(z)
                plt.plot(epochs, p(epochs), 'r--', label='Trend', zorder=2)

                # 添加误差范围
                std_dev = np.std(values_channel)
                plt.fill_between(epochs,
                                 values_channel - std_dev,
                                 values_channel + std_dev,
                                 alpha=0.2, color='blue', zorder=1)

                plt.title(f'{param_name} out_{out_ch}_in_{in_ch} Detailed View')
                plt.xlabel('Epoch')
                plt.ylabel('Value')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()

                # 添加统计信息
                stats_text = (f'Mean: {np.mean(values_channel):.2e}\n'
                              f'Std: {std_dev:.2e}\n'
                              f'Total Change: {total_change[out_ch, in_ch]:.2e}\n'
                              f'Trend Slope: {z[0]:.2e}')
                plt.text(0.02, 0.98, stats_text,
                         transform=plt.gca().transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir,
                                         f'{param_name}_channel_{out_ch}_{in_ch}_detail.png'),
                            bbox_inches='tight', dpi=300)
                plt.close()

    def visualize_filters(self, model, epoch, save_dir='./filter_plots'):
        """可视化当前的Gabor滤波器"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        for name, module in model.named_modules():
            if isinstance(module, GaborConv1d):
                # 获取滤波器的响应
                t = torch.linspace(-module.t0 + 1, module.t0, module.kernel_size[0])

                # 创建图表
                fig, axes = plt.subplots(module.out_channels, module.in_channels,
                                         figsize=(4 * module.in_channels, 3 * module.out_channels))
                if module.out_channels == 1 and module.in_channels == 1:
                    axes = np.array([[axes]])
                elif module.out_channels == 1:
                    axes = axes[np.newaxis, :]
                elif module.in_channels == 1:
                    axes = axes[:, np.newaxis]

                # 绘制每个滤波器
                for i in range(module.out_channels):
                    for j in range(module.in_channels):
                        sigma = module.sigma[i, j].item()
                        freq = module.freq[i, j].item()
                        psi = module.psi[i, j].item()

                        # 计算Gabor滤波器响应
                        g = torch.exp(-0.5 * (t ** 2 / (sigma + 1e-3) ** 2))
                        g = g * torch.cos(freq * t + psi)
                        g = g / (sigma * np.sqrt(2 * np.pi) + 1e-3)

                        axes[i, j].plot(t.cpu().numpy(), g.cpu().numpy())
                        axes[i, j].set_title(f'Filter out_{i}_in_{j}')
                        axes[i, j].grid(True)

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'{name}_filters_epoch_{epoch}.png'),dpi=300)
                plt.close()


# 使用示例
def train_with_monitoring(model, train_loader, criterion, optimizer, num_epochs,device):
    monitor = GaborParameterMonitor()

    for epoch in range(num_epochs):
        model.train()
        i =0
        for data, target in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training'):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output,_,_,_ = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


        # 在每个epoch结束后记录参数
        monitor.capture_parameters(model, epoch)
        # 可视化当前的滤波器
        monitor.visualize_filters(model, epoch)

    # 在训练结束后绘制参数变化图
    monitor.plot_parameter_changes()

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 参数设置
    train_ratio = 0.8
    batch_size = 32
    seed = 42
    shuffle = True
    sample_length = 110
    dim = 128
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
        num_classes=2
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    data = np.load("MODMA_features_labels_selected.npz")
    audio_data = data['features']
    labels = data['labels']
    if isinstance(labels[0], str):
        labels = [1 if label == 'MDD' else 0 for label in labels]
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        audio_data, labels,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels
    )
    # 创建数据集
    train_dataset = AudioNpzDataset(train_data, train_labels)
    valid_dataset = AudioNpzDataset(temp_data, temp_labels)
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    train_with_monitoring(model, train_loader, criterion, optimizer,num_epochs=10,device=device)

if __name__ == '__main__':
    main()
