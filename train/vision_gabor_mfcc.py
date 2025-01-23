from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from DFBL import GaborFilter, AdaptivePooling, NonlinearTransformation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import librosa
from sklearn.model_selection import train_test_split
from gabor1d import GaborNN1D
from gaborformer import gaborformer
import os
from datetime import datetime
from tqdm import tqdm
from sklearn.manifold import TSNE

save_dir = './results/modma'

# 自定义数据集类
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






# 8. 使用示例
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 参数设置
    train_ratio = 0.8
    valid_ratio = 0.1
    batch_size = 1
    seed = 42
    shuffle = True
    sample_length = 110
    dim =128
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
    model.load_state_dict(torch.load("best_model_gaborformer.pth"))
    data = np.load("MODMA_features_labels_selected.npz")
    data1 = np.load("MODMA_features_labels_scores.npz")
    audio_data = data['features']
    labels = data['labels']
    scores = data1['scores']
    if isinstance(labels[0],str):
        labels= [1 if label == 'MDD' else 0 for label in labels]

    """
        创建训练集、验证集和测试集的数据加载器

        参数:
        audio_data: 预处理后的音频数据
        labels: 标签
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        batch_size: 批次大小
        seed: 随机种子
        shuffle: 是否打乱数据

        返回:
        训练、验证和测试集的数据加载器
        """
    np.random.seed(42)
    torch.manual_seed(42)

    feature = torch.tensor(audio_data[0]).unsqueeze(0).unsqueeze(0)
    output,c,_,_ = model(feature.to(device))
    # 绘制特征热图
    plt.figure(figsize=(10, 8))
    plt.imshow(c.squeeze(0).detach().cpu().numpy(), cmap='hot', interpolation='nearest')
    plt.title('severe')
    plt.savefig(os.path.join(save_dir, f'severe_heatmap.png'), bbox_inches='tight', dpi=300)
    plt.close()




if __name__ == "__main__":
    main()