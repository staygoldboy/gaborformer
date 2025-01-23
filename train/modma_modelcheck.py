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


def visualize_features(X, y, title, save_path):

    # X = normalize_features(X)

    # 使用 t-SNE 对最终提取的特征集进行可视化
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # 可视化
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
    plt.colorbar(scatter, label="Target")
    plt.title(f"{title} Visualization of Features")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig(save_path,bbox_inches='tight', dpi=300)
    plt.close()

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



# 7. 评估函数
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    gabor_features = []
    conformer_features = []
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='test'):
            data, target = data.to(device), target.to(device)
            output,_,c1,c2 = model(data)
            probs = F.softmax(output, dim=1)
            _, predicted = torch.max(probs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
            gabor_features.extend(c1.cpu().numpy())
            conformer_features.extend(c2.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(true_labels, predictions)
    gabor_features = np.array(gabor_features)
    conformer_features = np.array(conformer_features)
    # gabor feautures
    # visualize_features(gabor_features,true_labels,title='gabor features',save_path='./results/modma/gabor_features.png')
    # conformer features
    # visualize_features(conformer_features,true_labels,title='conformer features',save_path='./results/modma/conformer_features.png')

    # 绘制混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('MODMA Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, f'MODMA_confusion.png'),bbox_inches='tight', dpi=300)
    plt.close()

    return accuracy, cm


# 8. 使用示例
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    # 参数设置
    train_ratio = 0.8
    valid_ratio = 0.1
    batch_size = 32
    seed = 42
    shuffle = True
    sample_length = 110
    dim =128
    # 创建模型
    # model = DFBLClassifier().to(device)
    # model= GaborNN1D(num_classes=2,sample_length=sample_length).to(device)
    # model.initialize_weights()
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
    audio_data = data['features']
    labels = data['labels']
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
    # 打印数据集信息
    print("\n数据集划分信息:")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(valid_dataset)}")
    # 训练模型
    accuracy, confusion_matrix = evaluate_model(model, valid_loader, device)



if __name__ == "__main__":
    main()
