from sklearn.metrics import mean_absolute_error, mean_squared_error,accuracy_score, confusion_matrix
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

log_path = os.path.join('/home/zlh/gabor/logs','log_modma_reg.txt')

# 自定义数据集类
class AudioNpzDataset(Dataset):
    def __init__(self, audio_data, scores):
        """
        Initialize dataset for regression
        audio_data: numpy array, shape (n_samples, n_timestamps)
        scores: numpy array, shape (n_samples,)
        """
        self.audio_data = torch.FloatTensor(audio_data)
        self.scores = torch.FloatTensor(scores).reshape(-1, 1)  # Reshape scores for regression

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return self.audio_data[idx].unsqueeze(0), self.scores[idx]


# 6. 训练函数
def train_model(model, train_loader, valid_loader, device, num_epochs=50):
    criterion = nn.SmoothL1Loss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    valid_maes = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for data, target in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training'):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        valid_loss = 0
        correct = 0
        total = 0
        predictions = []
        true_scores = []
        with torch.no_grad():
            for data, target in tqdm(valid_loader, desc='Validation'):
                data, target = data.to(device), target.to(device)
                output = model(data)
                valid_loss += criterion(output, target).item()
                predictions.extend(output.cpu().numpy().flatten())
                true_scores.extend(target.cpu().numpy().flatten())

        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)
        valid_mae = mean_absolute_error(true_scores, predictions)
        valid_rmse = np.sqrt(mean_squared_error(true_scores, predictions))
        valid_maes.append(valid_mae)
        # 更新学习率
        scheduler.step(valid_loss)

        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # torch.save(model.state_dict(), 'best_model_GCN1D.pth')
            # torch.save(model.state_dict(), 'best_model_gabor.pth')

        output = f'{datetime.now()}: Epoch: {epoch + 1}/{num_epochs} Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid MAE: {valid_mae:.4f}, Valid RMSE: {valid_rmse:.4f}'
        with open(log_path, "a+") as f:
            f.write(output + '\n')
            f.close()
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
        print(f'Valid MAE: {valid_mae:.4f}, Valid RMSE: {valid_rmse:.4f}')
        print('--------------------')

    return train_losses, valid_losses,valid_maes


# 7. 评估函数
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(target.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(true_labels, predictions)

    # 绘制混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return accuracy, cm


# 8. 使用示例
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        num_classes= 1
    ).to(device)
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_data, temp_data, train_scores, temp_scores = train_test_split(
        audio_data, scores,
        train_size=train_ratio,
        random_state=seed
    )
    # 创建数据集
    train_dataset = AudioNpzDataset(train_data, train_scores)
    valid_dataset = AudioNpzDataset(temp_data, temp_scores)
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
    # Print dataset information
    print("\nDataset Split Information:")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(valid_dataset)}")
    # Train model
    train_losses, valid_losses, maes = train_model(model, train_loader, valid_loader, device)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(maes, label='MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
