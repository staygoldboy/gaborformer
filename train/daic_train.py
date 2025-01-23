from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
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
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from collections import Counter
import os
from datetime import datetime

log_path = os.path.join('/home/zlh/gabor/logs','log_daic_7.txt')
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # 类别权重

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


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


# 6. 训练函数
def train_model(model, train_loader, valid_loader, device, num_epochs=50):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    accuracys = []
    best_f1_score = 0
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
        true_labels = []
        with torch.no_grad():
            for data, target in tqdm(valid_loader, desc='Validation'):
                data, target = data.to(device), target.to(device)
                output = model(data)
                valid_loss += criterion(output, target).item()
                probs = F.softmax(output, dim=1)
                _, predicted = torch.max(probs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(target.cpu().numpy())

        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)
        accuracy = 100. * correct / total
        accuracys.append(accuracy)
        cm = confusion_matrix(true_labels, predictions)
        # 更新学习率
        scheduler.step(valid_loss)

        precision = float(cm[0][0]) / (cm[0][0] + cm[1][0])
        recall = float(cm[0][0]) / (cm[0][0] + cm[0][1])
        f1_score = 2 * (precision * recall) / (precision + recall)
        # 保存最佳模型
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            torch.save(model.state_dict(), 'best_model_daic.pth')
            # torch.save(model.state_dict(), 'best_model_GCN1D.pth')
            # torch.save(model.state_dict(), 'best_model_gabor.pth')

        output = f'{datetime.now()}: Epoch: {epoch + 1}/{num_epochs} Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}'
        with open(log_path, "a+") as f:
            f.write(output + '\n')
            f.close()
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.2f}%')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')
        print('--------------------')

    return train_losses, valid_losses,accuracys


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
    # 设置随机种子以确保可重复性
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 参数设置
    train_ratio = 0.8
    batch_size = 64
    sample_length = 524
    dim = 128
    # 创建模型
    # model = DFBLClassifier().to(device)
    # model= GaborNN1D(num_classes=2,sample_length=sample_length).to(device)
    # model.initialize_weights()
    model = gaborformer(
        dim=dim,
        depth=4,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.5,
        ff_dropout=0.5,
        conv_dropout=0.5,
        conv_causal=False,
        sample_length=sample_length,
        num_classes= 2
    ).to(device)
    # 加载数据
    data = np.load("/home/zlh/DAIC_WOZ-generated_database/DAIC_features_all_selected.npz")
    combined_data = data['features']
    combined_labels = data['labels']
    # np.savez('/home/zlh/DAIC_WOZ-generated_database/DAIC_features_all.npz', features=combined_data, labels=combined_labels)
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
    # 数据标准化
    scaler = MinMaxScaler()
    combined_data_normalized = scaler.fit_transform(combined_data)


    # 划分数据集
    train_data, valid_data, train_labels, valid_labels = train_test_split(
        combined_data_normalized, combined_labels,
        train_size=train_ratio,
        random_state=SEED,
        stratify=combined_labels
    )

    # 创建数据加载器
    train_dataset = AudioNpzDataset(train_data, train_labels)
    valid_dataset = AudioNpzDataset(valid_data, valid_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # 打印数据集信息
    print("\n数据集划分信息:")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(valid_dataset)}")
    # 训练模型
    train_losses, valid_losses,accuracys = train_model(model, train_loader, valid_loader, device,num_epochs=50)
    #绘制损失函数和准确率图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train_losse')
    plt.plot(valid_losses, label='valid_losse')
    plt.xlabel('Epoch')
    plt.ylabel('LOSS')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(accuracys, label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()




if __name__ == "__main__":
    main()
