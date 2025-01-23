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
from wo_gabor import wo_gabor

log_path = os.path.join('/home/zlh/gabor/logs','log_modma_conformer.txt')

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

    best_f1_score = 0.0
    train_losses = []
    valid_losses = []
    accuracys = []

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
            # torch.save(model.state_dict(), 'best_model_gaborformer.pth')
            # print("best model saved")
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
    model = wo_gabor(
        dim=dim,
        depth=3,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_dropout=0.0,
        conv_causal=False,
        sample_length=sample_length,
        num_classes= 2
    ).to(device)
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
    train_losses, valid_losses,accuracys = train_model(model, train_loader, valid_loader, device)
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


    # 假设我们有以下数据

    # audio_files = [...] # 音频文件路径列表
    # labels = [...] # 对应的标签列表 (0 或 1)

    # 创建数据集和数据加载器
    # train_dataset = AudioDataset(train_audio_files, train_labels)
    # valid_dataset = AudioDataset(valid_audio_files, valid_labels)
    # test_dataset = AudioDataset(test_audio_files, test_labels)

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=32)
    # test_loader = DataLoader(test_dataset, batch_size=32)

    # 训练模型
    # train_losses, valid_losses = train_model(model, train_loader, valid_loader, device)

    # 评估模型
    # accuracy, confusion_matrix = evaluate_model(model, test_loader, device)

    # 预测新样本
    # model.load_state_dict(torch.load('best_model.pth'))
    # model.eval()
    # with torch.no_grad():
    #     new_audio = torch.randn(1, 1, 48000).to(device)  # 假设3秒的音频，16kHz采样率
    #     prediction = model(new_audio)
    #     _, predicted_class = torch.max(prediction, 1)
    #     print(f"Predicted class: {predicted_class.item()}")


if __name__ == "__main__":
    main()
