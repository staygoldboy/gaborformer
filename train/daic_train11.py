from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from gabor1d import GaborNN1D
from gaborformer import gaborformer


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class AudioNpzDataset(Dataset):
    def __init__(self, audio_data, labels, augment=False):
        self.audio_data = torch.FloatTensor(audio_data)
        self.labels = torch.LongTensor(labels)
        self.augment = augment

        if len(self.audio_data.shape) == 1:
            self.audio_data = self.audio_data.unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def add_noise(self, data):
        noise = torch.randn_like(data) * 0.01
        return data + noise

    def __getitem__(self, idx):
        data = self.audio_data[idx].unsqueeze(0)
        if self.augment and torch.rand(1) < 0.5:
            data = self.add_noise(data)
        return data, self.labels[idx]

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.stopped_epoch = 0

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_model = model.state_dict().copy()
            self.counter = 0
        return False


def train_model(model, train_loader, valid_loader, device, num_epochs=50):
    # 使用带权重的Focal Loss
    label_counts = Counter(train_loader.dataset.labels.numpy())
    total = sum(label_counts.values())
    class_weights = torch.FloatTensor([total / (len(label_counts) * count) for count in label_counts.values()]).to(
        device)
    criterion = FocalLoss(gamma=2, alpha=class_weights)

    # 使用带warmup的学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.1
    )
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

    train_losses = []
    valid_losses = []
    accuracies = []
    best_accuracy = 0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for data, target in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training'):
            data, target = data.to(device), target.to(device)

            # 梯度累积，每4步更新一次
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

        train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * train_correct / train_total
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        valid_loss = 0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for data, target in tqdm(valid_loader, desc='Validation'):
                data, target = data.to(device), target.to(device)
                output = model(data)

                loss = criterion(output, target)
                valid_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                valid_total += target.size(0)
                valid_correct += (predicted == target).sum().item()

        valid_loss = valid_loss / len(valid_loader)
        valid_accuracy = 100. * valid_correct / valid_total

        valid_losses.append(valid_loss)
        accuracies.append(valid_accuracy)

        # 保存最佳模型
        # if valid_accuracy > best_accuracy:
        #     best_accuracy = valid_accuracy
        #     torch.save(model.state_dict(), 'best_model.pth')

        # 打印训练信息
        print(f'\nEpoch: {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

        # 早停检查
        if early_stopping(model, valid_loss):
            print(f'Early stopping triggered at epoch {epoch + 1}')
            model.load_state_dict(early_stopping.best_model)
            break

    return train_losses, valid_losses, accuracies


def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    test_loss = 0
    criterion = FocalLoss()

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # 使用 softmax 进行预测
            probabilities = F.softmax(output, dim=1)
            _, predicted = torch.max(probabilities, 1)

            # 计算损失
            loss = criterion(output, target)
            test_loss += loss.item()

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(target.cpu().numpy())

    # 计算平均测试损失
    test_loss /= len(test_loader)

    # 计算准确率
    accuracy = accuracy_score(true_labels, predictions)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix\nTest Accuracy: {accuracy * 100:.2f}%\nTest Loss: {test_loss:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return accuracy, cm, test_loss


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
    sample_length = 988
    dim = 128

    # 加载数据
    data1 = np.load(
        "/home/zlh/DAIC_WOZ-generated_database/train/clipped_data/no_gender_balance/audio/MODMA_features_train.npz")
    data2 = np.load(
        "/home/zlh/DAIC_WOZ-generated_database/validation/clipped_data/no_gender_balance/audio/MODMA_features_val.npz")

    # 合并数据
    combined_data = np.concatenate((data1['features'], data2['features']), axis=0)
    combined_labels = np.concatenate((data1['labels'], data2['labels']), axis=0)

    # 计算类别权重
    label_counts = Counter(combined_labels)
    total = len(combined_labels)
    class_weights = torch.FloatTensor([total / (len(label_counts) * count) for count in label_counts.values()])

    print("\nClass distribution:")
    for label, count in label_counts.items():
        print(f"Class {label}: {count} samples ({count / total * 100:.2f}%)")
    print("\nClass weights:", class_weights.numpy())

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
    train_dataset = AudioNpzDataset(train_data, train_labels, augment=True)
    valid_dataset = AudioNpzDataset(valid_data, valid_labels, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # 打印数据集信息
    print("\nDataset Information:")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(valid_dataset)}")

    # 创建模型
    model = gaborformer(
        dim=dim,
        depth=3,
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
        num_classes=2
    ).to(device)

    # 训练模型
    train_losses, valid_losses, accuracies = train_model(model, train_loader, valid_loader, device,num_epochs=20)

    # 绘制训练过程
    plt.figure(figsize=(15, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
