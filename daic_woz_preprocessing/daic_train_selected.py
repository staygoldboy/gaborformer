import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd


def normalize_features(X):
    # 归一化特征矩阵 X 到 [0, 1] 区间
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized


def evaluate_feature_subset(X, y, feature_indices):
    # 根据给定的特征索引选择特征子集，并使用分类器评估错误率
    X_subset = X[:, feature_indices]
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)

    # 使用KNN分类器作为示例
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    error = 1 - accuracy_score(y_test, y_pred)  # 错误率
    return error


def visualize_features(X, y, title, save_path):

    X = normalize_features(X)

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

def main(X, y):
    # 第一步：对特征进行归一化
    X = normalize_features(X)

    # 第二步：使用NCA算法选择特征索引
    nca = NeighborhoodComponentsAnalysis(random_state=42)
    nca.fit(X, y)
    index = np.argsort(-np.abs(nca.components_).sum(axis=0))  # 特征重要性降序排列索引

    # 第三步：根据不同的特征子集大小计算分类误差
    errors = []
    for k in range(1, 901):
        feature_indices = index[:100 + k]
        error = evaluate_feature_subset(X, y, feature_indices)
        errors.append(error)

    # 错误率绘图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 101), errors[0:100], marker='o')
    plt.xlabel("Number of Features")
    plt.ylabel("Error Rate")
    plt.title("Error Rate vs Number of DAIC Features")
    plt.grid(True)
    plt.savefig("/home/zlh/gabor/results/daic/error_rate.png",bbox_inches='tight', dpi=300)
    plt.close()

    # 找到最小错误率对应的特征数量
    min_error = min(errors)
    min_index = errors.index(min_error) + 1  # 由于索引从0开始，+1

    # 最终选择最佳的特征子集
    best_feature_indices = index[:100 + min_index]
    last = X[:, best_feature_indices]  # 最终的特征矩阵

    return last, best_feature_indices, min_error


# 数据
data = np.load("/home/zlh/DAIC_WOZ-generated_database/DAIC_features_all.npz")
X = data['features']
y = data['labels']


# 执行特征选择过程
last, best_feature_indices, min_error = main(X, y)

print("最终选择的特征索引:", best_feature_indices)
print("最小错误率:", min_error)
print("最终特征矩阵的形状:", last.shape)

# np.savez("MODMA_features_labels_selected.npz", features=last, labels=y)
# print("特征选择完成，已保存为 MODMA_features_labels_selected1.npz")

#  未进行特征选择前的可视化
visualize_features(X, y,title="Original Features",save_path="/home/zlh/gabor/results/daic/original_features.png")

# 特征选择后的可视化
visualize_features(last, y, title="Selected Features",save_path="/home/zlh/gabor/results/daic/selected_features.png")


# file_path='/home/zlh/gabor/emobase1.csv'
# df = pd.read_csv(file_path,header=None)
# rows_to_extract = best_feature_indices
# extracted_data = df.iloc[rows_to_extract]
# extracted_data.to_csv('/home/zlh/gabor/feature csv/DAIC_selected_features.csv')
# print("特征选择完成，已保存为 DAIC_selected_features.csv")