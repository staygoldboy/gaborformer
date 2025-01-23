import os
import pandas as pd
import numpy as np
import opensmile

'''
频谱特征：

包括谱质心、谱带宽、谱平坦度等。

这些特征描述了音频信号的频谱形状和分布。

MFCC（梅尔频率倒谱系数）：

包括 12 个 MFCC 系数及其一阶和二阶差分。

这些特征用于表示声音的短期功率谱。

音高特征：

包括音高、音高的差分等。

这些特征用于描述声音的感知频率。

能量和响度特征：

包括信号能量、响度等。

这些特征用于描述音频信号的能量和感知响度。

LSP（线谱对）：

包括 LSP 系数及其差分。

这些特征用于表示语音信号的谱包络。

时间域特征：

包括零交叉率等。

这些特征用于描述信号在时间域内的变化。
'''

# 初始化 openSMILE 并选择 emobase 特征集
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.emobase,
    feature_level=opensmile.FeatureLevel.Functionals
)

# 读取被试信息 Excel 文件
subject_info_path = '/home/zlh/gabor/audio_lanzhou_16khz/subjects_information_audio_lanzhou_2015.xlsx'
subject_info = pd.read_excel(subject_info_path)

# 数据集路径
data_dir = '/home/zlh/gabor/audio_lanzhou_16khz'

# 存储特征和标签的列表
features = []
labels = []
scores = []

# 遍历每个被试和他们的语音文件
for idx, row in subject_info.iterrows():
    subject_id = row['subject id']  # 假设Excel文件中有subject_id这一列
    label = row['type']  # 假设Excel文件中有label这一列
    score = row['PHQ-9']  # 假设Excel文件中有score这一列

    # 对于每个被试的29条语音
    for i in range(1, 30):
        file_name = f"{0}{subject_id}/{i:02d}.wav"
        file_path = os.path.join(data_dir, file_name)

        # 检查文件是否存在
        if os.path.exists(file_path):
            # 提取特征
            feature_vector = smile.process_file(file_path)
            # 将特征添加到列表
            features.append(feature_vector.values.flatten())
            # 将标签添加到列表
            labels.append(label)
            scores.append(score)

        else:
            print(f"文件 {file_path} 不存在，跳过。")
    print(f"{subject_id} 的特征和标签已提取。")
# 将特征和标签转换为 NumPy 数组
features = np.array(features)
labels = np.array(labels)
scores = np.array(scores)

# 保存为 NPZ 文件
np.savez('MODMA_features_labels_scores.npz', features=features, labels=labels, scores=scores)

print("特征和标签已成功保存到 MODMA_features_labels_scores.npz 文件中。")


