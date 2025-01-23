import os
import pandas as pd
import numpy as np
import opensmile

# 初始化 openSMILE 并选择 emobase 特征集
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals
)

# 读取被试信息 Excel 文件
subject_info_path = 'F:/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx'
subject_info = pd.read_excel(subject_info_path)

# 数据集路径
data_dir = 'F:/audio_lanzhou_16khz/'

# 存储特征和标签的列表
features = []
labels = []

# 遍历每个被试和他们的语音文件
for idx, row in subject_info.iterrows():
    subject_id = row['subject id']  # 假设Excel文件中有subject_id这一列
    label = row['type']  # 假设Excel文件中有label这一列

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

        else:
            print(f"文件 {file_path} 不存在，跳过。")
    print(f"{subject_id} 的特征和标签已提取。")
# 将特征和标签转换为 NumPy 数组
features = np.array(features)
labels = np.array(labels)

# 保存为 NPZ 文件
np.savez('ComParE_2016.npz', features=features, labels=labels)

print("特征和标签已成功保存到 ComParE_2016.npz 文件中。")

