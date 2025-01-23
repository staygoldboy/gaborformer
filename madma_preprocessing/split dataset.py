import os
import shutil
import random


def split_dataset(source_dir, train_ratio=0.8):
    """
    将数据集分割为训练集和验证集

    Args:
        source_dir: 源数据目录
        train_ratio: 训练集占比，默认0.8
    """
    # 创建训练集和验证集目录
    train_dir = './split/train'
    val_dir = './split/val'

    for dir_name in [train_dir, val_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            # 在train和val下分别创建class_0和class_1目录
            os.makedirs(os.path.join(dir_name, 'class_0'))
            os.makedirs(os.path.join(dir_name, 'class_1'))

    # 处理每个类别
    for class_name in ['class_0', 'class_1']:
        # 获取该类别下所有特征图文件
        class_path = os.path.join(source_dir, class_name)
        files = os.listdir(class_path)

        # 随机打乱文件列表
        random.shuffle(files)

        # 计算训练集数量
        train_size = int(len(files) * train_ratio)

        # 分割文件列表
        train_files = files[:train_size]
        val_files = files[train_size:]

        # 复制文件到训练集
        for file in train_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(train_dir, class_name, file)
            shutil.copy2(src, dst)

        # 复制文件到验证集
        for file in val_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(val_dir, class_name, file)
            shutil.copy2(src, dst)

        print(f'{class_name}:')
        print(f'总文件数: {len(files)}')
        print(f'训练集: {len(train_files)}')
        print(f'验证集: {len(val_files)}')
        print('---')


if __name__ == '__main__':
    # 设置随机种子以确保可重复性
    random.seed(42)

    # 源数据目录
    source_dir = './feature_maps/features_20241112_111435'

    # 执行分割
    split_dataset(source_dir)