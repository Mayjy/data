import os
import shutil
import random

def split_dataset(base_dir, src_sequence="00", val_size=100, test_size=100, seed=42):
    """
    分割数据集并创建验证集和测试集
    :param base_dir: 数据集根目录（包含sequences文件夹的路径）
    :param src_sequence: 原始数据所在的序列目录（默认为00）
    :param val_size: 验证集样本数
    :param test_size: 测试集样本数
    :param seed: 随机种子（保证可重复性）
    """
    # 路径设置
    src_dir = os.path.join(base_dir, "sequences", src_sequence)
    val_dir = os.path.join(base_dir, "sequences", "01")
    test_dir = os.path.join(base_dir, "sequences", "02")

    # 获取所有样本（基于bin文件）
    bin_files = [f for f in os.listdir(os.path.join(src_dir, "velodyne")) if f.endswith(".bin")]
    base_names = [os.path.splitext(f)[0] for f in bin_files]  # 去除扩展名
    
    # 随机打乱并分割
    random.seed(seed)
    random.shuffle(base_names)
    
    # 分割数据集
    val_samples = base_names[:val_size]
    test_samples = base_names[val_size:val_size+test_size]
    
    # 创建目标目录
    for seq in ["01", "02"]:
        os.makedirs(os.path.join(base_dir, "sequences", seq, "velodyne"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "sequences", seq, "labels"), exist_ok=True)

    # 移动验证集（01）
    print(f"正在迁移验证集到01（{val_size}个样本）...")
    for base in val_samples:
        # 移动点云文件
        src_bin = os.path.join(src_dir, "velodyne", f"{base}.bin")
        dst_bin = os.path.join(val_dir, "velodyne", f"{base}.bin")
        shutil.move(src_bin, dst_bin)
        
        # 移动标签文件
        src_label = os.path.join(src_dir, "labels", f"{base}.label")
        dst_label = os.path.join(val_dir, "labels", f"{base}.label")
        shutil.move(src_label, dst_label)

    # 移动测试集（02）
    print(f"正在迁移测试集到02（{test_size}个样本）...")
    for base in test_samples:
        src_bin = os.path.join(src_dir, "velodyne", f"{base}.bin")
        dst_bin = os.path.join(test_dir, "velodyne", f"{base}.bin")
        shutil.move(src_bin, dst_bin)
        
        src_label = os.path.join(src_dir, "labels", f"{base}.label")
        dst_label = os.path.join(test_dir, "labels", f"{base}.label")
        shutil.move(src_label, dst_label)

    # 统计结果
    remaining = len(base_names) - val_size - test_size
    print(f"分割完成！剩余训练样本：{remaining} | 验证集：{val_size} | 测试集：{test_size}")

if __name__ == "__main__":
    # 配置参数
    DATASET_ROOT = "/home/may/data/aqc_phase1/dataset"  # 替换为实际路径
    
    split_dataset(
        base_dir=DATASET_ROOT,
        val_size=100,
        test_size=100,
        seed=2023  # 固定随机种子保证可重复性
    )