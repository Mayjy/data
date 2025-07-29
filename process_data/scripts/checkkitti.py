import os
import numpy as np

# 目录路径
velodyne_dir = "/home/may/data/kitti/dataset/sequences/02/velodyne"
labels_dir = "/home/may/data/kitti/dataset/sequences/02/labels"

# 加载点云文件
def load_velodyne_file(filepath):
    return np.fromfile(filepath, dtype=np.float32).reshape(-1, 3)

# 加载标签文件
def load_label_file(filepath):
    return np.fromfile(filepath, dtype=np.uint32)

# 获取所有文件名
velodyne_files = sorted(os.listdir(velodyne_dir))
label_files = sorted(os.listdir(labels_dir))

# 检查点云和标签行数是否对应
for v_file, l_file in zip(velodyne_files, label_files):
    v_path = os.path.join(velodyne_dir, v_file)
    l_path = os.path.join(labels_dir, l_file)

    # 加载点云和标签
    point_cloud = load_velodyne_file(v_path)
    labels = load_label_file(l_path)

    # 检查行数
    if point_cloud.shape[0] != labels.shape[0]:
        print(f"❌ 不一致: {v_file} - 点云行数={point_cloud.shape[0]}, 标签行数={labels.shape[0]}")
    else:
        print(f"✅ 一致: {v_file}")