import os
import numpy as np

def read_pcd(input_path):
    """
    读取 PCD 文件，返回点云数据和标签。
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    # 查找数据开始的位置
    data_start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('DATA'):
            data_start_idx = i + 1
            break
    
    # 读取点云数据
    point_cloud = []
    labels = []
    for line in lines[data_start_idx:]:
        values = line.split()
        if len(values) >= 4:
            x, y, z = map(float, values[:3])
            intensity = int(float(values[3]))  # 标签值
            point_cloud.append([x, y, z])
            labels.append(intensity)
    
    return np.array(point_cloud, dtype=np.float32), np.array(labels, dtype=np.int32)

def convert_to_bin_and_label(input_pcd_dir, output_bin_dir, output_label_dir):
    """
    将 PCD 文件转换为 .bin 和 .label 文件。
    """
    if not os.path.exists(output_bin_dir):
        os.makedirs(output_bin_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)
    
    for filename in os.listdir(input_pcd_dir):
        if filename.endswith('.pcd'):
            pcd_path = os.path.join(input_pcd_dir, filename)
            point_cloud, labels = read_pcd(pcd_path)
            
            # 保存 .bin 文件
            bin_path = os.path.join(output_bin_dir, filename.replace('.pcd', '.bin'))
            point_cloud.tofile(bin_path)
            
            # 保存 .label 文件
            label_path = os.path.join(output_label_dir, filename.replace('.pcd', '.label'))
            labels.tofile(label_path)

def organize_data(input_bin_dir, input_label_dir, output_dir):
    """
    按照 SemanticKITTI 的数据结构组织 .bin 和 .label 文件。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sequences = set()
    for filename in os.listdir(input_bin_dir):
        if filename.endswith('.bin'):
            sequence_id = filename.split('_')[0]  # 假设文件名格式为 'sequenceID_index.pcd'
            sequences.add(sequence_id)
    
    for seq in sequences:
        seq_dir = os.path.join(output_dir, seq)
        if not os.path.exists(seq_dir):
            os.makedirs(seq_dir)
        
        velodyne_dir = os.path.join(seq_dir, 'velodyne')
        labels_dir = os.path.join(seq_dir, 'labels')
        if not os.path.exists(velodyne_dir):
            os.makedirs(velodyne_dir)
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
        
        # 复制文件到对应的文件夹
        for filename in os.listdir(input_bin_dir):
            if filename.startswith(seq) and filename.endswith('.bin'):
                os.rename(
                    os.path.join(input_bin_dir, filename),
                    os.path.join(velodyne_dir, filename)
                )
        for filename in os.listdir(input_label_dir):
            if filename.startswith(seq) and filename.endswith('.label'):
                os.rename(
                    os.path.join(input_label_dir, filename),
                    os.path.join(labels_dir, filename)
                )

if __name__ == '__main__':
    input_pcd_dir = '/home/may/data/dataset'  # 替换为您的 PCD 文件夹路径
    output_bin_dir = '/home/may/data/kitti/dataset/sequences/00/velodyne'  # 替换为输出的 .bin 文件夹路径
    output_label_dir = 'kitti/dataset/sequences/00/labels'  # 替换为输出的 .label 文件夹路径
    output_dir = '/home/may/data/kitti/dataset/sequences/00'  # 替换为 SemanticKITTI 数据集根目录路径

    # 步骤 1：转换 PCD 文件为 .bin 和 .label 文件
    convert_to_bin_and_label(input_pcd_dir, output_bin_dir, output_label_dir)
    
    # 步骤 2：组织数据结构
    organize_data(output_bin_dir, output_label_dir, output_dir)
