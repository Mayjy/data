import os
import numpy as np
import shutil

def read_pcd_with_label(input_path):
    """读取 PCD 文件，返回点云数据和标签"""
    with open(input_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    data_start = lines.index('DATA ascii') + 1
    
    points = []
    labels = []
    for line in lines[data_start:]:
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 4:
            x, y, z = map(float, parts[:3])
            label = int(float(parts[3]))
            points.append([x, y, z])  # 去掉强度 # 强度为 0
            labels.append(label)

    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.uint32)


def convert_and_split_dataset(pcd_dir, output_root):
    """将所有 PCD 文件分成三个序列（00、01、02）并转换为 SemanticKITTI 格式"""
    # 所有 .pcd 文件排序后分组
    all_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.pcd')])
    total = len(all_files)
    assert total == 1192, f"期望 1198 个文件，实际找到 {total} 个"

    split_00 = all_files[:992]
    split_01 = all_files[992:1092]
    split_02 = all_files[1092:]

    splits = {
        "00": split_00,
        "01": split_01,
        "02": split_02
    }

    for seq_id, file_list in splits.items():
        print(f"🔧 正在处理 sequence {seq_id}（{len(file_list)} 个文件）")

        velo_dir = os.path.join(output_root, "sequences", seq_id, "velodyne")
        label_dir = os.path.join(output_root, "sequences", seq_id, "labels")
        os.makedirs(velo_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        for fname in file_list:
            input_path = os.path.join(pcd_dir, fname)
            points, labels = read_pcd_with_label(input_path)

            base_name = os.path.splitext(fname)[0]
            bin_path = os.path.join(velo_dir, f"{base_name}.bin")
            label_path = os.path.join(label_dir, f"{base_name}.label")

            points.tofile(bin_path)
            labels.tofile(label_path)

        print(f"✅ Sequence {seq_id} 处理完成，已写入 {len(file_list)} 个文件")

def validate_sequences(output_root):
    """验证每个序列中的 .bin 和 .label 文件是否一一对应"""
    for seq in ["00", "01", "02"]:
        seq_dir = os.path.join(output_root, "sequences", seq)
        velo_dir = os.path.join(seq_dir, "velodyne")
        label_dir = os.path.join(seq_dir, "labels")

        bin_files = {os.path.splitext(f)[0] for f in os.listdir(velo_dir) if f.endswith('.bin')}
        label_files = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.label')}

        missing_bin = label_files - bin_files
        missing_label = bin_files - label_files

        if missing_bin or missing_label:
            print(f"❌ Sequence {seq} 文件不匹配")
            if missing_bin:
                print(f"  缺失 bin 文件：{missing_bin}")
            if missing_label:
                print(f"  缺失 label 文件：{missing_label}")
        else:
            print(f"✅ Sequence {seq} 验证通过，共 {len(bin_files)} 个样本")

if __name__ == "__main__":
    # 修改这里的路径为你裁剪后 .pcd 文件所在的目录
    input_pcd_dir = "/home/may/data/process_data/data/afterDBSCAN_dataset"
    output_root = "/home/may/data/process_data/data/Final_dataset/dataset"

    convert_and_split_dataset(input_pcd_dir, output_root)
    validate_sequences(output_root)
