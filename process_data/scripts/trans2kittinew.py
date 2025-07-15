import os
import numpy as np

def read_pcd_with_label(input_path):
    """ 读取PCD文件，保留原始文件名 """
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
            label = int(float(parts[3]))  # 第四列为标签
            points.append([x, y, z, 0.0])  # 固定强度为0
            labels.append(label)
    
    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.uint32)

def convert_dataset_keep_names(src_dir, dst_root, sequence_id="00"):
    """ 保留原始文件名的转换函数 """
    # 创建目录
    seq_dir = os.path.join(dst_root, "sequences", sequence_id)
    os.makedirs(os.path.join(seq_dir, "velodyne"), exist_ok=True)
    os.makedirs(os.path.join(seq_dir, "labels"), exist_ok=True)

    # 获取所有PCD文件
    pcd_files = [f for f in os.listdir(src_dir) if f.endswith(".pcd")]
    
    for pcd_file in pcd_files:
        # 生成基础文件名（不带扩展名）
        base_name = os.path.splitext(pcd_file)[0]
        
        # 读取数据
        pcd_path = os.path.join(src_dir, pcd_file)
        points, labels = read_pcd_with_label(pcd_path)
        
        # 保存文件
        bin_path = os.path.join(seq_dir, "velodyne", f"{base_name}.bin")
        points.tofile(bin_path)
        
        label_path = os.path.join(seq_dir, "labels", f"{base_name}.label")
        labels.tofile(label_path)
    
    print(f"转换完成！共处理 {len(pcd_files)} 个文件")

def validate_with_original_names(dataset_root, sequence_id="00"):
    """ 文件名保留模式下的验证函数 """
    seq_dir = os.path.join(dataset_root, "sequences", sequence_id)
    velo_dir = os.path.join(seq_dir, "velodyne")
    label_dir = os.path.join(seq_dir, "labels")

    # 获取文件名主干（不带扩展名）
    bin_bases = set([os.path.splitext(f)[0] for f in os.listdir(velo_dir) if f.endswith(".bin")])
    label_bases = set([os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith(".label")])

    # 找出不匹配项
    missing_bin = label_bases - bin_bases
    missing_label = bin_bases - label_bases

    if missing_bin:
        print(f"缺失对应的bin文件: {missing_bin}")
    if missing_label:
        print(f"缺失对应的label文件: {missing_label}")

    assert bin_bases == label_bases, "文件不匹配"
    print(f"验证通过！匹配文件数：{len(bin_bases)}")

if __name__ == "__main__":
    # 输入输出配置
    RAW_PCD_DIR = "/home/may/data/dataset"  # 原始PCD文件夹
    OUTPUT_ROOT = "/home/may/data/aqc_phase1/dataset"  # 输出根目录
    
    # 执行转换
    convert_dataset_keep_names(RAW_PCD_DIR, OUTPUT_ROOT)
    
    # 验证结果
    validate_with_original_names(OUTPUT_ROOT)