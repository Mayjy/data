import os
import numpy as np
from sklearn.cluster import DBSCAN

def read_pcd_xyz_intensity(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    header = []
    data_start = 0
    for i, line in enumerate(lines):
        header.append(line)
        if line.strip().startswith('DATA'):
            data_start = i + 1
            break
    data = np.loadtxt(lines[data_start:])
    xyz = data[:, :3]
    intensity = data[:, 3].astype(int)
    return header, xyz, intensity

def write_pcd_xyz_intensity(filename, header, xyz, intensity):
    with open(filename, 'w') as f:
        for line in header:
            if line.startswith('POINTS'):
                f.write(f"POINTS {xyz.shape[0]}\n")
            elif line.startswith('WIDTH'):
                f.write(f"WIDTH {xyz.shape[0]}\n")
            else:
                f.write(line)
        for i in range(xyz.shape[0]):
            f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} {int(intensity[i])}\n")

def process_pcd_file(input_path, output_path, dbscan_eps=0.5, dbscan_min_samples=8):
    header, xyz, intensity = read_pcd_xyz_intensity(input_path)
    mask0 = intensity == 0
    mask1 = intensity == 1
    mask2 = intensity == 2
    xyz0 = xyz[mask0]
    xyz1 = xyz[mask1]
    xyz2 = xyz[mask2]
    inten0 = intensity[mask0]
    inten2 = intensity[mask2]

    xyz1_filtered = []
    total_removed = 0
    if xyz1.shape[0] > 0:
        clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(xyz1)
        labels = clustering.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)

        # 非噪声簇
        label_count_dict = {label: count for label, count in zip(unique_labels, counts) if label != -1}
        
        if label_count_dict:
            largest_label = max(label_count_dict, key=label_count_dict.get)
            largest_count = label_count_dict[largest_label]
            xyz1_filtered = xyz1[labels == largest_label]
            inten1_filtered = np.ones(xyz1_filtered.shape[0], dtype=int)
            total_removed = xyz1.shape[0] - xyz1_filtered.shape[0]
        else:
            xyz1_filtered = np.empty((0, 3))
            inten1_filtered = np.empty((0,), dtype=int)
            total_removed = xyz1.shape[0]
    else:
        xyz1_filtered = np.empty((0, 3))
        inten1_filtered = np.empty((0,), dtype=int)

    # 打印当前文件的处理结果
    print(f"文件: {os.path.basename(input_path)}")
    print(f"原始标签为1的点数: {xyz1.shape[0]}")
    print(f"最终保留的标签为1的点数（最大簇）: {xyz1_filtered.shape[0]}")
    print(f"删除的标签为1的点数: {total_removed}")
    print("-" * 50)

    # 合并最终点云
    xyz_all = np.vstack([xyz0, xyz1_filtered, xyz2])
    inten_all = np.concatenate([inten0, inten1_filtered, inten2])
    write_pcd_xyz_intensity(output_path, header, xyz_all, inten_all)

def process_pcd_folder(input_folder, output_folder, dbscan_eps=0.5, dbscan_min_samples=8):
    os.makedirs(output_folder, exist_ok=True)
    pcd_files = [f for f in os.listdir(input_folder) if f.endswith('.pcd')]

    for pcd_file in pcd_files:
        input_path = os.path.join(input_folder, pcd_file)
        output_path = os.path.join(output_folder, pcd_file)
        process_pcd_file(input_path, output_path, dbscan_eps, dbscan_min_samples)

if __name__ == "__main__":
    input_dir = 'process_data/data/aftercut_dataset'
    output_dir = 'process_data/data/afterDBSCAN_dataset'
    process_pcd_folder(
        input_folder=input_dir,
        output_folder=output_dir,
        dbscan_eps=2,
        dbscan_min_samples=5
    )
