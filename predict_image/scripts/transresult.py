import numpy as np
import os

def bin_npy_to_pcd(bin_path, npy_path, pcd_path):
    """
    将 .bin 点云数据和 .npy 预测标签转换为 .pcd 格式。
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"未找到 .bin 文件: {bin_path}")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"未找到 .npy 文件: {npy_path}")
    
    # 读取点云数据（XYZ）
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 3)
    
    # 读取预测标签
    labels = np.load(npy_path).astype(np.int32)
    
    # 确保点云和标签数量匹配
    if point_cloud.shape[0] != labels.shape[0]:
        raise ValueError("点云数据和标签数量不匹配")
    
    # 组合数据
    data = np.hstack((point_cloud, labels[:, None]))
    
    # 写入 PCD 文件
    with open(pcd_path, 'w') as f:
        f.write("""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F I
COUNT 1 1 1 1
WIDTH {}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {}
DATA ascii
""".format(data.shape[0], data.shape[0]))
        
        for point in data:
            f.write("{:.6f} {:.6f} {:.6f} {}\n".format(point[0], point[1], point[2], int(point[3])))
    
    print(f"转换完成: {pcd_path}")

def convert_directory(bin_dir, npy_dir, pcd_dir):
    """
    将 bin_dir 和 npy_dir 目录下的所有匹配文件批量转换为 .pcd
    """
    if not os.path.exists(pcd_dir):
        os.makedirs(pcd_dir)
    
    bin_files = {f: os.path.join(bin_dir, f) for f in os.listdir(bin_dir) if f.endswith('.bin')}
    npy_files = {f: os.path.join(npy_dir, f) for f in os.listdir(npy_dir) if f.endswith('.npy')}
    
    for npy_file in npy_files:
        npy_base = npy_file.replace("02_", "").replace("_pred.npy", ".bin")
        bin_path = bin_files.get(npy_base)
        if bin_path:
            npy_path = npy_files[npy_file]
            pcd_path = os.path.join(pcd_dir, npy_base.replace(".bin", ".pcd"))
            try:
                bin_npy_to_pcd(bin_path, npy_path, pcd_path)
            except Exception as e:
                print(f"转换失败: {npy_file}, 错误: {e}")
    
if __name__ == "__main__":
    bin_dir = "/home/may/data/process_data/data/Final_dataset2/dataset/sequences/02/velodyne"  # .bin 文件目录
    npy_dir = "/home/may/my_project/Pointcept/exp/aqc/semseg-pt-v3m1-5-train/result"  # .npy 文件目录
    pcd_dir = "/home/may/data/predict_image/data/predictresult/exp5"  # 目标 .pcd 存储目录
    
    convert_directory(bin_dir, npy_dir, pcd_dir)
