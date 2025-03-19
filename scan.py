import os
import numpy as np
from scipy.spatial import KDTree

DATA_ROOT = "kitti/dataset/sequences"  # 数据集根目录
COORD_DIM = 3  # 检查前3个坐标 (x,y,z)
ISOLATION_DISTANCE = 1.0  # 孤立点的距离阈值

def check_invalid_points(file_path):
    """检查单个点云文件中的NaN和Inf值"""
    try:
        # 读取二进制数据 (假设格式为x,y,z)
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 3)
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {str(e)}")
        return False, [], [], []

    # 向量化检查：同时检测NaN和Inf
    invalid_mask = np.logical_or(
        np.isnan(points[:, :COORD_DIM]),
        np.isinf(points[:, :COORD_DIM])
    )
    invalid_indices = np.where(np.any(invalid_mask, axis=1))[0]

    # 检查重复点
    unique_points, unique_indices = np.unique(points, axis=0, return_index=True)
    duplicate_indices = np.setdiff1d(np.arange(points.shape[0]), unique_indices)

    # 检查孤立点
    tree = KDTree(points)
    isolated_indices = []
    for i, point in enumerate(points):
        distances, _ = tree.query(point, k=2)
        if distances[1] > ISOLATION_DISTANCE:
            isolated_indices.append(i)

    # 生成详细报告
    invalid_points = []
    for idx in invalid_indices:
        coords = points[idx, :COORD_DIM]
        problem_types = []
        if np.any(np.isnan(coords)):
            problem_types.append("NaN")
        if np.any(np.isinf(coords)):
            problem_types.append("Inf")
        invalid_points.append((idx, "|".join(problem_types), coords))

    return len(invalid_points) > 0, invalid_points, points, duplicate_indices, isolated_indices

def scan_dataset(data_root):
    """遍历数据集并统计问题"""
    error_files = 0
    total_points = 0
    invalid_points_total = 0
    duplicate_points_total = 0
    isolated_points_total = 0

    for seq in sorted(os.listdir(data_root)):  # 按顺序检查00,01,02...
        seq_path = os.path.join(data_root, seq)
        if not os.path.isdir(seq_path):
            continue

        velodyne_path = os.path.join(seq_path, "velodyne")
        if not os.path.exists(velodyne_path):
            continue

        print(f"正在扫描序列 {seq}...")
        for file in sorted(os.listdir(velodyne_path)):
            if not file.endswith(".bin"):
                continue

            file_path = os.path.join(velodyne_path, file)
            has_invalid, invalid_points, points, duplicate_indices, isolated_indices = check_invalid_points(file_path)
            total_points += len(points)
            duplicate_points_total += len(duplicate_indices)
            isolated_points_total += len(isolated_indices)

            if has_invalid:
                error_files += 1
                invalid_points_total += len(invalid_points)
                print(f" 发现无效点: {file}")
                print(f"    首5个无效点示例:")
                for pt in invalid_points[:5]:
                    print(f"     点索引 {pt[0]} - 问题类型 {pt[1]} - 坐标 {pt[2]}")
            if duplicate_indices.size > 0:
                print(f" 发现重复点: {file}")
                print(f"    首5个重复点索引: {duplicate_indices[:5]}")
            if len(isolated_indices) > 0:
                print(f" 发现孤立点: {file}")
                print(f"    首5个孤立点索引: {isolated_indices[:5]}")

    # 最终统计报告
    print("\n===== 检查完成 =====")
    print(f"扫描序列数量: {len(os.listdir(data_root))}")
    print(f"总点数: {total_points}")
    print(f"包含无效点的文件数: {error_files}")
    print(f"无效点总数: {invalid_points_total}")
    print(f"重复点总数: {duplicate_points_total}")
    print(f"孤立点总数: {isolated_points_total}")

if __name__ == "__main__":
    scan_dataset(DATA_ROOT)