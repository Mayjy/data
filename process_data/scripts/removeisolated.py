import os
import numpy as np
import open3d as o3d

def load_point_cloud(file_path):
    """
    加载 PCD 点云文件，支持 XYZ + 语义标签
    """
    # 读取 PCD 文件，手动解析
    points = []
    labels = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 跳过 PCD 文件头部
    data_started = False
    for line in lines:
        if line.startswith("DATA ascii"):
            data_started = True
            continue
        if data_started:
            # 读取每行点云数据（包括标签）
            parts = line.split()
            if len(parts) == 4:
                x, y, z, label = map(float, parts)
                points.append([x, y, z])
                labels.append(int(label))  # 确保标签是整数
    points = np.array(points)
    labels = np.array(labels).reshape(-1, 1)  # 标签转为列向量
    return points, labels

def remove_invalid_points(points, labels):
    """
    移除无效点（NaN/Inf）
    """
    mask = np.all(np.isfinite(points), axis=1)
    return points[mask], labels[mask]

def remove_outliers(pcd, points, labels, nb_neighbors=20, std_ratio=2.0):
    """
    统计滤波去除离群点
    """
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cl, points[ind], labels[ind]

def check_point_cloud_range(points):
    """
    计算点云的最小/最大范围
    """
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    return min_bound, max_bound

def save_pcd_with_labels(file_path, points, labels):
    """
    以 PointXYZI（XYZ + 标签）格式保存 PCD，确保 ROS 读取时能找到 'intensity' 字段
    """
    # 组合 XYZ + 标签（作为 Intensity）
    structured_array = np.zeros(points.shape[0], dtype=[("x", np.float32), 
                                                         ("y", np.float32), 
                                                         ("z", np.float32), 
                                                         ("intensity", np.int32)])  # 标签为整型
    structured_array["x"] = points[:, 0]
    structured_array["y"] = points[:, 1]
    structured_array["z"] = points[:, 2]
    structured_array["intensity"] = labels.flatten()  # 将标签作为整型保存

    # 保存 PCD
    with open(file_path, 'w') as f:
        # 写入头部信息
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z intensity\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F I\n")  # int32 for intensity
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")

        # 写入点云数据
        for i in range(len(points)):
            f.write(f"{structured_array['x'][i]} {structured_array['y'][i]} {structured_array['z'][i]} {structured_array['intensity'][i]}\n")

def process_point_clouds(directory, output_directory):
    """
    处理点云：去除无效点、离群点，计算范围，并保存新的 PCD
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    min_bounds = []
    max_bounds = []
    avg_min_bound = []
    avg_max_bound = []

    for filename in os.listdir(directory):
        if filename.endswith(".pcd"):
            file_path = os.path.join(directory, filename)
            print(f"Processing: {file_path}")

            # 1. 读取 PCD
            points, labels = load_point_cloud(file_path)

            # 2. 去除无效点
            points, labels = remove_invalid_points(points, labels)

            # 3. 统计滤波去除离群点
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd, points, labels = remove_outliers(pcd, points, labels)

            # 4. 计算点云范围
            min_bound, max_bound = check_point_cloud_range(points)
            min_bounds.append(min_bound)
            max_bounds.append(max_bound)
            avg_min_bound.append(min_bound)
            avg_max_bound.append(max_bound)

            print(f"Range: min {min_bound}, max {max_bound}")

            # 5. 以 PointXYZI 格式保存
            output_path = os.path.join(output_directory, filename)
            save_pcd_with_labels(output_path, points, labels)
            print(f"Saved: {output_path}\n")

    # 计算整体的平均范围、最小值范围和最大值范围
    avg_min_bound = np.mean(avg_min_bound, axis=0) if avg_min_bound else None
    avg_max_bound = np.mean(avg_max_bound, axis=0) if avg_max_bound else None
    min_bound_overall = np.min(min_bounds, axis=0) if min_bounds else None
    max_bound_overall = np.max(max_bounds, axis=0) if max_bounds else None

    print(f"Average Point Cloud Range: min {avg_min_bound}, max {avg_max_bound}")
    print(f"Overall Minimum Bound: {min_bound_overall}")
    print(f"Overall Maximum Bound: {max_bound_overall}")

if __name__ == "__main__":
    dataset_path = "/home/may/data/dataset"
    processed_path = "/home/may/data/processed_pcd"
    process_point_clouds(dataset_path, processed_path)
