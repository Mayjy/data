import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

def read_pcd_ascii(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith('DATA'):
            data_start = i + 1
            break
    header = lines[:data_start]
    data = []
    for line in lines[data_start:]:
        if line.strip() == '':
            continue
        vals = line.strip().split()
        if len(vals) == 4:
            data.append([float(vals[0]), float(vals[1]), float(vals[2]), int(vals[3])])
    points = np.array(data)
    return header, points

def write_pcd_ascii(file_path, header, points):
    with open(file_path, 'w') as f:
        for line in header:
            if line.startswith('POINTS'):
                f.write(f'POINTS {points.shape[0]}\n')
            elif line.startswith('WIDTH'):
                f.write(f'WIDTH {points.shape[0]}\n')
            else:
                f.write(line)
        for pt in points:
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {int(pt[3])}\n")

def filter_label2_by_normal_cluster(points, normal_knn=20, cos_threshold=0.95, dbscan_eps=1, dbscan_min_samples=10):
    xyz = points[:, :3]
    labels = points[:, 3].copy()
    mask2 = labels == 2
    print(f"标签2点数: {np.sum(mask2)}")
    if np.sum(mask2) == 0:
        return points

    # 只用xy坐标聚类
    xy2 = xyz[mask2][:, :2]
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(xy2)
    cluster_labels = db.labels_
    print(f"聚类簇分布: {np.unique(cluster_labels, return_counts=True)}")

    # 估算所有点的法向量
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normal_knn))
    normals = np.asarray(pcd.normals)
    normals2 = normals[mask2]

    # 对每个轨道簇分别筛选
    for clu in np.unique(cluster_labels):
        clu_mask = cluster_labels == clu
        idx = np.where(mask2)[0][clu_mask]  # 获取原数组中的索引
        if clu == -1:
            print(f"簇{clu}（噪声）: {len(idx)} 个点被设为0")
            labels[idx] = 0
            continue
        clu_normals = normals2[clu_mask]
        if clu_normals.shape[0] == 0:
            print(f"簇{clu}为空，跳过")
            continue
        mean_normal = np.mean(clu_normals, axis=0)
        mean_normal /= np.linalg.norm(mean_normal) + 1e-8
        cos_sim = np.dot(clu_normals, mean_normal)
        print(f"簇{clu} cos_sim范围: {cos_sim.min():.3f} ~ {cos_sim.max():.3f}")
        keep_mask = cos_sim > cos_threshold
        print(f"簇{clu} 保留点数: {np.sum(keep_mask)}, 总点数: {len(keep_mask)}")
        labels[idx] = np.where(keep_mask, 2, 0)
    points[:, 3] = labels
    return points

if __name__ == "__main__":
    input_path = "/home/may/data/improve_perfomance/data/predicted/aqc_808_2024-11-06-04-56-27_1730869008128618162.pcd"
    output_path = "/home/may/data/improve_perfomance/data/improved/aqc_808_2024-11-06-04-56-27_1730869008128618162.pcd"
    header, points = read_pcd_ascii(input_path)
    num_label2_before = np.sum(points[:, 3] == 2)
    filtered_points = filter_label2_by_normal_cluster(points, cos_threshold=0.95, dbscan_eps=1, dbscan_min_samples=10)
    num_label2_after = np.sum(filtered_points[:, 3] == 2)
    print(f"处理前标签为2的点数: {num_label2_before}")
    print(f"处理后标签为2的点数: {num_label2_after}")
    write_pcd_ascii(output_path, header, filtered_points)