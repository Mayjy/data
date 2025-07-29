import os
import numpy as np

# ====== 修改这里的目录路径 ======
input_dir = "process_data/data/dataset"               # 输入文件夹
output_dir = "process_data/data/aftercut_dataset"     # 输出文件夹

too_few_points_files = []  # 用来记录点数少于5000的文件名和对应点数

def parse_pcd_header(lines):
    header = []
    data_start = 0
    for i, line in enumerate(lines):
        header.append(line)
        if line.strip().startswith("DATA"):
            data_start = i + 1
            break
    return header, data_start

def remove_outliers_by_percentile(points, lower_percentile=1, upper_percentile=99):
    # 分离标签为1的点，保留它们，不参与裁剪
    mask_label1 = (points[:, 3] == 1)
    preserved_label1 = points[mask_label1]
    to_clean = points[~mask_label1]

    for i in [0, 2]:  # 只处理 X 和 Z
        low = np.percentile(to_clean[:, i], lower_percentile)
        high = np.percentile(to_clean[:, i], upper_percentile)
        to_clean = to_clean[(to_clean[:, i] >= low) & (to_clean[:, i] <= high)]

    return np.vstack([to_clean, preserved_label1])

def get_dynamic_bounds(points):
    coords = points[:, :3]
    labels = points[:, 3].astype(int)

    # 全局 min/max 和范围
    min_bound = np.min(coords, axis=0)
    max_bound = np.max(coords, axis=0)
    range_ = max_bound - min_bound

    new_min = min_bound.copy()
    new_max = max_bound.copy()

    # ===== X轴：优先用标签2的范围，但加判断 =====
    mask_label2 = (labels == 2)
    label2_count = np.sum(mask_label2)

    if label2_count >= 20:
        x_min = np.min(coords[mask_label2, 0]) - 3
        x_max = np.max(coords[mask_label2, 0]) + 3
        new_min[0] = x_min
        new_max[0] = x_max
    else:
        print(f"⚠️ 标签为2的点太少（仅 {label2_count} 个），X轴使用默认裁剪策略")
        new_min[0] = min(-14, min_bound[0] + range_[0] * 0.15)
        new_max[0] = max(14, max_bound[0] - range_[0] * 0.15)

    # ===== Y轴：不变 =====
    # new_min[1] 和 new_max[1] 保持不变

    # ===== Z轴：使用旧策略 =====
    if range_[2] > 30:
        new_min[2] = max(min_bound[2] + range_[2] * 0.3, 30.0)

    return new_min, new_max

def process_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    header, data_start = parse_pcd_header(lines)
    point_lines = lines[data_start:]

    try:
        points = np.loadtxt(point_lines, dtype=np.float32)
    except Exception as e:
        print(f"⚠️ 读取失败：{input_file}")
        print(f"错误信息：{e}")
        return

    # 去除离群点（标签1不参与）
    points = remove_outliers_by_percentile(points, 1, 99)

    coords = points[:, :3]
    labels = points[:, 3].astype(int)

    # 获取新的裁剪边界
    new_min, new_max = get_dynamic_bounds(points)
    print(f"裁剪范围：X[{new_min[0]:.2f}, {new_max[0]:.2f}], Y[{new_min[1]:.2f}, {new_max[1]:.2f}], Z[{new_min[2]:.2f}, {new_max[2]:.2f}]")

    # 三类处理逻辑
    mask_label1 = (labels == 1)  # 始终保留
    mask_label0 = (labels == 0)
    mask_label2 = (labels == 2)

    mask_in_bound = np.all((coords >= new_min) & (coords <= new_max), axis=1)

    # 标签0和2的点只保留在范围内的
    mask_keep_label0 = mask_label0 & mask_in_bound
    mask_keep_label2 = mask_label2 & mask_in_bound

    mask_keep = mask_label1 | mask_keep_label0 | mask_keep_label2
    filtered_points = points[mask_keep]

    # 点数统计
    if filtered_points.shape[0] < 5000:
        too_few_points_files.append((os.path.basename(input_file), filtered_points.shape[0]))

    # 修改 header
    new_header = []
    for line in header:
        if line.startswith("POINTS"):
            new_header.append(f"POINTS {filtered_points.shape[0]}\n")
        elif line.startswith("WIDTH"):
            new_header.append(f"WIDTH {filtered_points.shape[0]}\n")
        else:
            new_header.append(line)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.writelines(new_header)
        np.savetxt(f, filtered_points, fmt="%.6f %.6f %.6f %d")

    print(f"✅ 已处理：{input_file}")
    print(f"   点数从 {points.shape[0]} -> {filtered_points.shape[0]}")
    print(f"   标签统计：", dict(zip(*np.unique(filtered_points[:, 3], return_counts=True))))

def main():
    for filename in os.listdir(input_dir):
        if filename.endswith(".pcd"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            process_file(input_file, output_file)

    print("✅ 所有文件处理完成！")
    if too_few_points_files:
        print("\n⚠️ 以下文件处理后点数少于 5000：")
        for fname, count in too_few_points_files:
            print(f" - {fname}: {count} points")
    else:
        print("\n🎉 所有文件点数均 >= 5000")

if __name__ == "__main__":
    main()
