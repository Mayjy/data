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
    cleaned_points = points.copy()
    for i in [0, 2]:  # 只处理 X (0) 和 Z (2)，跳过 Y (1)
        low = np.percentile(cleaned_points[:, i], lower_percentile)
        high = np.percentile(cleaned_points[:, i], upper_percentile)
        mask = (cleaned_points[:, i] >= low) & (cleaned_points[:, i] <= high)
        cleaned_points = cleaned_points[mask]
    return cleaned_points

def get_dynamic_bounds(points):
    min_bound = np.min(points[:, :3], axis=0)
    max_bound = np.max(points[:, :3], axis=0)
    range_ = max_bound - min_bound

    new_min = min_bound.copy()
    new_max = max_bound.copy()

    # 第一个坐标（X轴）保留中间 70%（两边各裁剪 15%）
    new_min[0] = min(-14, min_bound[0] + range_[0] * 0.15)
    new_max[0] = max(14, max_bound[0] - range_[0] * 0.15)

    # 第二个坐标（Y轴）保持不变

    # 第三个坐标（Z轴）下边界取 max(原始 min_z, 30.0)
    if range_[2] > 30:
        new_min[2] = max(min_bound[2]+range_[2]*0.3, 30.0)
    # new_max[2] 保持不变

    return new_min, new_max

def process_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    header, data_start = parse_pcd_header(lines)
    point_lines = lines[data_start:]
    points = np.loadtxt(point_lines, dtype=np.float32)

    # 先去除离群点
    points = remove_outliers_by_percentile(points, 1, 99)

    # 分离坐标和标签
    coords = points[:, :3]
    labels = points[:, 3].astype(int)

    # 计算动态边界
    new_min, new_max = get_dynamic_bounds(points)

    # 标签为0的点根据范围筛选
    mask_label0 = (labels == 0)
    mask_in_bound = np.all((coords >= new_min) & (coords <= new_max), axis=1)
    mask_keep_label0 = mask_label0 & mask_in_bound

    # 标签为1或2的点全部保留
    mask_keep_label12 = (labels == 1) | (labels == 2)

    # 合并保留的点
    mask_keep = mask_keep_label0 | mask_keep_label12
    filtered_points = points[mask_keep]

    # 统计点数少于 5000 的文件，存文件名和点数
    if filtered_points.shape[0] < 5000:
        too_few_points_files.append((os.path.basename(input_file), filtered_points.shape[0]))

    # 修改 header 中 POINTS 和 WIDTH
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
