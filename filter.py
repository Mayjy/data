import os
import shutil

# 设置数据集根目录
root_dir = "/home/may/data/unloading"  # 你的 loading 目录
output_dir = "/home/may/data/output"  # 目标文件夹

# 确保目标文件夹存在
os.makedirs(output_dir, exist_ok=True)

# 遍历所有子文件夹（按时间戳命名的）
for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)
    images_path = os.path.join(subfolder_path, "images")
    pointclouds_path = os.path.join(subfolder_path, "pointclouds")

    if not os.path.isdir(images_path) or not os.path.isdir(pointclouds_path):
        continue  # 跳过非文件夹的内容

    # 读取 images 目录中的 JSON 文件（去掉后缀）
    json_files = {os.path.splitext(f)[0] for f in os.listdir(images_path) if f.endswith(".json")}

    # 处理每个 JSON 对应的图片和点云
    for file_name in json_files:
        img_file = os.path.join(images_path, file_name + ".jpg")  # 你的图片格式可能是 .png
        json_file = os.path.join(images_path, file_name + ".json")
        pcd_file = os.path.join(pointclouds_path, file_name + ".pcd")  # 可能是 .bin 或其他格式
        bin_file = os.path.join(pointclouds_path, file_name + ".bin")

        # 确保图片和点云至少有一个存在
        if os.path.exists(img_file) and (os.path.exists(pcd_file) or os.path.exists(bin_file)):
            # 复制文件到目标目录
            shutil.copy(img_file, output_dir)
            shutil.copy(json_file, output_dir)
            if os.path.exists(pcd_file):
                shutil.copy(pcd_file, output_dir)
            else:
                shutil.copy(bin_file, output_dir)
        else:
            print(f"缺少图片或点云，跳过: {file_name} in {subfolder}")

print("✅ 筛选完成！")