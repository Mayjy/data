import json
import cv2
import numpy as np
from PIL import Image
import os
import shutil

def process_image(image_path, json_path, output_path):
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist.")
        return
    
    # 检查JSON文件是否存在
    if not os.path.exists(json_path):
        print(f"Error: JSON file {json_path} does not exist.")
        return
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to open image file {image_path}")
        return
    height, width, _ = image.shape
    
    # 创建一个白色背景
    mask = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # 读取JSON标签
    with open(json_path, 'r') as f:
        labels = json.load(f)
    
    # 颜色映射
    color_map = {
        "spreader": (0, 255, 0),  # 绿色 (BGR)
        "cell_guide": (255, 0, 0)  # 蓝色 (BGR)
    }
    
    # 遍历 shapes 解析多边形
    for shape in labels.get("shapes", []):
        label = shape.get("label", "")
        points = shape.get("points", [])
        
        if label in color_map and len(points) > 2:  # 确保是多边形
            polygon = np.array(points, dtype=np.int32)  # 转换为 NumPy 数组
            cv2.fillPoly(mask, [polygon], color_map[label])  # 填充多边形
    
    # 保存处理后的图像
    result = Image.fromarray(mask)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path)
    print(f"Processed image saved to {output_path}")

def main():
    # 输入和输出目录
    input_dir = "/home/may/data/rawdata"  # 遍历的目录
    processed_dir = "/home/may/data/processed"  # 保存处理后的内容
    
    os.makedirs(processed_dir, exist_ok=True)

    # 遍历 output 文件夹中的所有文件
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        
        # 跳过非 JSON 或图片的文件
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            # 查找对应的 JSON 文件
            base_name = os.path.splitext(file_name)[0]
            json_file = os.path.join(input_dir, base_name + ".json")
            
            if os.path.exists(json_file):
                # 处理图片并保存到 processed 文件夹
                processed_image_path = os.path.join(processed_dir, file_name)
                process_image(file_path, json_file, processed_image_path)
            else:
                print(f"JSON file not found for {file_name}, skipping.")
        
        # 如果是点云文件（.pcd 或 .bin），直接复制到 processed 文件夹
        elif file_name.endswith(".pcd") or file_name.endswith(".bin"):
            shutil.copy(file_path, processed_dir)
            print(f"Copied point cloud file: {file_name}")

    print("✅ 所有文件处理完成！")

if __name__ == "__main__":
    main()