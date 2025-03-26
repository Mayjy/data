import os
import numpy as np

def check_npy_files(directory):
    # 获取目录下所有的 .npy 文件
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    
    if not npy_files:
        print("该目录下没有 .npy 文件。")
        return
    
    for file in npy_files:
        file_path = os.path.join(directory, file)
        
        try:
            # 读取 .npy 文件
            data = np.load(file_path)
            
            # 统计每个类别的数量
            unique, counts = np.unique(data, return_counts=True)
            label_counts = dict(zip(unique, counts))
            
            # 显示统计结果
            print(f"文件: {file}")
            print(f"数据形状: {data.shape}, 数据类型: {data.dtype}")
            for label in sorted(label_counts.keys()):
                print(f"  标签 {label} 的数量: {label_counts[label]}")
            print("-" * 40)
        except Exception as e:
            print(f"无法读取文件 {file}: {e}")

# 指定要检查的目录
directory_path = "/home/may/my_project/Pointcept/exp/aqc/semseg-pt-v3m1-4-train/result/"
check_npy_files(directory_path)
