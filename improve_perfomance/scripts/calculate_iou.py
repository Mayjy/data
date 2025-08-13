import os
import numpy as np

def load_pcd(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # 找到数据起始行
    for i, line in enumerate(lines):
        if line.strip().startswith('DATA'):
            data_start = i + 1
            break
    data = []
    for line in lines[data_start:]:
        if line.strip() == '':
            continue
        vals = line.strip().split()
        if len(vals) == 4:
            data.append([float(vals[0]), float(vals[1]), float(vals[2]), int(float(vals[3]))])
    return np.array(data)

def compute_iou(labels1, labels2, target_label=2):
    mask1 = labels1 == target_label
    mask2 = labels2 == target_label
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union > 0 else 0.0
    return iou, intersection, union

# ...existing code...
def compute_accuracy(labels1, labels2, target_label=2):
    # 只统计原始标签为2的点中，预测也为2的比例
    mask1 = labels1 == target_label
    if mask1.sum() == 0:
        return 0.0, 0, 0
    correct = (labels2[mask1] == target_label).sum()
    total = mask1.sum()
    acc = correct / total
    return acc, correct, total

# ...existing code...

if __name__ == "__main__":
    raw_dir = "/home/may/data/improve_perfomance/data/raw"
    improve_dir = "/home/may/data/improve_perfomance/data/improved"
    files = [f for f in os.listdir(raw_dir) if f.endswith('.pcd')]
    iou_list = []
    acc_list = []
    valid_file_count = 0
    for fname in files:
        raw_path = os.path.join(raw_dir, fname)
        improve_path = os.path.join(improve_dir, fname)
        if not os.path.exists(improve_path):
            continue
        raw_points = load_pcd(raw_path)
        improve_points = load_pcd(improve_path)
        if raw_points.shape[0] != improve_points.shape[0]:
            continue
        if np.sum(improve_points[:, 3] == 2) == 0:
            continue
        iou, intersection, union = compute_iou(raw_points[:, 3], improve_points[:, 3])
        acc, correct, total = compute_accuracy(raw_points[:, 3], improve_points[:, 3])
        print(f"{fname} 标签2 IOU: {iou:.4f} (交集: {intersection}, 并集: {union}) | Accuracy: {acc:.4f} ({correct}/{total})")
        iou_list.append(iou)
        acc_list.append(acc)
        valid_file_count += 1
    if iou_list:
        avg_iou = np.mean(iou_list)
        avg_acc = np.mean(acc_list)
        print(f"\n平均标签2 IOU: {avg_iou:.4f}，平均Accuracy: {avg_acc:.4f}，有效文件数: {valid_file_count}")
    else:
        print("\n没有有效文件计算平均IOU和Accuracy。")