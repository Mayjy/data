import re
import matplotlib.pyplot as plt

# 修改为你的 log 文件路径
log_file = '/home/may/my_project/Pointcept/exp/aqc/semseg-pt-v3m1-4-train/train.log'  # 或你实际的路径，比如 'analyze/logs/train.log'

# 保存每轮评估的指标
epochs = []
miou_list, macc_list, allacc_list = [], [], []

# 每类指标，包含 iou 和 acc
class_metrics = {
    0: {'iou': [], 'acc': []},
    1: {'iou': [], 'acc': []},
    2: {'iou': [], 'acc': []},
}

with open(log_file, 'r') as f:
    lines = f.readlines()

epoch = 0  # 当前第几轮评估（验证）

for i, line in enumerate(lines):
    # 检查是否是 Val result 起始行
    if 'Val result' in line:
        epoch += 1
        epochs.append(epoch)

        # 提取总体指标 mIoU/mAcc/allAcc
        match = re.search(r'mIoU/mAcc/allAcc ([\d\.]+)[^\d]*?/([\d\.]+)[^\d]*?/([\d\.]+)', line)
        if match:
            miou_list.append(float(match.group(1).rstrip('.')))
            macc_list.append(float(match.group(2).rstrip('.')))
            allacc_list.append(float(match.group(3).rstrip('.')))

        # 提取每一类的指标
        for cid in range(3):
            if i + cid + 1 < len(lines):
                line_class = lines[i + cid + 1]
                match_class = re.search(rf'Class_{cid}.*?iou/accuracy ([\d\.]+)[^\d]*?/([\d\.]+)', line_class)
                if match_class:
                    class_metrics[cid]['iou'].append(float(match_class.group(1).rstrip('.')))
                    class_metrics[cid]['acc'].append(float(match_class.group(2).rstrip('.')))

# --------------------- 绘图 ---------------------
plt.figure(figsize=(14, 10))

# 整体指标（mIoU / mAcc / allAcc）
plt.subplot(3, 1, 1)
plt.plot(epochs, miou_list, label='mIoU')
plt.plot(epochs, macc_list, label='mAcc')
plt.plot(epochs, allacc_list, label='allAcc')
plt.xlabel('Validation Round')
plt.ylabel('Score')
plt.title('Overall Metrics (mIoU / mAcc / allAcc)')
plt.legend()
plt.grid(True)

# 每类 IoU 曲线
plt.subplot(3, 1, 2)
for cid in class_metrics:
    plt.plot(epochs, class_metrics[cid]['iou'], label=f'Class {cid} IoU')
plt.xlabel('Validation Round')
plt.ylabel('IoU')
plt.title('Per-class IoU')
plt.legend()
plt.grid(True)

# 每类 Accuracy 曲线
plt.subplot(3, 1, 3)
for cid in class_metrics:
    plt.plot(epochs, class_metrics[cid]['acc'], label=f'Class {cid} Accuracy')
plt.xlabel('Validation Round')
plt.ylabel('Accuracy')
plt.title('Per-class Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 如需保存图片，可加：plt.savefig('iou_acc_plot.png')
