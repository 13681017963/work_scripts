import torch
import numpy as np

# 加载模型
model = ...  # 实例化模型
model.load_state_dict(torch.load("last.pt"))
model.eval()

# 读取验证集数据
val_dataset = ...  # 实例化验证集


# 定义IoU计算函数
def bbox_iou(bbox1, bbox2):
    """
    计算两个矩形框的IoU
    :param bbox1: [x1, y1, x2, y2]
    :param bbox2: [x1, y1, x2, y2]
    :return: IoU
    """
    # 计算交集面积
    inter_area = (torch.min(bbox1[:, 2:], bbox2[2:]) - torch.max(bbox1[:, :2], bbox2[:2])).clamp(0).prod(1)
    # 计算总面积
    bbox1_area = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    total_area = bbox1_area + bbox2_area - inter_area
    # 计算IoU
    iou = inter_area / total_area
    return iou


# 定义mAP计算函数
def compute_map(model, val_dataset):
    # 准备存储结果的数组
    APs = []
    for i in range(val_dataset.num_classes):
        APs.append([])

    # 遍历每个样本
    for data in val_dataset:
        # 获取图像和标注
        images = ...
        labels = ...

        # 运行模型，得到预测边界框
        with torch.no_grad():
            pred_bboxes = model(images)

        # 遍历每个目标类别
        for c in range(val_dataset.num_classes):
            # 提取该类别的真实边界框和预测边界框
            true_bboxes = labels[labels[:, -1] == c, :-1]
            pred_bboxes_class = pred_bboxes[pred_bboxes[:, -1] == c, :-1]

            # 如果该类别没有任何目标，则跳过
            if true_bboxes.shape[0] == 0:
                continue

            # 计算预测边界框的得分（IoU）
            scores = np.zeros((pred_bboxes_class.shape[0]))
            for j in range(pred_bboxes_class.shape[0]):
                ious = np.array([bbox_iou(pred_bboxes_class[j], tb) for tb in true_bboxes])
                scores[j] = ious.max

                # 对得分进行排序，从高到低
                order = np.argsort(-scores)
                pred_bboxes_class = pred_bboxes_class[order, :]

                # 计算平均精度（mAP）
                tp = np.zeros((pred_bboxes_class.shape[0]))
                fp = np.zeros((pred_bboxes_class.shape[0]))
                for j in range(pred_bboxes_class.shape[0]):
                    if scores[order[j]] >= 0.5:
                        if np.any(bbox_iou(pred_bboxes_class[j], true_bboxes) >= 0.5):
                            tp[j] = 1
                            fp[j] = 0
                        else:
                            tp[j] = 0
                            fp[j] = 1
                    else:
                        break
                cumulative_tp = np.cumsum(tp)
                cumulative_fp = np.cumsum(fp)
                recall = cumulative_tp / true_bboxes.shape[0]
                precision = cumulative_tp / (cumulative_fp + cumulative_tp + 1e-16)

                # 计算平均精度的不同阈值的值
                for th in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
                    idx = np.where(recall >= th)[0]
                    if idx.shape[0] == 0:
                        APs[c].append(0)
                    else:
                        APs[c].append(np.max(precision[idx]))

        # 计算总mAP
        mean_AP = []
        for i in range(val_dataset.num_classes):
            mean_AP.append(np.mean(APs[i]))
        mean_AP = np.mean(mean_AP)

        return mean_AP, APs


    mean_AP, class_APs = compute_map(model, val_dataset)
    print("mAP50-95:", mean_AP)
