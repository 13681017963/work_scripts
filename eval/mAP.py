import os
import sys
import cv2
import mmcv
import copy
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from multiprocessing import freeze_support

# device = "cuda"
# # 加载模型
# model = torch.load("n.pt")
# # model.load_state_dict(torch.load("n.pt"))
# model.eval()
# model.to(device)

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("n.pt")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
# metrics = model.val(data="coco128.yaml")  # evaluate model performance on the validation set
#
# results = model(r"D:\data\yolo_v4\val\images\123_1619434050941.png")  # predict on an image

# success = model.export(format="onnx")  # export the model to ONNX format


# model = YOLO("n.pt")
# img = mmcv.imread(r"D:\data\yolo_v4\val\images\123_1619434050941.png")
# inputs = [img]  # list of np arrays
# results = model(inputs)  # List of Results objects

# print(results)
class DataSet:
    def __init__(self, path, num_classes):
        self.path = path
        self.num_classes = num_classes


def yolov8_bbox(model_path, val_dataset):
    model = YOLO(model_path)
    inputs = []
    outputs = []
    for name in os.listdir(os.path.join(val_dataset.path, "images")):
        img_dir = os.path.join(val_dataset.path, "images", name)
        img = mmcv.imread(img_dir)
        inputs.append(img)
        results = model(inputs)
        for result in results:
            boxes = result.boxes.boxes.to('cpu').numpy().astype(np.int64)  # Boxes object for bbox outputs
            # print(boxes)
            outputs.append(boxes)
            # for box in boxes:
            #     print(box)

            # cv2.rectangle(img, box[:2], box[2:4], (0, 0, 255), 2)
            # bb = np.asarray(box[0, :5])
            # print(bb)
        # mmcv.imshow(img)
    np.save('./dd.npy', outputs)
    return outputs


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


def yolo_bbox(box, width, height):
    centerX = box[:, 0] * width
    centerY = box[:, 1] * height
    w = box[:, 2] * width
    h = box[:, 3] * height
    x1 = centerX - (w / 2)
    y1 = centerY - (h / 2)
    x2 = centerX + (w / 2)
    y2 = centerY + (h / 2)
    return np.around(np.hstack((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1))), 0).astype(np.int64)


# 定义mAP计算函数
def compute_map(pred_bboxes, val_dataset):
    # 准备存储结果的数组
    APs = []
    for i in range(val_dataset.num_classes):
        APs.append([])

    # 遍历每个样本
    for data in os.listdir(os.path.join(val_dataset.path, "labels")):
        # 获取图像和标注
        images = os.path.join(val_dataset.path, "images", data.replace(".txt", ".png"))
        labels = os.path.join(val_dataset.path, "labels", data)
        img = mmcv.imread(images)
        height = img.shape[0]
        width = img.shape[1]
        print(labels)
        labels = np.loadtxt(labels, dtype=np.float32)
        print(labels)
        print(labels.shape)
        # 遍历每个目标类别
        for c in range(val_dataset.num_classes):
            # 提取该类别的真实边界框和预测边界框
            true_bboxes = labels[labels[:, 0] == c, 1:]
            true_bboxes = yolo_bbox(true_bboxes, width, height)
            print(true_bboxes)
            print(true_bboxes.shape)
            # for bb in true_bboxes:
            #     cv2.rectangle(img, bb[:2], bb[2:4], (0, 0, 255), 2)
            # mmcv.imshow(img)
            break
        break
        #     pred_bboxes_class = pred_bboxes[pred_bboxes[:, -1] == c, :-1]
        #
        #     # 如果该类别没有任何目标，则跳过
        #     if true_bboxes.shape[0] == 0:
        #         continue
        #
        #     # 计算预测边界框的得分（IoU）
        #     scores = np.zeros((pred_bboxes_class.shape[0]))
        #     for j in range(pred_bboxes_class.shape[0]):
        #         ious = np.array([bbox_iou(pred_bboxes_class[j], tb) for tb in true_bboxes])
        #         scores[j] = ious.max
        #
        #         # 对得分进行排序，从高到低
        #         order = np.argsort(-scores)
        #         pred_bboxes_class = pred_bboxes_class[order, :]
        #
        #         # 计算平均精度（mAP）
        #         tp = np.zeros((pred_bboxes_class.shape[0]))
        #         fp = np.zeros((pred_bboxes_class.shape[0]))
        #         for j in range(pred_bboxes_class.shape[0]):
        #             if scores[order[j]] >= 0.5:
        #                 if np.any(bbox_iou(pred_bboxes_class[j], true_bboxes) >= 0.5):
        #                     tp[j] = 1
        #                     fp[j] = 0
        #                 else:
        #                     tp[j] = 0
        #                     fp[j] = 1
        #             else:
        #                 break
        #         cumulative_tp = np.cumsum(tp)
        #         cumulative_fp = np.cumsum(fp)
        #         recall = cumulative_tp / true_bboxes.shape[0]
        #         precision = cumulative_tp / (cumulative_fp + cumulative_tp + 1e-16)
        #
        #         # 计算平均精度的不同阈值的值
        #         # [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        #         for th in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        #             idx = np.where(recall >= th)[0]
        #             if idx.shape[0] == 0:
        #                 APs[c].append(0)
        #             else:
        #                 APs[c].append(np.max(precision[idx]))
        #
        # # 计算总mAP
        # mean_AP = []
        # for i in range(val_dataset.num_classes):
        #     mean_AP.append(np.mean(APs[i]))
        # mean_AP = np.mean(mean_AP)
        #
        # return mean_AP, APs


    # mean_AP, class_APs = compute_map(model, val_dataset)
    # print("mAP50-95:", mean_AP)


if __name__ == '__main__':
    # freeze_support()
    # model = YOLO("l.pt")  # load a pretrained model (recommended for training)
    # metrics = model.val(data="coco128.yaml")  # evaluate model performance on the validation set

    # 读取验证集数据
    val_dataset = DataSet(r'D:\data\yolov8\val', 1)  # 实例化验证集
    model_path = './n.pt'
    pred_bboxes = yolov8_bbox(model_path, val_dataset)
    pred_bboxes = None

    compute_map(pred_bboxes, val_dataset)

