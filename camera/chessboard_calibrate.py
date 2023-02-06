import os
import mmcv
import cv2
import numpy as np
from calibrate import get_obj_points, cal_internal_monocular, cal_outside_image_monocular


def chessboard_calib(img_dir, img, checker_board, square_size):
    """
        寻找标定板交点，在此基础上寻找亚像素焦点优化。
        :param img_dir: 内参标定图片集路径，一系列棋盘格图片
        :param img: 外参标定图片路径，一张棋盘格图片
        :param checker_board: 棋盘格内角点数，格式为元组
        :param square_size: 棋盘格大小，单位mm
        :return: 内参矩阵K，畸变系数D，旋转向量rvec，旋转矩阵R，平移向量T
    """
    obj_points = get_obj_points(checker_board, square_size)
    img_list = []
    for img_name in os.listdir(img_dir):
        pic = os.path.join(img_dir, img_name)
        pic = mmcv.imread(pic)
        img_list.append(pic)
    ret, K, D = cal_internal_monocular(obj_points, img_list, checker_board)
    img = mmcv.imread(img)
    ret, rvec, R, T = cal_outside_image_monocular(obj_points, img, checker_board, K, D)
    return K, D, rvec, R, T


K1, D1, rvec1, R1, T1 = chessboard_calib(r"C:\Users\38698\work_space\data\stereo_vision\left", r"C:\Users\38698\work_space\data\stereo_vision\left\left08.jpg", (6, 9), (10, 10))
K2, D2, rvec2, R2, T2 = chessboard_calib(r"C:\Users\38698\work_space\data\stereo_vision\right", r"C:\Users\38698\work_space\data\stereo_vision\right\right08.jpg", (6, 9), (10, 10))
# K1, D1, rvec1, R1, T1 = chessboard_calib(r"C:\Users\38698\Desktop\bottom", r"C:\Users\38698\Desktop\bottom\0_0_xw_white_small_stand_660_MAG_rgb.jpg", (6, 9), (28, 28))
# K2, D2, rvec2, R2, T2 = chessboard_calib(r"C:\Users\38698\Desktop\top", r"C:\Users\38698\Desktop\top\0_0_xw_white_small_stand_660_orbbec_rgb.jpg", (6, 9), (28, 28))
print("T1:", T1)
print("T2", T2)
print("R1:", R1)
print("R2:", R2)

obj_points = get_obj_points((6, 9), (10, 10))
# print(obj_points.shape)
# obj_points[0, 0, 1] = 1
# obj_points[:, :, 2] = 1
print(obj_points.shape)
# obj_points = np.asarray([[[0., 0., 1.]]])
image_points, _ = cv2.projectPoints(obj_points, rvec1, T1, K1, D1)
print(image_points)
image_points, _ = cv2.projectPoints(obj_points, rvec2, T2, K2, D2)
print(image_points)
print(image_points.shape)

print(obj_points)
