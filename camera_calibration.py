# # *coding:utf-8 *
# import cv2
# import numpy as np
# import glob
#
# # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
# criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
#
# # 获取标定板角点的位置
#
# CHECKERBOARD = (6, 9)
# objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
# objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
#
# # objp = np.zeros((6 * 9, 3), np.float32)
# # objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
#
# size = (1920, 1920)
# obj_points = []  # 存储3D点
# img_points = []  # 存储2D点
#
# # images = glob.glob("D:/bd/*.bmp")
# images = glob.glob("C:/Users/zyan/Web/CaptureFiles/2021-12-17/*.jpg")
# for fname in images:
#     img = cv2.imread(fname)
#     # cv2.imshow('img', img)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # cv2.imshow('img', gray)
#     # cv2.waitKey(5000)
#
#     size = gray.shape[::-1]
#     ret, corners = cv2.findChessboardCorners(gray, (6, 9), None)
#     print(ret)
#
#     if ret:
#
#         obj_points.append(objp)
#
#         corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
#         # print(corners2)
#         if [corners2]:
#             img_points.append(corners2)
#         else:
#             img_points.append(corners)
#
#         # cv2.drawChessboardCorners(img, (8, 6), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
#         # cv2.namedWindow("img", 0)
#         # cv2.resizeWindow("img", 2880, 1616)
#         # cv2.imshow('img', img)
#         # cv2.waitKey(5000)
#
# print(len(img_points))
#
# calibration_flags = cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
# # calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
# N_OK = len(obj_points)
# # K = np.array([[581.1058307906718, 0.0, 955.5987388116735], [0.0, 579.8976865646564, 974.0212406615763], [0.0, 0.0, 1.0]])
# # D = np.array([[-0.015964497003735242], [-0.002789473611910958], [0.005727838947159351], [-0.0025185770227346576]])
#
# # # print(obj_points.shape)
# # obj_points = np.array([obj_points]*N_OK, dtype=np.float64)
# # obj_points = cv2.Mat(np.reshape(obj_points, (N_OK, -1, 3)), wrap_channels=3)
# # print(obj_points.shape)
# # # print(img_points.shape)
# # img_points = np.asarray(img_points, dtype=np.float64)
# # img_points = cv2.Mat(np.reshape(img_points, (N_OK, -1, 2)), wrap_channels=2)
# # print(img_points.shape)
#
# K = np.eye(3)
# D = np.zeros(4)
# rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
# tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
# # rvecs = np.asarray([[[np.zeros(3).tolist() for i in range(20)]]], dtype='float64').reshape(-1, 1, 1, 3)
# # print(rvecs, rvecs.shape)
# # print(rvecs, rvecs.shape)
# # tvecs = np.asarray([[[np.zeros(3).tolist() for i in range(N_OK)]]], dtype='float64').reshape(-1, 1, 1, 3)
# # obj_points = np.asarray([obj_points], dtype='float64').reshape(-1, 1, 54, 3)
# # img_points = np.asarray([img_points], dtype='float64').reshape(-1, 1, 54, 2)
# print(obj_points)
# rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(obj_points, img_points, size, K, D, rvecs, tvecs, calibration_flags,
#                                                 (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
#
# DIM = gray.shape[:2]
# print("Found " + str(N_OK) + " valid images for calibration")
# print("DIM=" + str(DIM))
# print("K=np.array(" + str(K.tolist()) + ")")
# print("D=np.array(" + str(D.tolist()) + ")")
#
# # obj_points=np.asarray([obj_points],dtype='float64').reshape(-1,1,n,3)
# # img_points=np.asarray([img_points],dtype='float64').reshape(-1,1,n,2)
# # n = number of detected corners in each image
#
#
# # 标定
# # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
# #
# # print("ret:", ret)
# # print("mtx:\n", mtx)  # 内参数矩阵 K
# # print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
# # print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
# # print("tvecs:\n", tvecs)  # 平移向量  # 外参数
# #
# # print("-----------------------------------------------------")
# #
# # import math
# #
# # fx = mtx[0, 0]
# # fy = mtx[1, 1]
# # fx = fx
# # fy = fy
# # print(fx, fy)
# # H = 1920
# # W = 1920
# # pitchVisionAngle = 2 * math.atan2(H, 2 * fx)
# # horizontalVisionAngle = 2 * math.atan2(W, 2 * fy)
# # pitchVisionAngle = pitchVisionAngle / math.pi * 180
# # horizontalVisionAngle = horizontalVisionAngle / math.pi * 180
# # print(pitchVisionAngle, horizontalVisionAngle)
#
# # print(2 * math.atan2(1.7, 2) / math.pi * 180)
# # print(2 * 1.9 * math.tan(81 / 2 / 180 * math.pi))
#
# # 内参矩阵
# # fx 0  u0
# # 0  fy v0
# # 0  0  1
#
# # 前提条件
# #
# #     1. 相机已标定出内参数(水平焦距fx,竖直焦距fy)
# #
# #     2. 相机拍摄图片的大小（高H,宽W）
# #
# # 结果
# #
# #     pitchVisionAngle=2*atan(H/(2*fx));
# #
# #     horizontalVisionAngle=2*atan(W/(2*fy));
#
#
# # [[3.17184400e+04 0.00000000e+00 2.17829120e+03]
# #  [0.00000000e+00 1.66472947e+04 5.70175435e+02]
# # [0.00000000e+00 0.00000000e+00 1.00000000e+00]]


import math
import cv2
import numpy as np
import glob

assert cv2.__version__[0] == '4', 'The fisheye module requires opencv version >= 4.0.0'

CHECKERBOARD = (6, 9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

import os
# for name in os.listdir('C:/Users/zyan/Web/CaptureFiles/2021-12-17/'):
#     img = cv2.imread('C:/Users/zyan/Web/CaptureFiles/2021-12-17/' + name)
#     img = cv2.resize(img, (640, 640))
#     cv2.imwrite("D:/100_100/" + name, img)

# images = glob.glob("D:/bd/*.bmp")
images = glob.glob("C:/Users/zyan/Web/CaptureFiles/2021-12-17/*.jpg")
# images = glob.glob("D:/100_100/*.jpg")
# images = glob.glob("D:/camera_20211221/*.png")
for fname in images:
    img = cv2.imread(fname)
    if _img_shape is None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    print(ret)
    if ret:
        corners /= 3
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
size = gray.shape[::-1]

size = (640, 640)
print(size)
rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        size,
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")

fx = K[0, 0]
fy = K[1, 1]
fx = fx
fy = fy
H = 1944
W = 2592
# pitchVisionAngle = 2 * math.atan2(H / 2, fx)
# horizontalVisionAngle = 2 * math.atan2(W / 2, fy)
# 等距投影模型
pitchVisionAngle = 2 * (H / 2 / fx)
horizontalVisionAngle = 2 * (W / 2 / fy)
pitchVisionAngle = pitchVisionAngle / math.pi * 180
horizontalVisionAngle = horizontalVisionAngle / math.pi * 180
print("俯仰视场角：", pitchVisionAngle, "水平视场角：", horizontalVisionAngle)

# print(2 * math.atan2(1.7, 2) / math.pi * 180)
# print(2 * 1.9 * math.tan(81 / 2 / 180 * math.pi))
