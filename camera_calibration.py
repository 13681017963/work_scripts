# *coding:utf-8 *
import os
import math
import cv2
import numpy as np
import glob

assert cv2.__version__[0] == '4', 'The fisheye module requires ' \
                                  'opencv version >= 4.0.0 '

CHECKERBOARD = (6, 9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = \
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
    cv2.fisheye.CALIB_CHECK_COND + \
    cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob("C:/Users/zyan/Web/CaptureFiles/2021-12-17/*.jpg")
# images = glob.glob("D:/100_100/*.jpg")
# images = glob.glob("D:/camera_20211221/*.png")
for fname in images:
    img = cv2.imread(fname)
    if _img_shape is None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share " \
                                            "the same size."
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_FAST_CHECK +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )
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
