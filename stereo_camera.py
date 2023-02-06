# 实现三维坐标向二维坐标的转换

# import numpy as np
#
# """相机内、外参矩阵"""
#
# # 外参矩阵 (需要改)
# Out = np.mat([
#     [-0.117, -0.992, 0.028, -0.125],
#     [-0.0033, -0.0278, -0.9996, 0.2525],
#     [0.993, -0.1174, 0.00000315, 0.0716],
#     [0, 0, 0, 1]
# ])
#
# # 内参矩阵 (需要改)
# K = np.mat([
#     [610.53, 0, 368.114],
#     [0, 605.93, 223.969],
#     [0, 0, 1]
# ])
#
# """坐标转换"""
# # 打开用于存放世界坐标的txt文件，将其中的以字符串格式保存的世界坐标转换成（Xw， Yw， Zw， 1）的元组格式
# # f = open('database', 'r')
# # database = []
# # for line in f.readlines():
# #     coordinate = line.strip()  # 去掉左右的空格符
# #     coordinate = eval(coordinate)  # 将字符串格式的坐标转换为元组格式
# #     database.append(coordinate)
# # print(database)
#
# database = [(30, 30, 30)]
# world_coordinate_list = []
# for item in database:
#     world_coordinate_part = (item[0], item[1], item[2], 1)
#     world_coordinate_list.append(world_coordinate_part)
# print(world_coordinate_list)
#
# pixel_coordinate_list = []
#
# for item in world_coordinate_list:
#     world_coordinate = np.mat([
#         [item[0]],
#         [item[1]],
#         [item[2]],
#         [item[3]]
#     ])
#     print(f'世界坐标为：\n{world_coordinate}')
#     # print(type(world_coordinate))
#
#     # 世界坐标系转换为相加坐标系 （Xw,Yw,Zw）--> (Xc,Yc,Zc)
#     camera_coordinate = Out * world_coordinate
#     print(f'相机坐标为：\n{camera_coordinate}')
#     Zc = float(camera_coordinate[2])
#     print(f'Zc={Zc}')
#     raise TypeError
#     # 相机坐标系转图像坐标系 (Xc,Yc,Zc) --> (x, y)  下边的f改为焦距
#     focal_length = np.mat([
#         [f, 0, 0, 0],
#         [0, f, 0, 0],
#         [0, 0, 1, 0]
#     ])
#     image_coordinate = (focal_length * camera_coordinate) / Zc
#     print(f'图像坐标为：\n{image_coordinate}')
#
#     # 图像坐标系转换为像素坐标系
#     pixel_coordinate = K * image_coordinate
#     print(f'像素坐标为：\n{pixel_coordinate}')
#     pixel_coordinate_list.append(pixel_coordinate)
#     print('---------------------分割线--------------------------------')
#
# print(pixel_coordinate_list)
# f = open("result.txt", "w", encoding="utf-8")
# for item in pixel_coordinate_list:
#     f.write(str(item) + '\n')
#     f.write('------------分割线-----------------' + '\n')
# f.close()

import cv2
import os
import numpy as np

# opencv sample
leftpath = 'C:/Users/38698/work_space/data/stereo_vision/left'
rightpath = 'C:/Users/38698/work_space/data/stereo_vision/right'

# leftpath = 'C:/Users/38698/Desktop/bottom'
# rightpath = 'C:/Users/38698/Desktop/top'
CHECKERBOARD = (6, 9)  # 棋盘格内角点数
square_size = (10, 10)  # 棋盘格大小，单位mm

# square_size = (28, 28)  # 棋盘格大小，单位mm

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
imgpoints_l = []  # 存放左图像坐标系下角点位置
imgpoints_r = []  # 存放左图像坐标系下角点位置
objpoints = []  # 存放世界坐标系下角点位置
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp[0, :, 0] *= square_size[0]
objp[0, :, 1] *= square_size[1]

for ii in os.listdir(leftpath):
    img_l = cv2.imread(os.path.join(leftpath, ii).replace("\\", "/"))
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    # img_r = cv2.imread(os.path.join(rightpath, ii.replace("MAG", "orbbec")).replace("\\", "/"))
    # opencv sample
    img_r = cv2.imread(os.path.join(rightpath, ii.replace("left", "right")).replace("\\", "/"))

    # print(ii)
    # img_r = cv2.imread(os.path.join(rightpath, ii).replace("\\", "/"))
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD)  # 检测棋盘格内角点
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD)
    if ret_l and ret_r:
        objpoints.append(objp)
        corners2_l = cv2.cornerSubPix(gray_l, corners_l, (5, 5), (-1, -1), criteria)
        imgpoints_l.append(corners2_l)
        corners2_r = cv2.cornerSubPix(gray_r, corners_r, (5, 5), (-1, -1), criteria)
        imgpoints_r.append(corners2_r)
        # img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
        # cv2.imwrite('./ChessboardCornersimg.jpg', img)
ret, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, gray_l.shape[::-1], None,
                                                           None)  # 先分别做单目标定
ret, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, gray_r.shape[::-1], None, None)
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv2.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r, gray_l.shape[::-1])  # 再做双目标定

print("stereoCalibrate : \n")
print("Camera matrix left : \n")
print(cameraMatrix1)
print("distCoeffs left  : \n")
print(distCoeffs1)
print("cameraMatrix right : \n")
print(cameraMatrix2)
print("distCoeffs right : \n")
print(distCoeffs2)
print("R : \n")
print(R)
print("T : \n")
print(T)
print("E : \n")
print(E)
print("F : \n")
print(F)


def cat2images(limg, rimg):
    HEIGHT = limg.shape[0]
    WIDTH = limg.shape[1]
    imgcat = np.zeros((HEIGHT, WIDTH * 2 + 20, 3))
    imgcat[:, :WIDTH, :] = limg
    imgcat[:, -WIDTH:, :] = rimg
    for i in range(int(HEIGHT / 32)):
        imgcat[i * 32, :, :] = 255
    return imgcat


# opencv sample
left_image = cv2.imread("C:/Users/38698/work_space/data/stereo_vision/left/left08.jpg")
right_image = cv2.imread("C:/Users/38698/work_space/data/stereo_vision/right/right08.jpg")

# left_image = cv2.imread('C:/Users/38698/Desktop/bottom/0_0_xw_white_small_stand_670_MAG_rgb.jpg')
# right_image = cv2.imread('C:/Users/38698/Desktop/top/0_0_xw_white_small_stand_670_orbbec_rgb.jpg')

imgcat_source = cat2images(left_image, right_image)
HEIGHT = left_image.shape[0]
WIDTH = left_image.shape[1]
cv2.imwrite('imgcat_source.jpg', imgcat_source)

camera_matrix0 = cameraMatrix1

distortion0 = distCoeffs1

camera_matrix1 = cameraMatrix2
distortion1 = distCoeffs2

# R, _ = cv2.Rodrigues(cv2.RQDecomp3x3(R)[0])
R = R
T = T
print("RRRRRRRRRRRRRRRRRR:", R)
print(T)
tran_ma = np.hstack((R, T))
tran_ma = np.vstack((tran_ma, np.asarray([0, 0, 0, 1])))
print(tran_ma)
print(tran_ma.shape)
# [[ 0.9999854   0.00376799  0.00387441]
#  [-0.00374142  0.99996958 -0.00684332]
#  [-0.00390008  0.00682873  0.99996908]]
# R 旋转矩阵
# P 投影矩阵 内参·外参
(R_l, R_r, P_l, P_r, Q, validPixROI1, validPixROI2) = \
    cv2.stereoRectify(camera_matrix0, distortion0, camera_matrix1, distortion1, np.array([WIDTH, HEIGHT]), R, T)  # 计算旋转矩阵和投影矩阵
print(P_l.shape)
print('!!!!!!!!!!', (R_l, R_r, P_l, P_r, Q, validPixROI1, validPixROI2))
(map1, map2) = \
    cv2.initUndistortRectifyMap(camera_matrix0, distortion0, R_l, P_l, np.array([WIDTH, HEIGHT]),
                                cv2.CV_32FC1)  # 计算校正查找映射表
print(map1[80, 522])
print(map2[80, 522])
rect_left_image = cv2.remap(left_image, map1, map2, cv2.INTER_CUBIC)  # 重映射
# rect_left_image = cv2.remap(left_image, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)  # 重映射
cv2.imwrite('imgcat_left.jpg', rect_left_image)
# 左右图需要分别计算校正查找映射表以及重映射 (x, y)
(map1, map2) = \
    cv2.initUndistortRectifyMap(camera_matrix1, distortion1, R_r, P_r, np.array([WIDTH, HEIGHT]), cv2.CV_32FC1)
print(map1[80, 378])
print(map2[80, 378])
rect_right_image = cv2.remap(right_image, map1, map2, cv2.INTER_CUBIC)
# rect_right_image = cv2.remap(right_image, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
cv2.imwrite('imgcat_right.jpg', rect_right_image)
imgcat_out = cat2images(rect_left_image, rect_right_image)
cv2.imwrite('imgcat_out.jpg', imgcat_out)

# left08 棋盘右上角黑格(511, 81) right08(364, 87)
# 矫正后             (522, 80)         (378, 80)

a = np.asmatrix(np.asarray([[320, 240, 1]]).reshape(3, 1))
# b = np.asmatrix(np.asarray([[163, 253, 1]]).reshape(3, 1))
#
# a = np.asmatrix(np.asarray([[511, 80, 1]]).reshape(3, 1))
# b = np.asmatrix(np.asarray([[364, 87, 1]]).reshape(3, 1))
#
# a = np.asmatrix(np.asarray([[383, 223, 1]]).reshape(3, 1))

# print((R_l * a + T))
# print((R_r * b + T))
# c = np.asmatrix([[474.53803511],
#                  [-23.54139756],
#                  [-95.93967479],
#                  [1]])
# d = np.asmatrix([[329.71603007],
#                  [90.20568062],
#                  [-15.0216635],
#                  [1]])
# print(c.shape)
# print(P_l * c)
# print(P_r * d)


def estimate_depth(left_path, right_path, show=True):
    # 读取左右两张图像
    img_left = cv2.imread(left_path, 0)
    img_right = cv2.imread(right_path, 0)
    height, width = img_left.shape[:2]

    # 初始化stereo block match对象
    stereo = cv2.StereoBM_create(numDisparities=0, blockSize=5)

    # 获取视差图
    disparity = stereo.compute(img_left, img_right)

    if show:
        # 将视差图归一化
        # min_val = disparity.min()
        # max_val = disparity.max()
        # disparity = np.uint8(6400 * (disparity - min_val) / (max_val - min_val))

        # 显示视差图
        cv2.imshow('disparity image', disparity)
        cv2.imwrite('disparity.jpg', disparity)
        cv2.waitKey(0)

    return disparity


disparity = estimate_depth('imgcat_left.jpg', 'imgcat_right.jpg')
points = cv2.reprojectImageTo3D(disparity, Q)
left_pixel_coords = np.column_stack((points[..., 0], points[..., 1], points[..., 2]))
right_pixel_coords = np.column_stack((points[..., 0] - disparity, points[..., 1], points[..., 2]))
print(left_pixel_coords)
print(left_pixel_coords.shape)
print(left_pixel_coords[0, 0])
print(right_pixel_coords[0, 0])
# print(points)
# print(points.shape)
# print(disparity.shape)
# print(disparity)
# print(disparity[95, 476])
# print(disparity[95, 334])
# 476 95
# 334 95

# 381 93
# print(disparity.min())
# print(disparity.min().shape)


# import open3d as o3d
# import numpy as np

# print("->正在加载点云... ")
# pcd = o3d.io.read_point_cloud("test.pcd")
# print(pcd)
#
# print("->正在保存点云")
# points = np.random.randn(35947, 3)
# points = np.random.randint(-1000, 1000, size=(480*640, 3))
# print(points.reshape(-1, 3))
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
# # pcd.colors = o3d.utility.Vector3dVector(np.abs(points.reshape(-1, 3)) / np.max(np.abs(points.reshape(-1, 3))))
# # o3d.io.write_point_cloud("D:/workspace/work_scripts/test.pcd", pcd)	# 默认false，保存为Binarty；True 保存为ASICC形式
# print(pcd)
#
# # pcd = o3d.io.read_point_cloud("D:/workspace/work_scripts/test.pcd")
# pcd.paint_uniform_color([0, 0, 1])#指定显示为蓝色
# # print(pcd)
# # # colors = np.array(pcd.colors)
# # colors[inliers] = [0, 0, 1]#平面内的点设置为蓝色
# # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
#
# o3d.visualization.draw_geometries([pcd], width=640, height=480)
#
# vis = o3d.visualization.Visualizer()
# vis.create_window()	#创建窗口
# render_option: o3d.visualization.RenderOption = vis.get_render_option()	#设置点云渲染参数
# render_option.background_color = np.array([0, 0, 0])	#设置背景色（这里为黑色）
# render_option.point_size = 2.0	#设置渲染点的大小
# vis.add_geometry(pcd)	#添加点云
# vis.run()

# 388 71
# h = 134
# camera_l_pix = a
# X = np.linalg.inv(camera_matrix0).dot(camera_l_pix) * h
# # X0 = (camera_l_pix[0] - camera_matrix0[0, 2]) / camera_matrix0[0, 0]
# # Y0 = (camera_l_pix[1] - camera_matrix0[1, 2]) / camera_matrix0[1, 1]
# X = np.vstack((X, np.asarray(1)))
# # X = np.asarray([X0, Y0, 1, 1]).reshape(4, 1)
# print("X:", X)
#
# # 使用外参矩阵把点转换到右相机坐标系中
# X_r = tran_ma.dot(X) / h
# # X_r = E.dot(X)
# X_r = X_r[:3]
# print(X_r)
# print(X_r.shape)
# # X_r = np.asarray([0, 0, 1])
# # 使用右相机内参矩阵把点转换成像素坐标
# x_r = camera_matrix1.dot(X_r)
# # x_r = x_r / x_r[-1]
# print(x_r)
# [327.5862056  248.88223545   1.        ]

# print("左侧相机坐标系", np.matrix(camera_matrix0).I * a)
# print("右侧相机坐标系", R * (np.matrix(camera_matrix0).I * a) + T)
# print("右侧像素坐标系", camera_matrix1 * ((R * (np.matrix(camera_matrix0).I * a) + T) / 1.14125625))
#
# a = np.asarray([[364, 87, 1]]).reshape(3, 1)
# print("右侧相机坐标系", np.matrix(camera_matrix1).I * a)
# print("左侧相机坐标系", R * (np.matrix(camera_matrix1).I * a) + T)
# print("左侧像素坐标系", camera_matrix0 * ((R * (np.matrix(camera_matrix1).I * a) + T) / 1.14212534))
