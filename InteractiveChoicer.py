from logging import ERROR
import numpy as np
import cv2
from matplotlib.patches import Polygon
import sys, os
from IPython import embed
from shapely.geometry import LineString
import shapely
import pandas as pd


def eat_region(img, case='通道'):
    if case == '通道':
        img[:80, :, :] = [255, 0, 0]
    elif case == '出猪台':
        img[:100, :, :] = [255, 0, 0]
    else:
        print('Unkown case when cut region %s' % case)
    return img


class RegionChoicer():
    def __init__(self, win_name='select region by click', file_name='no_name',
                 verbose=False, max_num_points=100):
        self.win_name = win_name
        self.region = []
        self.poly_region = None
        self.color = (255, 0, 0)
        self.closed = True
        self.thickness = 2
        self.full_file_name = file_name
        self.file_name = '%s.mp4' % file_name.split('/')[-1].split('.')[0]
        self.verbose = verbose
        self.max_num_points = max_num_points

    def _get_matplotlib_poly_(self):
        self.poly_region = Polygon(np.array(self.region))

    def _mouse_(self, event, x, y, flags, param):
        self.region = list(self.region)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.region.append([x, y])
            if self.verbose:
                print(self.region)
        if event == cv2.EVENT_MBUTTONDOWN:
            if len(self.region) > 0:
                self.region.pop()
            if self.verbose:
                print(self.region)
        self._draws_()  # 每次mouse event之后都要draw一次，更新显示

    def _draws_(self):
        if len(self.region) > self.max_num_points:
            print("%s only contians maximum of %s points, dropping old..." % (
            self.__class__.__name__, self.max_num_points))
            self.region = self.region[len(self.region) - self.max_num_points:]
        # draws:
        draw_frame = self.frame.copy()
        for i in self.region:
            x, y = i[0], i[1]
            cv2.circle(draw_frame, (x, y),
                       max(2, int(draw_frame.shape[0] / 250)), self.color,
                       -1)  # 画点
        cv2.polylines(draw_frame, [np.array(self.region).astype(np.int32)],
                      isClosed=self.closed, color=self.color,
                      thickness=self.thickness, lineType=1)  # 画线
        if self.__class__.__name__ == 'TunnelChoicer' and len(
                self.region) >= 2:
            cv2.arrowedLine(draw_frame, tuple(self.region[0]),
                            tuple(self.region[-1]), color=self.color,
                            thickness=self.thickness)  # 画箭头
        cv2.imshow(self.win_name, draw_frame)

    def set_region(self, frame, region_config_path='tmp/',
                   check_exists_region=False):
        if os.path.exists('%s/%s_%s' % (
        region_config_path, self.file_name.split('/')[-1],
        self.__class__.__name__)):
            print('Region exists', '%s/%s_%s' % (
            region_config_path, self.file_name.split('/')[-1],
            self.__class__.__name__))
            self.load_region(region_config_path)
            if not check_exists_region: return
        self.frame = frame
        cv2.namedWindow(self.win_name, 0)
        cv2.resizeWindow(self.win_name, 1500, 1500)
        cv2.imshow(self.win_name, frame)
        cv2.setMouseCallback(self.win_name, self._mouse_)
        self._draws_()  # draw一次，开始时显示
        while 1:
            key_value = cv2.waitKey(0)
            if key_value == 13:  # enter
                cv2.destroyAllWindows()
                break
            if key_value == 27:  # esc
                cv2.destroyAllWindows()
                sys.exit()
            else:
                print(
                    "Left/Middle(scroll) click to add/cancel point, or press Enter to finish")
        if self.has_region:
            self._get_matplotlib_poly_()
            self.region = np.array(self.region)
            if not os.path.exists(region_config_path):
                os.mkdir(region_config_path)
            save_path = '%s/%s_%s' % (
            region_config_path, self.file_name.split('/')[-1],
            self.__class__.__name__)
            np.savetxt(save_path, self.region)
            print("save at %s" % save_path)

    def load_region(self, region_config_path='tmp/', scale=1):
        region_file = '%s/%s_%s' % (
        region_config_path, self.file_name.split('/')[-1],
        self.__class__.__name__)
        if os.path.exists(region_file):
            self.region = np.loadtxt(region_file).astype(np.int64)
            print("Loading %s, max of it: %s" % (
            region_file, np.max(self.region)))
        else:
            print("%s not found! Falling back using default..." % region_file)
            self.region = np.loadtxt('%s/default_%s' % (
            region_config_path, self.__class__.__name__)).astype(np.int64)
            raise ERROR
        self.region = np.array(self.region * scale, dtype=np.int)
        self._get_matplotlib_poly_()

    def region_control(self, bboxes, segms=None, mode='fastest',
                       speed_up_factor=5):
        keep_index = []
        if len(bboxes) == 0:
            return keep_index
        if mode == 'strict':
            '''
            regions poly 转换成mask，然后跟segms mask判断交叠。  这里指严苛的region，需要卡着猪舍的下边缘。
            '''
            base = np.zeros(segms.shape[1:3], dtype="uint8")  # 黑色底色
            cv2.fillPoly(base, self.region[None, ...].astype(np.int),
                         1)  # 栏位给1
            segms = segms[:, ::speed_up_factor, ::speed_up_factor]
            base = base[::speed_up_factor, ::speed_up_factor]
            base = base.astype(np.int)
            base_segms = base + segms  # 栏位 交叠 逐pigs
            keep_index = (base_segms > 1).sum(
                axis=(1, 2)) > 0  # 交叠矩阵的交叠点个数大于0才说明有覆盖
        if segms is None or mode == 'fastest':  # 不是分割，只有bbox，或者使用最快模式，也是判断bbox中心
            if bboxes.shape[1] != 4 and bboxes.shape[
                1] != 5:  # 不是bboxes,有可能直接给的是中心点
                Xs = bboxes[:, 0]
                Ys = bboxes[:, 1]
            else:
                Xs = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
                Ys = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
            Xs_and_Ys = np.array([Xs, Ys]).T
            shapely_poly = shapely.geometry.Polygon(self.region)
            shapely_centers = shapely.geometry.MultiPoint(Xs_and_Ys)
            for idx, center in enumerate(shapely_centers):
                if shapely_poly.contains(center):
                    keep_index.append(idx)
        return keep_index

    def region_control_fastest(self, bboxes, segms=None, speed_up=False,
                               speed_up_factor=100):
        '''
        bboxes 必须是 左上xy 右下xy
        这个太粗糙了，鱼眼下region是个弧形的，会包裹进去无用的，所以猪舍盘估不能用，同时要求了镜头区域不能斜着，不然更多其他栏位的会进去
        '''
        self.region_max_x = np.max(self.region[:, 0])
        self.region_min_x = np.min(self.region[:, 0])
        self.region_max_y = np.max(self.region[:, 1])
        self.region_min_y = np.min(self.region[:, 1])
        keep_index = []
        if len(bboxes) == 0:
            return keep_index
        if segms is not None:
            print('To be implemented for segm fatest control')
            sys.exit()
        else:
            Xs = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
            Ys = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
            coord_ok = (self.region_min_x <= Xs) & (
                        Xs <= self.region_max_x) & (
                                   self.region_min_y <= Ys) & (
                                   Ys <= self.region_max_y)
            keep_index = np.where(coord_ok)[0].tolist()
        return keep_index

    @property
    def has_region(self):
        return len(self.region) > 0

    def region_expansion(self, extend_factor):
        max_x_point = self.region[np.argmax(self.region, axis=0)[0]].astype(
            np.float32)
        max_y_point = self.region[np.argmax(self.region, axis=0)[1]].astype(
            np.float32)
        min_x_point = self.region[np.argmin(self.region, axis=0)[0]].astype(
            np.float32)
        min_y_point = self.region[np.argmin(self.region, axis=0)[1]].astype(
            np.float32)

        max_x_point_extend = np.array([max_x_point[0] + (
                    max_x_point[0] - min_x_point[0]) * extend_factor,
                                       max_x_point[1]]).astype(np.float32)
        min_x_point_extend = np.array([min_x_point[0] - (
                    max_x_point[0] - min_x_point[0]) * extend_factor,
                                       min_x_point[1]]).astype(np.float32)
        max_y_point_extend = np.array([max_y_point[0], max_y_point[1] + (
                    max_y_point[1] - min_y_point[1]) * extend_factor]).astype(
            np.float32)
        min_y_point_extend = np.array([min_y_point[0], min_y_point[1] - (
                    max_y_point[1] - min_y_point[1]) * extend_factor]).astype(
            np.float32)
        M = cv2.getPerspectiveTransform(
            np.array([max_x_point, max_y_point, min_x_point, min_y_point]),
            np.array(
                [max_x_point_extend, max_y_point_extend, min_x_point_extend,
                 min_y_point_extend]))
        new_region = np.round(
            cv2.perspectiveTransform(np.float32(self.region)[None, :, :], M))[
                     0, :, :].astype(int)
        return new_region


class EntranceChoicer(RegionChoicer):
    def __init__(self, win_name='select Entrance by click',
                 file_name='no_name'):
        super(EntranceChoicer, self).__init__(win_name, file_name)
        self.color = (0, 0, 255)
        self.closed = False
        self.max_num_points = 2


class TunnelChoicer(RegionChoicer):
    def __init__(self, win_name='select Tunnel by click', file_name='no_name'):
        super(TunnelChoicer, self).__init__(win_name, file_name)
        self.color = (0, 255, 0)
        self.closed = False
        self.thickness = 5
        self.max_num_points = 2


class FourPointChoicer(RegionChoicer):
    def __init__(self,
                 win_name='select only 4 points by click for fisheye-rectangle-calibrate',
                 file_name='no_name'):
        super(FourPointChoicer, self).__init__(win_name, file_name)
        self.color = (255, 255, 0)
        self.closed = True
        self.thickness = 5
        self.max_num_points = 4
