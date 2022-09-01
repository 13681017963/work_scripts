# *coding:utf-8 *
import os
import cv2
import numpy as np
from InteractiveChoicer import RegionChoicer


def fill_mask_outside(image, poly):
    mask = ~np.zeros_like(image)
    random_mask = (np.random.random(image.shape) * 255).astype(np.uint8)
    mask = ~mask
    cv2.fillPoly(mask, [poly], (255, 255, 255))
    cv2.fillPoly(random_mask, [poly], (255, 255, 255))
    image = cv2.bitwise_and(image, mask)
    left_top = np.min(poly, axis=0)
    right_bottom = np.max(poly, axis=0)
    return image[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]


class MaskChoicer(RegionChoicer):
    def __init__(self,
                 win_name='select only one mask by click for picture',
                 file_name='no_name'):
        super(MaskChoicer, self).__init__(win_name, file_name)
        self.color = (255, 255, 0)
        self.closed = True
        self.thickness = 1
        self.max_num_points = 1e8

    def set_region(self, frame, region_config_path='tmp/',
                   check_exists_region=False, enter_list=[ord('1'), ord('2')], class_name=['stand', 'crouch']):
        cnt = 0
        while True:
            self.frame = frame
            cv2.namedWindow(self.win_name, 0)
            cv2.resizeWindow(self.win_name, 1500, 1500)
            cv2.imshow(self.win_name, frame)
            cv2.setMouseCallback(self.win_name, self._mouse_)
            self._draws_()  # draw一次，开始时显示
            key_value = cv2.waitKey(0)
            if key_value in enter_list:  # enter
                if self.has_region:
                    self._get_matplotlib_poly_()
                    self.region = np.array(self.region)
                    if not os.path.exists(region_config_path + class_name[enter_list.index(key_value)]):
                        os.makedirs(region_config_path + class_name[enter_list.index(key_value)])
                    save_path = '%s/%s/%s_%d.jpg' % (region_config_path, class_name[enter_list.index(key_value)], self.file_name.split('/')[-1].split('.')[0], cnt)
                    img = fill_mask_outside(frame.copy(), np.asarray(self.region))
                    cv2.imwrite(save_path, img)
                    print("save at %s" % save_path)
                    cnt += 1
                    self.region = []
                    cv2.imshow(self.win_name, frame)
                continue
            if key_value == 27:  # esc
                cv2.destroyAllWindows()
                break
            else:
                print("Left/Middle(scroll) click to add/cancel point, or press Enter to finish")


img_dir = 'D:/data/zz/zzz/data/'
for img_name in os.listdir(img_dir):
    MaskChoicer(file_name=img_name.split('.')[0]).set_region(cv2.imread(img_dir + img_name))
