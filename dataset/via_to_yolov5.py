# *coding:utf-8 *

import os
import cv2
import json

anno_file_path = r'C:\Users\38698\work_space\data\20220929101743\val\via_region_data.json'

saveFolder = r'C:\Users\38698\work_space\data\showroom\labels\val'
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

annotations = json.load(open(anno_file_path, 'r', encoding='UTF-8'))
imgs = annotations  # ["_via_img_metadata"]

objClass = 0
model_name = os.path.abspath('..').split('/')[-1]

# 遍历每个图片
for imgId in imgs:
    filename = imgs[imgId]['filename']
    imgName = filename.split('.')[0]
    # print('filename:', filename)
    regions = imgs[imgId]['regions']
    # if len(regions) <= 0:
    #     continue
    img_dir = ""
    for idx, i in enumerate(anno_file_path.split('\\')):
        if idx != len(anno_file_path.split('\\')) - 1:
            img_dir += i
            img_dir += '/'
    img_dir += filename
    img = cv2.imread(img_dir)
    WIDTH = img.shape[1]
    HEIGHT = img.shape[0]
    data = ''
    # 遍历每个区域
    for region in regions:
        # print(region)
        shape = region['shape_attributes']
        x = shape['all_points_x']
        y = shape['all_points_y']
        # boxW = shape['width']
        # boxH = shape['height']

        # minX = int(x)
        # minY = int(y)
        # maxX = int(x + boxW)
        # maxY = int(y + boxH)
        minX = min(x)
        minY = min(y)
        maxX = max(x)
        maxY = max(y)

        centerX = round((minX + maxX) / 2 / WIDTH, 6)
        centerY = round((minY + maxY) / 2 / HEIGHT, 6)
        w = round((maxX - minX) / WIDTH, 6)
        h = round((maxY - minY) / HEIGHT, 6)

        data = data + f'{objClass} {centerX} {centerY} {w} {h}\n'
    file = open(f'{saveFolder}/{imgName}.txt', 'w')
    file.write(data[:-1])
    # file.write(data)
    file.close()
