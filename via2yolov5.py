# *coding:utf-8 *

import os
import json

anno_file_path = r'D:\data\20220823145459\train\via_region_data.json'

saveFolder = r'D:\data\20220823145459\yolo_train'
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

annotations = json.load(open(anno_file_path, 'r', encoding='UTF-8'))
imgs = annotations  # ["_via_img_metadata"]

WIDTH = 1920
HEIGHT = 1080
objClass = 0

# 遍历每个图片
for imgId in imgs:
    filename = imgs[imgId]['filename']
    imgName = filename.split('.')[0]
    # print('filename:', filename)
    regions = imgs[imgId]['regions']
    # if len(regions) <= 0:
    #     continue

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
