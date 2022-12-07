# *coding:utf-8 *

import os
import mmcv

anno_file_path = r'C:\Users\38698\Downloads\raw-data\annotations\instances_train2017.json'

saveFolder = r'C:\Users\38698\Downloads\raw-data\training\label_2'
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

annotations = mmcv.load(anno_file_path)
imgs = annotations  # ["_via_img_metadata"]

objClass = 0
model_name = os.path.abspath('.').split('/')[-1]

# 遍历每个图片
for img in mmcv.track_iter_progress(imgs['images']):
    id = img['id']
    file_name = img['file_name']
    data = ''
    for ann in imgs['annotations']:
        if ann['image_id'] == id:
            category_class = ''
            if ann['category_id'] == 1:
                category_class = 'pig_ear'
            elif ann['category_id'] == 2:
                category_class = 'pig_anus'
            elif ann['category_id'] == 3:
                category_class = 'sow_genitalia'
            truncated = 0
            occluded = 0
            alpha = 0
            left_top_x = ann['bbox'][0]
            left_top_y = ann['bbox'][1]
            right_bottom_x = ann['bbox'][0] + ann['bbox'][2]
            right_bottom_y = ann['bbox'][1] + ann['bbox'][3]
            dimensions_height = 0
            dimensions_width = 0
            dimensions_length = 0
            location_x = 0
            location_y = 0
            location_z = 0
            rotation_y = 0
            data = data + f'{category_class} {truncated} {occluded} {alpha} {left_top_x} {left_top_y} {right_bottom_x} {right_bottom_y} {dimensions_height} {dimensions_width} {dimensions_length} {location_x} {location_y} {location_z} {rotation_y}\n'
    file_name = file_name.split('.')[0]
    file = open(f'{saveFolder}/{file_name}.txt', 'w')
    file.write(data[:-1])
    file.close()
