import mmcv
import cv2
import copy
import numpy as np
import xml.etree.ElementTree as ET


annotations_file = r'C:\Users\38698\work_space\data\ear\annotations.xml'
yolov5_dataset = r'C:\Users\38698\work_space\data\yolov5_ear'


mytree = ET.parse(annotations_file)
root = mytree.getroot()
saveFolder = yolov5_dataset
dic = {}
for idx, img in enumerate(root.findall('image')):
    dir = 'C:/Users/38698/work_space/data/ear/images/Train/' + img.attrib["name"]
    imm = cv2.imread(dir)
    # imm[0, 0, :] = 1
    # imm[0, 1, :] = 12
    # , [int(cv2.IMWRITE_JPEG_QUALITY), 30]
    cv2.imwrite('C:/Users/38698/work_space/data/ear/val/' + img.attrib["name"].split('/')[-1].split('.')[0] + '_0.jpg', imm)
    img_name = img.attrib["name"].split('/')[-1].split('.')[0] + '_0.jpg'
    dic[img_name] = {"filename": img_name, "regions": [], "size": 0}
    add_dic = {"difficult": 0, "region_attributes": {"belongId": -1, "graphId": 0, "targetGroupId": 1, "type": "pig"}, "shape_attributes": {"all_points_x": [], "all_points_x": [], "name": "polygon"}, "truncated": 0}
    for i, poly in enumerate(img):
        add_dic["region_attributes"]["targetGroupId"] = i + 1
        ann_type = ""
        if poly.attrib["label"] == 'anus':
            ann_type = 'pig_anus'
        elif poly.attrib["label"] == 'pussy':
            ann_type = 'sow_genitalia'
        elif poly.attrib["label"] == 'ear':
            ann_type = 'pig_ear'
        add_dic["region_attributes"]["type"] = ann_type
        x = []
        y = []
        for xy in poly.attrib["points"].split(';'):
            x.append(eval(xy.split(',')[0]))
            y.append(eval(xy.split(',')[1]))
        x = (np.asarray(x) + 0.5).astype(int)
        y = (np.asarray(y) + 0.5).astype(int)
        x = x.tolist()
        y = y.tolist()
        add_dic["shape_attributes"]["all_points_x"] = x
        add_dic["shape_attributes"]["all_points_y"] = y
        print(add_dic)
        dic[img_name]["regions"].append(copy.deepcopy(add_dic))
print(dic)
mmcv.dump(dic, r"C:\Users\38698\work_space\data\sow.json", indent=4)
