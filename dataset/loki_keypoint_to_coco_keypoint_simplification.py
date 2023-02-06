import re
import os.path as osp
import mmcv
import sys
import os
from shapely.geometry import Polygon
import numpy as np
from cut_outside_black import LokiLoadFetcher, ResourceContent

pig_keypoint = ['pig_point_1', 'pig_point_2', 'pig_point_3', 'pig_point_4', 'pig_point_5', 'pig_point_6',
                'pig_point_7', 'pig_point_8', 'pig_point_9', 'pig_point_10', 'pig_point_11', 'pig_point_12',
                'pig_point_13', 'pig_point_14', 'pig_point_15', 'pig_point_16', 'pig_point_17', 'pig_point_18',
                'pig_point_19', 'pig_point_20', 'pig_point_21', 'pig_point_22', 'pig_point_23', 'pig_point_24',
                'pig_point_25', 'pig_point_26', 'pig_point_27']

# pig_keypoint_choose = ['pig_point_1', 'pig_point_22', 'pig_point_2', 'pig_point_20', 'pig_point_3', 'pig_point_19',
#                        'pig_point_4',
#                        'pig_point_16', 'pig_point_23', 'pig_point_11', 'pig_point_6', 'pig_point_18', 'pig_point_7',
#                        'pig_point_15', 'pig_point_8', 'pig_point_12', 'pig_point_10', 'pig_point_14']

pig_keypoint_choose = ['pig_point_1', 'pig_point_2', 'pig_point_20', 'pig_point_3', 'pig_point_19', 'pig_point_4',
                       'pig_point_16', 'pig_point_5', 'pig_point_17', 'pig_point_6', 'pig_point_18', 'pig_point_8',
                       'pig_point_12', 'pig_point_9', 'pig_point_13', 'pig_point_10', 'pig_point_14']


def filter(filename, content, verbose=False):
    lens = re.findall(r'(.*?)%s(.*?)' % content, filename)
    status = True if len(lens) > 0 else False
    if not status and verbose:
        print('\n Skip %s' % filename)
    return status


def manual_modification(raw, on):
    """特殊情况下，改变某些class的名称 or 改变某些category的id"""
    if on == 'type':
        if raw == 'indoor_aisle':
            raw = '%s'
        elif raw == '%s':
            pass
        else:
            print('Unkown type %s' % raw)
            sys.exit()
        return raw
    elif on == 'category':
        if raw == 1:
            raw = 0
        return raw
    else:
        print('Unkown modification %s' % on)
        sys.exit()


def parse_category_id(LOKI_ann_file_train, LOKI_ann_file_val, given_id_dict={}, ignore_keys=[]):
    """遍历训练测试数据集，给出对应的dict。如果已经有given，则given类的id固定，按顺序排后面的"""
    print("Parsing category-id dict...")
    data_infos_val = mmcv.load(LOKI_ann_file_train).values()
    data_infos_train = mmcv.load(LOKI_ann_file_val).values()
    categoryid_space = 0 if given_id_dict == {} else int(max(given_id_dict.values()) + 1)

    def run(data_infos, categoryid_space):
        for idx, v in enumerate(mmcv.track_iter_progress(data_infos)):
            for region in v["regions"]:
                ratype = region['region_attributes']['type']
                if ratype in given_id_dict or ratype in ignore_keys or ratype in pig_keypoint:
                    pass
                else:
                    given_id_dict[ratype] = categoryid_space
                    categoryid_space += 1
        return given_id_dict, categoryid_space

    given_id_dict, categoryid_space = run(data_infos_train, categoryid_space)
    given_id_dict, categoryid_space = run(data_infos_val, categoryid_space)
    print('given_id_dict: %s' % given_id_dict)
    return given_id_dict


def tran_id(num, num_lst=[1, 2, 20, 3, 19, 4, 16, 5, 17, 6, 18, 8, 12, 9, 13, 10, 14]):
    try:
        return num_lst.index(num)
    except ValueError:
        return -1


def convert_balloon_to_coco(LOKI_ann_file, COCO_out_json, image_prefix, category_id_dict):
    """
    using mmcv , convert ballon json to  coco json file .
    LOKI_ann_file : input , ballon format json file
    COCO_out_json : output , file  in coco format to write
    image_prefix: input ,  folder contains all images
    category_id_dict: input , None for no use. or like this : { "pig":0 , 'pigsty_corral': 1, 'pigsty_floor': 2 }
    """
    data_infos = mmcv.load(LOKI_ann_file)

    annotations = []
    images = []
    obj_count = 0
    for idx, v in enumerate(mmcv.track_iter_progress(data_infos.values())):
        filename = v['filename']
        status = filter(filename, content='.', verbose=True)  # Note optional, '.' means leave all filenames
        if not status:
            continue
        img_path = osp.join(image_prefix, filename)
        if os.path.exists(img_path):
            height, width = mmcv.imread(img_path).shape[:2]
        else:
            print("\n***Warning, %s not exists!" % img_path)
            continue
        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))
        bboxes = []
        labels = []
        masks = []
        all_data = {}
        for region in v["regions"]:
            ratype = region['region_attributes']['type']
            if ratype in ignore_keys or ratype in pig_keypoint or region['shape_attributes']['name'] != 'polygon':
                continue
            group_id = region['region_attributes']['targetGroupId']
            if group_id not in all_data:
                all_data[group_id] = {}
            all_data[group_id]['category_id'] = -1
            all_data[group_id]['num_keypoints'] = 0
            all_data[group_id]['keypoints'] = [0, 0, 0] * 17
            all_data[group_id]['graphId'] = region['region_attributes']['graphId']
            # ratype = manual_modification(ratype, on='type')           # Note  optional
            all_data[group_id]['category_id'] = category_id_dict[ratype]
            obj = region['shape_attributes']
            if 'poly' not in all_data[group_id]:
                all_data[group_id]['poly'] = [].copy()
            px = obj['all_points_x']
            py = obj['all_points_y']
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            x_min, y_min, x_max, y_max = (
                min(px), min(py), max(px), max(py))

            all_data[group_id]['x_min'] = 10000000
            all_data[group_id]['y_min'] = 10000000
            all_data[group_id]['x_max'] = -10000000
            all_data[group_id]['y_max'] = -10000000

            all_data[group_id]['poly'].append(poly)
            if x_min < all_data[group_id]['x_min']:
                all_data[group_id]['x_min'] = x_min

            if y_min < all_data[group_id]['y_min']:
                all_data[group_id]['y_min'] = y_min

            if x_max > all_data[group_id]['x_max']:
                all_data[group_id]['x_max'] = x_max

            if y_max > all_data[group_id]['y_max']:
                all_data[group_id]['y_max'] = y_max
        for region in v["regions"]:
            ratype = region['region_attributes']['type']
            if ratype in pig_keypoint:
                for key, current_pig in all_data.items():
                    if current_pig['graphId'] == region['region_attributes']['belongId']:
                        all_data[key]['num_keypoints'] += 1
                        k = tran_id(eval(ratype.split('_')[-1]))
                        if k != -1:
                            all_data[key]['keypoints'][k * 3] = region['shape_attributes']['all_points_x'][0]
                            all_data[key]['keypoints'][k * 3 + 1] = region['shape_attributes']['all_points_y'][0]
                            all_data[key]['keypoints'][k * 3 + 2] = 2
                            break
        for k, v in all_data.items():
            x_min, y_min, x_max, y_max = v['x_min'], v['y_min'], v['x_max'], v['y_max']
            poly = v['poly']
            category_id = v['category_id']
            # category_id = manual_modification(category_id, on='category')   #Note optional
            if category_id == -1:
                continue
            # area = (x_max - x_min) * (y_max - y_min)     # defaults to area of Rectangle
            area = 0
            for one_poly in poly:  # fixed by lmw: some poly may contains more than one closed region.
                tmp_poly = Polygon(np.reshape(one_poly, (-1, 2)))
                area += tmp_poly.area
            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=category_id,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=area,  # fixed by lmw: area could refer to either Rectangle area or Segment area.
                segmentation=poly,
                iscrowd=0,
                num_keypoints=v['num_keypoints'],
                keypoints=v['keypoints'])
            annotations.append(data_anno)
            obj_count += 1
    categories = []
    for item in category_id_dict.items():
        categories.append({'supercategory': 'person', 'id': 1, 'name': 'person', 'keypoints': pig_keypoint_choose,
                           'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
                                        [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
                                        [4, 6], [5, 7]]})

    # [1, 2], [2, 4], [1, 5], [5, 7], [1, 8], [8, 9], [9, 10], [1, 11],
    # [11, 12], [12, 13], [12, 6], [6, 9], [6, 3], [1, 0], [0, 14], [14, 16],
    # [1, 6], [0, 15], [15, 17]

    # [15, 17], [8, 9], [1, 2], [1, 3], [2, 4], [3, 5], [6, 7], [6, 10],
    # [7, 11], [6, 12], [7, 13], [12, 13], [12, 14], [13, 15], [14, 16], [14, 8],
    # [8, 15], [4, 6], [5, 7]]
    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=categories)

    mmcv.dump(coco_format_json, COCO_out_json)

    print('convert over %s, with type : id map :' % LOKI_ann_file, category_id_dict)


if __name__ == '__main__':
    file_load_fetcher = LokiLoadFetcher()
    image = ResourceContent(file_load_fetcher)
    file_load_from = r'C:\Users\38698\work_space\data\20221109100704\train\via_region_data.json'
    img_read_from_dir = r'C:\Users\38698\work_space\data\20221109100704\train'
    img_write_to_dir = r'C:\Users\38698\work_space\data\pk\train'
    remain_folder = True
    mask_name = 'valid_area'
    image.cut_image(file_load_from, img_read_from_dir, img_write_to_dir, remain_folder=remain_folder,
                    mask_name=mask_name)
    file_load_from = r'C:\Users\38698\work_space\data\20221109100704\val\via_region_data.json'
    img_read_from_dir = r'C:\Users\38698\work_space\data\20221109100704\val'
    img_write_to_dir = r'C:\Users\38698\work_space\data\pk\val'
    image.cut_image(file_load_from, img_read_from_dir, img_write_to_dir, remain_folder=remain_folder,
                    mask_name=mask_name)

    data_name = 'C:/Users/38698/work_space/data/20221109100704'

    LOKI_ann_file_train = "%s/train/via_region_data.json" % data_name
    COCO_out_json_train = "%s/train/annotation_coco.json" % data_name
    image_prefix_train = "%s/train/" % data_name

    LOKI_ann_file_val = "%s/val/via_region_data.json" % data_name
    COCO_out_json_val = "%s/val/annotation_coco.json" % data_name
    image_prefix_val = "%s/val/" % data_name

    # given_id_dict = {'pig_head':0, 'pig_hip':1}
    # ignore_keys = ['center_ROI']
    given_id_dict = {'pig': 1}
    ignore_keys = ['valid_area']

    id_dict = parse_category_id(LOKI_ann_file_train, LOKI_ann_file_val, given_id_dict, ignore_keys)

    # 坐标转换函数，via2coco
    # via文件，coco输出文件，包含所有图片的文件夹路径，类别字典
    convert_balloon_to_coco(LOKI_ann_file=LOKI_ann_file_train, COCO_out_json=COCO_out_json_train,
                            image_prefix=image_prefix_train, category_id_dict=id_dict)
    convert_balloon_to_coco(LOKI_ann_file=LOKI_ann_file_val, COCO_out_json=COCO_out_json_val,
                            image_prefix=image_prefix_val, category_id_dict=id_dict)
