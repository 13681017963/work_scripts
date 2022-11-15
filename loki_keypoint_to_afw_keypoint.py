import cv2
import mmcv
import os
import json
import numpy as np
from cut_outside_black import LokiLoadFetcher, ResourceContent

global_key = {}
global_enlarge_bbox = {}


def convert_loki_to_afw(loki_saved_dir, afw_out_dir, output_json_path,  bbox_enlarge_ratio=1.5):
    loki_ann_file = loki_saved_dir + 'via_region_data.json'
    data_infos = mmcv.load(loki_ann_file)
    bbox = {}
    for idx, v in enumerate(mmcv.track_iter_progress(data_infos.values())):
        obj_count = 1
        filename = v['filename']
        graphId_obj = {}
        for region in v["regions"]:
            ratype = region['region_attributes']['type']
            # 只有一个keypoint的猪不支持训练要被滤掉
            if ratype == 'pig' and len(region['shape_attributes']['all_points_x']) > 1 and len(region['shape_attributes']['all_points_y']) > 1:
                img = cv2.imread(afw_out_dir + filename.split('.')[-2] + '.jpg')
                cv2.imwrite(afw_out_dir + filename.split('.')[-2] + '_' + str(obj_count) + '.jpg', img)
                f = open(afw_out_dir + filename.split('.')[-2] + '_' + str(obj_count) + '.pts', "w")
                f.write("version: 1\n")
                f.write("n_points: 31\n")
                f.write("{\n")
                for i in range(31):
                    f.write("-1 -1\n")
                f.write("}")
                f.close()
                x = region['shape_attributes']['all_points_x']
                y = region['shape_attributes']['all_points_y']
                minX = min(x)
                minY = min(y)
                maxX = max(x)
                maxY = max(y)
                w = float(maxX - minX)
                h = float(maxY - minY)
                left_top_x = float(minX)
                left_top_y = float(minY)
                val = {"face_outer_bboxx": left_top_x, "face_outer_bboxy": left_top_y, "face_outer_bboxwidth": w,
                       "face_outer_bboxheight": h, "face_tight_bboxx": left_top_x, "face_tight_bboxy": left_top_y,
                       "face_tight_bboxwidth": w, "face_tight_bboxheight": h}
                bbox[filename.split('.')[-2] + '_' + str(obj_count)] = val
                graphId_obj[region['region_attributes']['graphId']] = obj_count
                r = bbox_enlarge_ratio - 1
                global_enlarge_bbox[filename.split('.')[-2] + '_' + str(obj_count)] = \
                    ((minX - ((w * r) / 2), minY - ((h * r) / 2)),
                     (maxX + ((w * r) / 2), minY - ((h * r) / 2)),
                     (maxX + ((w * r) / 2), maxY + ((h * r) / 2)),
                     (minX - ((w * r) / 2), maxY + ((h * r) / 2)))
                obj_count += 1
        for region in v["regions"]:
            ratype = region['region_attributes']['type']
            if 'pig_point' in ratype:
                # loki bug, some operations on loki can cause the keypoint to have a belongId of -1, such data will cause training to fail in TAO toolkit
                # If the bug is fixed, this part can be removed
                if region['region_attributes']['belongId'] == -1:
                    for name in os.listdir(afw_out_dir):
                        if filename.split('.')[-2] in name:
                            os.remove(afw_out_dir + name)
                    continue

                with open(afw_out_dir + filename.split('.')[-2] + '_' + str(
                        graphId_obj[region['region_attributes']['belongId']]) + '.pts', 'r') as f:
                    lines = f.readlines()
                x = float(region['shape_attributes']['all_points_x'][0])
                y = float(region['shape_attributes']['all_points_y'][0])
                global_key[filename.split('.')[-2] + '_' + str(graphId_obj[region['region_attributes']['belongId']])] = (x, y)
                lines[eval(ratype.split('_')[-1]) + 2] = '%.6lf %.6lf' % (x, y) + '\n'
                with open(afw_out_dir + filename.split('.')[-2] + '_' + str(
                        graphId_obj[region['region_attributes']['belongId']]) + '.pts', 'w') as f:
                    for data in lines:
                        f.write(data)
                    f.flush()
        try:
            os.remove(afw_out_dir + filename.split('.')[-2] + '.jpg')
        except FileNotFoundError:
            pass
    # save json
    with open(output_json_path, "w") as config_file:
        json.dump(bbox, config_file, indent=4)


def any2jpg(dir):
    for name in os.listdir(dir):
        if name.split('.')[1] != 'jpg':
            img = cv2.imread(dir + '/' + name)
            cv2.imwrite(dir + '/' + name.split('.')[0] + ".jpg", img)
            os.remove(dir + '/' + name)


def get_keypoints_from_file(keypoints_file):
    """
    This function reads the keypoints file from afw format.

    Input:
        keypoints_file (str): Path to the keypoints file.
    Output:
        keypoints (np.array): Keypoints in numpy format [[x, y], [x, y]].
    """
    keypoints = []
    with open(keypoints_file) as fid:
        for line in fid:
            if "version" in line or "points" in line or "{" in line or "}" in line:
                continue
            else:
                loc_x, loc_y = line.strip().split(sep=" ")
                keypoints.append([float(loc_x), float(loc_y)])
    keypoints = np.array(keypoints, dtype=np.float)
    assert keypoints.shape[1] == 2, "Keypoints should be 2D."
    return keypoints


def convert_dataset(afw_data_path, output_json_path, afw_image_save_path, bbox_json_path, key_points=80, debug_bbox=False):
    """
    Function to convert afw dataset to Sloth format json.

    Input:
        afw_data_path (str): Local path to images and labels.
        output_json_path (str): Local path to output sloth format json file.
        afw_image_save_path (str): afw_data_path in docker, os.path.join(os.environ["USER_EXPERIMENT_DIR"], 'afw') in jupyter notebook.
        key_points (int): keypoint number, pig have 27 keypoints. 4 points for enlarge bbox.
    Returns:
        None
    """
    # get dataset file lists
    all_files = os.listdir(afw_data_path)
    images = [x for x in all_files if x.endswith('.jpg')]
    keypoint_files = [img_path.split(".")[-2] + ".pts" for img_path in images]

    output_folder = os.path.dirname(output_json_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # read and convert to sloth format
    sloth_data = []
    with open(bbox_json_path, 'r') as f:
        bbox = json.load(f)
    for image in images:
        image_path = os.path.join(afw_data_path, image)
        image_read = cv2.imread(image_path)
        if image_read is None:
            print('Bad image:{}'.format(image_path))
            continue
        # convert image to png
        image_png = image.replace('.jpg', '.png')
        cv2.imwrite(os.path.join(afw_data_path, image_png), image_read)
        image_data = {}
        image_data['filename'] = os.path.join(afw_image_save_path, image_png).replace('\\', '/')
        image_data['class'] = 'image'

        if debug_bbox:
            annotations_bbox = {}
            annotations_bbox['class'] = 'FaceBbox'
            annotations_bbox['tool-version'] = '1.0'
            annotations_bbox['Occlusion'] = 0
            bbox_key = image_png.split('.')[-2]
            annotations_bbox['face_outer_bboxx'] = bbox[bbox_key]['face_outer_bboxx']
            annotations_bbox['face_outer_bboxy'] = bbox[bbox_key]['face_outer_bboxy']
            annotations_bbox['face_outer_bboxwidth'] = bbox[bbox_key]['face_outer_bboxwidth']
            annotations_bbox['face_outer_bboxheight'] = bbox[bbox_key]['face_outer_bboxheight']
            annotations_bbox['face_tight_bboxx'] = bbox[bbox_key]['face_tight_bboxx']
            annotations_bbox['face_tight_bboxy'] = bbox[bbox_key]['face_tight_bboxy']
            annotations_bbox['face_tight_bboxwidth'] = bbox[bbox_key]['face_tight_bboxwidth']
            annotations_bbox['face_tight_bboxheight'] = bbox[bbox_key]['face_tight_bboxheight']

        annotations = {}
        annotations['tool-version'] = '1.0'
        annotations['version'] = 'v1'
        annotations['class'] = 'FiducialPoints'

        keypoint_file = image.split(".")[-2] + ".pts"
        image_keypoints = get_keypoints_from_file(os.path.join(afw_data_path, keypoint_file))

        if key_points == 80:
            for num, keypoint in enumerate(image_keypoints):
                annotations["P{}x".format(num + 1)] = keypoint[0]
                annotations["P{}y".format(num + 1)] = keypoint[1]

            # fill in dummy keypoints for keypoints 69 to 80
            for num in range(69, 81, 1):
                annotations["P{}x".format(num)] = image_keypoints[0][0]
                annotations["P{}y".format(num)] = image_keypoints[0][1]
                annotations["P{}occluded".format(num)] = True
        elif key_points == 10:
            key_id = 1
            for num, keypoint in enumerate(image_keypoints):
                # change to 10-points dataset:
                if (num + 1) in [1, 9, 17, 20, 25, 39, 45, 34, 49, 55]:
                    annotations["P{}x".format(key_id)] = keypoint[0]
                    annotations["P{}y".format(key_id)] = keypoint[1]
                    key_id += 1
        elif key_points == 31:
            for num, keypoint in enumerate(image_keypoints):
                if num == 27 or num == 28 or num == 29 or num == 30:
                    annotations["P{}x".format(num + 1)] = global_enlarge_bbox[image_png.split('.')[-2]][num - 27][0]
                    annotations["P{}y".format(num + 1)] = global_enlarge_bbox[image_png.split('.')[-2]][num - 27][1]
                    annotations["P{}occluded".format(num + 1)] = True
                    continue
                if keypoint[0] == -1 and keypoint[1] == -1:
                    annotations["P{}x".format(num + 1)] = global_key[image_png.split('.')[-2]][0]
                    annotations["P{}y".format(num + 1)] = global_key[image_png.split('.')[-2]][1]
                    annotations["P{}occluded".format(num + 1)] = True
                else:
                    annotations["P{}x".format(num + 1)] = keypoint[0]
                    annotations["P{}y".format(num + 1)] = keypoint[1]
        else:
            raise ValueError("This script only generates 10 & 80 & 31 keypoints dataset.")

        # image_data['annotations'] = [annotations_bbox, annotations]
        image_data['annotations'] = [annotations]
        sloth_data.append(image_data)

    # save json
    with open(output_json_path, "w") as config_file:
        json.dump(sloth_data, config_file, indent=4)
    os.remove(bbox_json_path)


if __name__ == '__main__':
    # loki dataset: cut outside black
    print("\nloki dataset: cut outside black, start.")
    img_write_to_dir = 'C:/Users/38698/work_space/data/pig_keypoint/fpenet'
    file_load_fetcher = LokiLoadFetcher()
    image = ResourceContent(file_load_fetcher)
    file_load_from = 'C:/Users/38698/work_space/data/pig_keypoint/20221026144428/train/via_region_data.json'
    img_read_from_dir = 'C:/Users/38698/work_space/data/pig_keypoint/20221026144428/train'
    remain_folder = True
    mask_name = 'valid_area'
    image.cut_image(file_load_from, img_read_from_dir, img_write_to_dir, remain_folder=remain_folder,
                    mask_name=mask_name, processes_num=4)
    print("\nloki dataset: cut outside black, end.")

    # loki format to afw format
    print("\nloki format to afw format start.")
    #  calculate bbox when generate tfrecords, the file will remove in the end
    bbox_json_path = 'C:/Users/38698/work_space/data/pig_keypoint/bbox.json'
    any2jpg(img_write_to_dir)
    data_name = 'C:/Users/38698/work_space/data/pig_keypoint/20221026144428'
    loki_saved_dir_train = "%s/train/" % data_name
    afw_out_dir_train = "%s/" % img_write_to_dir
    convert_loki_to_afw(loki_saved_dir=loki_saved_dir_train, afw_out_dir=afw_out_dir_train,
                        output_json_path=bbox_json_path, bbox_enlarge_ratio=1.1)
    print("\nloki format to afw format end.")

    # afw format to sloth format json file
    print("\nafw format to sloth format json file start.")
    convert_dataset(afw_data_path=img_write_to_dir,
                    output_json_path='C:/Users/38698/work_space/data/pig_keypoint/pig.json',
                    afw_image_save_path='/workspace/tao-experiments/fpenet/pig', bbox_json_path=bbox_json_path,
                    key_points=31, debug_bbox=False)
    print("\nafw format to sloth format json file end.")
