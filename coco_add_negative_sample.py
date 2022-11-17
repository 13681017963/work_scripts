import glob
import mmcv
import sys
from IPython import embed


# sys[1] 负样本图片文件夹
# sys[2] 原coco文件路径
# sys[3] 添加负样本后的输出文件
jpgs = glob.glob("%s/*.jpg" % sys.argv[1])
jpegs = glob.glob("%s/*.jpeg" % sys.argv[1])
pngs = glob.glob("%s/*.png" % sys.argv[1])
files = jpgs + jpegs + pngs
add_info = mmcv.load(sys.argv[2])
id = -1
for i in add_info['images']:
    id = max(id, i['id'])
id += 1
coco_out_json = sys.argv[3]

for file in mmcv.track_iter_progress(files):
    file = file.replace('\\', '/')
    jpg_name = file.split('/')[-1]
    img = mmcv.imread(file)
    height = img.shape[0]
    width = img.shape[1]
    tmp = {"id": id, "file_name": jpg_name, "height": height, "width": width}
    add_info['images'].append(tmp)
    id += 1
mmcv.dump(add_info, coco_out_json, indent=4)
