import glob
import json
import sys
import cv2
import tqdm
from IPython import embed

jpgs = glob.glob("%s/*.jpg" % sys.argv[1])
jpegs = glob.glob("%s/*.jpeg" % sys.argv[1])
pngs = glob.glob("%s/*.png" % sys.argv[1])
files = jpgs + jpegs + pngs
names = []
id = int(sys.argv[2])
for file in tqdm.tqdm(files):
    jpg_name = file.split('/')[-1]
    img = cv2.imread(file)
    height = img.shape[0]
    width = img.shape[1]
    tmp = {"id": id, "file_name": jpg_name, "height": height, "width": width}
    names.append(tmp)
    id += 1
print(json.dumps(names, ensure_ascii=False))