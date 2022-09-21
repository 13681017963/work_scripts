import xml.etree.ElementTree as ET


annotations_file = r'C:\Users\38698\work_space\data\ear\annotations.xml'
yolov5_dataset = r'C:\Users\38698\work_space\data\yolov5_ear'


mytree = ET.parse(annotations_file)
root = mytree.getroot()
saveFolder = yolov5_dataset
for idx, img in enumerate(root.findall('image')):
    print(idx, img)
    WIDTH = eval(img.attrib["width"])
    HEIGHT = eval(img.attrib["height"])
    imgName = img.attrib["name"].split('/')[-1].split('.')[0]
    data = ''
    for poly in img:
        if poly.attrib["label"] == 'anus':
            objClass = 0
        elif poly.attrib["label"] == 'pussy':
            objClass = 1
        elif poly.attrib["label"] == 'ear':
            objClass = 2
        else:
            objClass = 3
        xy = poly.attrib["points"].split(';')
        x = []
        y = []
        for point in xy:
            x.append(eval(point.split(',')[0]))
            y.append(eval(point.split(',')[1]))

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
    file.close()
