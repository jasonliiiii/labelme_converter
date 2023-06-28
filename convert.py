'''
Based on: https://github.com/WoodsGao/cv_utils.git

'''


import copy
import json
import os
import os.path as osp
from tqdm import tqdm
from pycocotools import mask
import cv2

# Helper Function
def find_all_img_anns(coco):
    img_id_list = []
    anns_list = []
    for img_info in coco['images']:
        img_id_list.append(img_info['id'])
        anns_list.append([])
    for ann in tqdm(coco['annotations']):
        index = img_id_list.index(ann['image_id'])
        anns_list[index].append(ann)
    return coco['images'], anns_list

DEFAULT_LABELME = {
    "flags": {},
    "fillColor": [255, 0, 0, 128],
    "lineColor": [0, 255, 0, 128],
    "version": "3.16.7",
    'imageData': None,
}

DEFAULT_SHAPE = {"flags": {}, "line_color": None, "fill_color": None}

input_json = './dataset/stuff_train2017.json' # path to the COCO annotations
output = './dataset/label/' # path of the outputs

os.makedirs(output, exist_ok=True)
with open(input_json, 'r') as f:
    coco = json.loads(f.read())
img_info_list, anns_list = find_all_img_anns(coco)

def coco2labelme(img_info_list, anns_list):
    for img_info, anns in zip(img_info_list, anns_list):
        labelme_json = copy.deepcopy(DEFAULT_LABELME)
        labelme_json['imageHeight'] = img_info['height']
        labelme_json['imageWidth'] = img_info['width']
        shapes = []
        for ann in anns:
            shape = copy.deepcopy(DEFAULT_SHAPE)
            segmentation = ann['segmentation']
            # segmentation is encoded in RLE, requires decoding
            binary_mask = mask.decode(segmentation)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            points = []
            for contour in contours:
                contour = contour.squeeze().tolist()
                for point in contour:
                    points.append(point)
            label = None
            for category in coco['categories']:
                if category['id'] == ann['category_id']:
                    label = category['name']
                    break
            if len(points) == 1:
                shape_type = 'point'
            elif len(points) == 2:
                shape_type = 'line'
            elif len(points) <= 0:
                continue
            else:
                shape_type = 'polygon'
            shape['points'] = points
            shape['shape_type'] = shape_type
            shape['label'] = label
            shapes.append(shape)

        labelme_json['shapes'] = shapes
        labelme_json['imagePath'] = osp.relpath(
            osp.join(osp.dirname(input_json), img_info['file_name']), output)
        with open(osp.join(output, osp.splitext(osp.basename(img_info['file_name']))[0] + '.json'), 'w') as f:
            f.write(json.dumps(labelme_json, indent=4, sort_keys=True))

if __name__ == '__main__':
    coco2labelme(img_info_list, anns_list)

