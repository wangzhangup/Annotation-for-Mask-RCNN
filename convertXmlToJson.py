# -*- coding: UTF-8 -*-
## create by WANG Zhang
## create date: 2018-04-11

## convert xml to json
## using LabelImgTools to get xml files
## the final json's format is cocodataset format

## xml
import xml.etree.cElementTree as ET

from skimage.measure import label, regionprops
import numpy as np
import cv2
# from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt

import json

import sys

## process bar
def process_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

## init coco
# coco = COCO('/Users/zhang/Downloads/annotations/captions_val2014.json')

## load image list from txt file

imgs_list = open('image.txt','r').readlines()

## init elements: images = [], categories = [], annotations =[]
images, categories, annotations = [], [], []


## ------------------
## ----categories----
## ------------------
category_dict = {"one": 1, "two":2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "night": 9, "ten": 10}

for cat_n in category_dict:
    categories.append({"supercategory":"","id":category_dict[cat_n],"name":cat_n})

# print categories

## --------------
## ----images----
## --------------
count = 1
total = len(imgs_list)

anno_id_count = 0

for img in imgs_list:
    process_bar(count, total)
    count += 1
    if 'jpg' in img:
        img_cv2 = cv2.imread(img.strip())
        [height, width, _] = img_cv2.shape
        img_name = img.strip().split('/')[-1]
        img_id = int(img_name.split('.')[0])

        images.append({"file_name":img_name, "height":height, "width":width, "id":img_id})

    ## ------------------
    ## ----annotation----
    ## ------------------
    ## parsing xml to dict

        xml_n = img.strip().split('/')[0] + '/' + 'image_ch09_720p' + str(img_id) + '.xml'

        tree = ET.ElementTree(file=xml_n)

        root = tree.getroot()
        for child_of_root in root:
            if child_of_root.tag == 'filename':
                image_id = int(child_of_root.text)
            if child_of_root.tag == 'object':
                for child_of_object in child_of_root:
                    if child_of_object.tag == 'name':
                        category_id = category_dict[child_of_object.text]
                        segmentation = []
                        segment = []
                    if child_of_object.tag == 'polygon':
                        for points in child_of_object:
                            point = points.text.split(',')
                            segmentation.append([int(point[0]), int(point[1])])
                            segment.extend([int(point[0]), int(point[1])])

                        # print segmentation
                        img = np.zeros((720,1280))
                        cv2.fillPoly(img, [np.array(segmentation)], (1))

                        area = np.sum(img)


                        label_img = label(img)
                        regions = regionprops(label_img)

                        prop = regions[0]
                        # print prop.centroid, prop.orientation, prop.bbox
                        bboxes = prop.bbox
                        # cv2.rectangle(img,(bboxes[1],bboxes[0]),(bboxes[3],bboxes[2]), (1), 2)
                        # cv2.imshow('ploy', img)
                        # cv2.waitKey(0)
                        bbox = [bboxes[1], bboxes[0], bboxes[3]-bboxes[1], bboxes[2]-bboxes[0]]
                        anno_info = {'id':anno_id_count,'category_id':category_id,'bbox': bbox,'segmentation':[segment],'erea':area,'iscrowd':0, 'image_id': image_id}
                        annotations.append(anno_info)
                        anno_id_count += 1


## summary
all_json = {"images":images,"annotations":annotations,"categories":categories}

## write JSON data to a file

with open("dataset.json","w") as outfile:
    json.dump(all_json, outfile)


# I = io.imread('./42520.jpg')
# for i in range(len(anns)):
#     bbox = anns[i]["bbox"]
#     cv2.rectangle(I, (int(bbox[0]), int(bbox[1])),(int(bbox[0]+bbox[2]), int(bbox[1]+ bbox[3])), (255,255,255), 1)
# plt.imshow(I);plt.axis('off')
# coco.showAnns(anns)
# plt.show()
