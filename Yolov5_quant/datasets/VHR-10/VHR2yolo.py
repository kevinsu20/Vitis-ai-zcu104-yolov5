import os
import pandas as pd
import cv2
ann_path = 'D:\Project\pycharm_project\yolov5-VHR-10\datasets\VHR-10\groundtruth'
output_path = 'D:\Project\pycharm_project\yolov5-VHR-10\datasets\VHR-10\labels'
im_path = 'D:\Project\pycharm_project\yolov5-VHR-10\datasets\VHR-10\images'
ann_list = os.listdir(ann_path)
for index, ann_filename in enumerate(ann_list):
    ann_filepath = os.path.join(ann_path, ann_filename)
    ann_df = pd.read_csv(ann_filepath, header=None)
    annstr = ''
    for i, ann in ann_df.iterrows():
        img_name = ann_filename[0:-3]+'jpg'
        img = cv2.imread(os.path.join(im_path, img_name))
        width = img.shape[1]
        height = img.shape[0]
        x1 = int(ann[0][1:])
        y1 = int(ann[1][0:-1])
        x2 = int(ann[2][1:])
        y2 = int(ann[3][0:-1])
        label = int(ann[4]) - 1
        x_center = (x1+x2)/2/width
        y_center = (y1+y2)/2/height
        w = (x2-x1)/width
        h = (y2-y1)/height
        annstr += f'{label} {x_center} {y_center} {w} {h}\n'
    with open(os.path.join(output_path, ann_filename),'w') as f:
        f.write(annstr)
    print(f'{index} th file done!')