num = 6500

import numpy as np
import os
import cv2 as cv

flag = 'train'

pic_names = []
pic_label = []
with open(r'/data/correct_train.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        pic_label.append(line)
        pic_name = line.strip(' ').split(' ')[0]
        pic_names.append(pic_name)

random_index = np.random.permutation(num)

if flag == 'train':
    pic_names_idx = random_index[:800]
    pic_save_dir = r'C:\Users\pc\Desktop\tmp\make_ocr_image\train'
    label_file = r'C:\Users\pc\Desktop\tmp\make_ocr_image\char_train.txt'
elif flag:
    pic_names_idx = random_index[800:1000]
    pic_save_dir = r'C:\Users\pc\Desktop\tmp\make_ocr_image\test'
    label_file = r'C:\Users\pc\Desktop\tmp\make_ocr_image\char_test.txt'

pic_dir_path = r'/data/new'

for i in pic_names_idx:
    pic_full_name = os.path.join(pic_dir_path,pic_names[i])
    pic_save_name = os.path.join(pic_save_dir,pic_names[i])
    image = cv.imread(pic_full_name)
    cv.imwrite(pic_save_name,image)

with open(label_file,'a') as f:
    for idx in pic_names_idx:
        f.write(pic_label[idx])