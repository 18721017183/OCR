import os
import cv2 as cv
import numpy as np
import random

number_classes_dir = r'F:\pc\work\test_python\test\OCR\Train\data\image\others\number'   #数字图片文件夹
#所有标签
number_dir_list = os.listdir(number_classes_dir)

flag = 'train'    #train // test

if flag == 'train':
    image_save_path = r'F:\pc\work\test_python\test\OCR\Train\data\lv_generate_image\train'    #合成图片保存的文件夹，train // test
    label_txt = r'F:\pc\work\test_python\test\OCR\Train\data\lv_generate_txt\train_label_20200702.txt'      #图片的标签 train.txt // test.txt
    generate_image_num = 500
elif flag == 'test':
    image_save_path = r'F:\pc\work\test_python\test\OCR\Train\data\lv_generate_image\test'  # 合成图片保存的文件夹，train // test
    label_txt = r'F:\pc\work\test_python\test\OCR\Train\data\lv_generate_txt\test_label_20200702.txt'  # 图片的标签 train.txt // test.txt
    generate_image_num = 2500   #生成的图片数量

## 图片旋转
def rotate_bound(image, angle):
    # 获取宽高
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # 提取旋转矩阵 sin cos
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像的新边界尺寸
    nW = int((h * sin) + (w * cos))
    #     nH = int((h * cos) + (w * sin))
    nH = h

    # 调整旋转矩阵
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv.warpAffine(image, M, (nW, nH), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

#将数字图片与标签保存在字典中
dic_all_classes = {}
#遍历所有的标签
for number_dir in number_dir_list:
    #具体标签路径
    number_class_dir = os.path.join(number_classes_dir, number_dir)
    image_list = os.listdir(number_class_dir)
    #遍历标签下的所有图片
    for image in image_list:
        image_full_name = os.path.join(number_class_dir, image)
        #读取图片
        number_image = cv.imread(image_full_name,0)  #灰度读取
        #将图片保存到字典中
        if number_dir not in dic_all_classes:
            dic_all_classes[number_dir] = []
            dic_all_classes[number_dir].append(number_image)
        else:
            dic_all_classes[number_dir].append(number_image)

#字典中所有的key
dic_all_classes_keys = list(dic_all_classes.keys())

"""
根据数字图片与背景，合成图片，并且保存图片与标签
"""
for idx in range(generate_image_num):
    FLAG = False
    # 读取背景，并且resize
    image2 = cv.imread(r'./bg.jpg',0)
    height_bg, width_bg = image2.shape
    h_bg = random.randint(32,48)
    w_bg = int((280 / 32.0) * h_bg)
    image2 = cv.resize(image2,(w_bg,h_bg)).reshape(h_bg,w_bg)
    # h_bg = 32
    # w_bg = 280
    # image2 = cv.resize(image2, (280, 32)).reshape(32, 280)

    #第一张图片开始位置
    start_width = random.randint(5,70)
    start_height = random.randint(0,h_bg-32)
    end_width = 0
    words = ''
    for i in range(12):
        #随机选取一张数字图片，并且resize
        a = random.choice(dic_all_classes_keys)
        a_images = os.listdir(os.path.join(number_classes_dir,a))
        b = random.randint(0,len(a_images)-1)
        hehght,width = dic_all_classes[a][b].shape
        h = 32
        w = int(width / hehght * 32)

        #图片旋转
        angels = np.random.randint(-2, 2)
        image = rotate_bound(dic_all_classes[a][b],angels)
        image = cv.resize(image,(w,h))
        image = image.reshape((h,w))

        #将数字图片与背景融合
        if start_width+w >= w_bg:
            FLAG = True
            continue
        image2[start_height:start_height+32,start_width:start_width+w] = image

        start_width = start_width+w
        #保存标签
        words = words + a

    if FLAG:
        continue
    #增加噪声点
    # rows, cols,_ = image2.shape
    # noise_num = random.randint(0,50)
    # noise_value = random.randint(0,255)
    # for i in range(noise_num):
    #     x = np.random.randint(cols)
    #     y = np.random.randint(rows)
    #     image2[y,x] = noise_value

    # # 随机灰度的背景
    # rate = np.random.randint(50, 255)
    # print('rate',rate)
    # re,image2 = cv.threshold(src=image2,thresh=240,maxval=rate,type=cv.THRESH_BINARY)
    # print('rate',re)

    #将整个图片resize到（32，280）
    image2 = cv.resize(image2,(280,32))

    #图片及标签保存
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    image_save_full_name = os.path.join(image_save_path,str(idx)+'#1_.jpg')
    print(idx)
    cv.imwrite(image_save_full_name, image2)
    with open(label_txt,'a') as f:
        line = str(idx)+'#1_.jpg' + ' ' + words + '\n'
        f.write(line)