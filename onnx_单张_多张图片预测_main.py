#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import onnxruntime as rt
import cv2
from pylab import *     #引入numpy
import argparse
import os
np.set_printoptions(threshold=np.inf)

def predict(onnx_model,image):
    sess = rt.InferenceSession(onnx_model)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: image})[0]
    return pred_onx

def process_image(image_path,H,W):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img,code=cv2.COLOR_BGR2GRAY)

    height, weight= img.shape
    if opt.show_image:
        cv2.imshow('ori_img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(weight, height)
    if weight >= (W / H) * height:
        w = W
        h = int(height / (weight / w))
        img = cv2.resize(img, (w, h))
    else:
        h = H
        w = int(weight / (height / h))
        img = cv2.resize(img, (w, h))
    height, weight = img.shape

    # 左上角点的坐标
    x = int((W - weight) / 2)
    y = int((H - height) / 2)
    #把图片的像素平均值当背景色。
    bg_num = int(np.mean(np.asarray(img)))
    #把图片外边框的像素平局值当背景色。
    # sum = 0
    # num = 0
    # for i in range(w):
    #     for j in range(h):
    #         if i == 0 or j == 0 or i == w-1 or j == h-1:
    #             sum += img[j][i]
    #             num += 1
    # bg_num = sum // num

    join_bg = np.asarray(np.ones((H, W)) * bg_num, dtype=np.uint8)

    join_bg[y:y + height, x:x + weight] = img
    image_data = np.array(join_bg, dtype='float32')
    if opt.show_image:
        cv2.imshow('org_bg_img',join_bg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    image_data /= 255.
    image_data = np.reshape(image_data,(1,1,H,W))
    return image_data

def decode_onnx_pred(pred):
    # 预测结果转换为最大值形式
    Max = np.argmax(pred, axis=2)
    Max = Max.reshape((71))
    MaxList = Max.tolist()

    # 预测结果输出
    alphabetChinese = r' 1234567890qwerTYUIOPASDFGHtyuiopasdfghjklzxcvbnmQWERJKLZXCVBNM/-!.'
    str1 = ''
    for i in range(0, len(MaxList)):
        if MaxList[i] != 0 and (i == 0 or (i != 0 and MaxList[i] != MaxList[i - 1])):
            m = MaxList[i]
            # print(alphabetChinese[m])
            str1 += alphabetChinese[m]
    return str1

def show_image(image_path,show_name):
    image = cv2.imread(image_path)
    cv2.imshow(show_name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image_path = opt.image_path
    if os.path.isfile(image_path):
        image = process_image(image_path,H=opt.image_height,W=opt.image_width)
        onnx_model = opt.onnx_model
        pred = predict(onnx_model,image)
        str1 = decode_onnx_pred(pred)
        print(str1)
        if opt.show_image:
            show_image(image_path,str1)
    elif os.path.isdir(image_path):
        images = os.listdir(image_path)
        for img in images:
            image_full_path = os.path.join(image_path,img)
            image = process_image(image_full_path,H=opt.image_height,W=opt.image_width)
            onnx_model = opt.onnx_model
            pred = predict(onnx_model, image)
            str1 = decode_onnx_pred(pred)
            print(str1)
            if opt.show_image:
                show_image(image_full_path,str1)


if __name__ == '__main__':
    # image_path = r'F:\pc\work\test_python\test\ultralytics\yolov3\output\cut'
    image_path = r'F:\pc\work\test_python\test\ultralytics\yolov3\output\cut'

    parser = argparse.ArgumentParser()
    parser.add_argument('--show_image',default=True,help='choose to show image')
    parser.add_argument('--image_path',default=image_path,help='the image_path or directory for predict')
    parser.add_argument('--image_height',default=32,help='the image_height for predict')
    parser.add_argument('--image_width',default=280,help='the image_width for predict')
    parser.add_argument('--onnx_model',default="./torch_model_train_Acc100.000_test_Acc99.567_2_3.onnx",help='the onnx_model to predict')
    opt = parser.parse_args()
    main()
