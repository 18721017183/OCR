#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import torch
import crnn
from PIL import Image
from config import opt
from torchvision import transforms
device = torch.device('cpu')
import numpy as np

#定义归一化函数
class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

#定义i编码函数
def decode(preds, char_set):
    pred_text = ''
    for i in range(len(preds)):
        if preds[i] != 0 and ((i==0) or (i != 0 and preds[i] != preds[i-1])):
            pred_text += char_set[int(preds[i]-1)]

    return pred_text


if __name__ == '__main__':

    image_path = r'./1.jpg'    #输入图片
    # image_path = r'./data/lv_generate_image/test/1.jpg'

    #从config中读取相关参数
    img_h = opt.img_h
    use_gpu = opt.use_gpu
    model_path = opt.model_path
    #编码值
    char_set = open('char.txt', 'r', encoding='utf-8').readlines()
    char_set = ''.join([ch.strip('\n') for ch in char_set[1:]] + ['$'])
    n_class = len(char_set)
    # n_class = 5990

    #读取模型
    model = crnn.CRNN(img_h, 1, n_class, 256)
    #if torch.cuda.is_available and use_gpu:
        #model.cuda()
    #加载参数
    if os.path.exists(model_path):
        print('Load model from "%s" ...' % model_path)
        model.load_state_dict(torch.load(model_path, map_location = 'cpu'))
        print('Done')

    '''
    Image方式读取图片
    '''
    image = Image.open(image_path).convert('L')
    print(type(image),np.shape(image))
    (w, h) = image.size
    size_h = 32
    ratio = size_h / float(h)
    size_w = int(w * ratio)
    print(type(image))

    #图片预处理，转换为符合模型要求的格式
    transform = resizeNormalize((size_w, size_h))
    # transform = resizeNormalize((256, size_h))
    image = transform(image)
    image = image.unsqueeze(0)
    #if torch.cuda.is_available and use_gpu:
        #image = image.cuda()
    print('输入',image[0][0][1])
    model.eval()

    #模型预测
    preds = model(image)
    print('preds_shape',preds.shape)
    print('11',preds[1])
    preds = preds.max(2)[1]
    print('max',preds)
    preds = preds.squeeze()
    pred_text = decode(preds, char_set)
    print('predict ==> ', pred_text)
