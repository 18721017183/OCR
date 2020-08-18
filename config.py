#!/usr/bin/env python
# -*- coding:utf-8 -*-

class DefaultConfig(object):
    # train_data_root = '/home/ubuntu/ocr/Char/correct_train.txt'
    # valid_data_root = '/home/ubuntu/ocr/Char/correct_test.txt'
    # model_path = r'F:\hagong\ocr\Train\model\weight_ocr-{epoch%3A14%3Aacc98.159}.pth'
    # save_path = '/home/ubuntu/ocr/Train/model/'
    # image_path = '/home/ubuntu/ocr/new'

    '''合成图片'''

    # train_data_root = './data/lv_txt/char_train.txt'     #训练数据txt文件
    # valid_data_root = './data/lv_txt/char_test.txt'     #测试数据txt文件
    # # model_path = r'./model/weight_ocr-{epoch%3A14%3Aacc98.159}.pth'
    # model_path = r'./model/aa.pth'
    # train_image_path = './data/lv_make_image/train'     #训练图片文件夹
    # valid_image_path = './data/lv_make_image/test'      #测试图片文件夹

    train_data_root = './data/lv_generate_txt/char_train.txt'     #训练数据txt文件
    valid_data_root = './data/lv_generate_txt/char_test.txt'     #测试数据txt文件
    # model_path = r'./model/weight_ocr-{epoch%3A14%3Aacc98.159}.pth'
    model_path = r'./model/aa.pth'
    train_image_path = './data/lv_generate_image/train'     #训练图片文件夹
    valid_image_path = './data/lv_generate_image/test'      #测试图片文件夹



    batch_size = 32
    # batch_size = 1
    img_h = 32
    num_workers = 2
    use_gpu = True
    max_epoch = 2
    learning_rate = 0.0001
    weight_decay = 1e-4
    print_interval = 5
    valid_interval = 378

def parse(self, **kwargs):
    for k, v in kwargs.items():
        setattr(self, k, v)
        # setattr(self, k, v)用来扩区属性值，k:属性；v:属性值

DefaultConfig.parse = parse
opt = DefaultConfig
