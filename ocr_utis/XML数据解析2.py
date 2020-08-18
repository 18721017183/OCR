# import xml.dom.minidom
# import os
#
# import cv2 as cv
#
#
# path = r'C:\Users\pc\Desktop\tmp\数据\2flowers\xml\image_0561.xml'
# DOMTree = xml.dom.minidom.parse(path)
# collection = DOMTree.documentElement
# filename = collection.getElementsByTagName('filename')[0].childNodes[0].data
# print(filename)
# dir_path = collection.getElementsByTagName('path')[0].childNodes[0].data
# print(dir_path)
#
# object = collection.getElementsByTagName('object')[0]
# name = object.getElementsByTagName('name')[0].childNodes[0].data
# bndbox = object.getElementsByTagName('bndbox')[0]
# xmin = bndbox.getElementsByTagName('xmin')[0].childNodes[0].data
# ymin = bndbox.getElementsByTagName('ymin')[0].childNodes[0].data
# xmax = bndbox.getElementsByTagName('xmax')[0].childNodes[0].data
# ymax= bndbox.getElementsByTagName('ymax')[0].childNodes[0].data
# print(xmin,ymin,xmax,ymax)
#
# dir_path=r'C:\Users\pc\Desktop\tmp\image_0561.jpg'
# img = cv.imread(filename=dir_path)
# cv.imshow('img',img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# xmin,xmax,ymin,ymax=int(xmin),int(xmax),int(ymin),int(ymax)
# box = img[xmin:xmax,ymin:ymax]
# cv.imshow('img2',box)
# cv.waitKey(0)
# cv.destroyAllWindows()
# cv.imwrite(filename,box)

import os
import cv2 as cv
import xml.dom.minidom
import numpy as np
from PIL import Image

# path = r'F:\pc\work\test_python\test\OCR\Train\data\image\others\xml'    #xml文件夹路径
path = r'C:\Users\pc\Desktop\img\tmp\xml'    #xml文件夹路径
xmls = os.listdir(path=path)

# tmp_path = r'C:\Users\pc\Desktop\tmp\jpg'
classes_dir = r'C:\Users\pc\Desktop\img\tmp'  #截取的图片保存路径，类别路径

def f1():
    '''
    1、用labelImg标注原始图片，得到xml文件
    2、解析xml文件，并且在原图中截取需要的部分
    3. 将每个类别的图片分别放入对应的文件夹中。
    :return:
    '''
    for x in xmls:
        print(x)
        if x == 'desktop.ini':
            continue
        xml_dir = os.path.join(path,x)

        # 要解析的xml文件
        DOMTree = xml.dom.minidom.parse(xml_dir)
        collection = DOMTree.documentElement

        #获取图片名称
        filename = collection.getElementsByTagName('filename')[0].childNodes[0].data
        print(filename)

        #获取图片路径
        dir_path = collection.getElementsByTagName('path')[0].childNodes[0].data
        print(dir_path)
        #获取边框objects
        objects = collection.getElementsByTagName('object')
        # 路径中有中文，opencv没法解析，所以转成临时路径
        # name = collection.getElementsByTagName('name')[0].childNodes[0].data
        # dir_path = os.path.join(tmp_path,name,filename)
        # print(dir_path)

        #读取整张图片
        img = cv.imread(dir_path)
        # 可视化
        # cv.imshow('img',img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        #获取边框数量
        picture_num = len(objects)
        for idx,object in enumerate(objects):
            # 获取类别名name，窗口xmin,ymin,xmax,ymax坐标
            name = object.getElementsByTagName('name')[0].childNodes[0].data
            bndbox = object.getElementsByTagName('bndbox')[0]
            xmin = bndbox.getElementsByTagName('xmin')[0].childNodes[0].data
            ymin = bndbox.getElementsByTagName('ymin')[0].childNodes[0].data
            xmax = bndbox.getElementsByTagName('xmax')[0].childNodes[0].data
            ymax= bndbox.getElementsByTagName('ymax')[0].childNodes[0].data

            xmin,xmax,ymin,ymax=int(xmin),int(xmax),int(ymin),int(ymax)
            box = img[ymin:ymax,xmin:xmax]

            #将图片resize为相同大小
            # box = cv.resize(box, (118, 308))
            # box = np.asarray(box).resize((118,308,3))

            #截取图片保存位置
            picture_name = '{}_{}_{}'.format(picture_num, idx+1, filename)
            class_dir = os.path.join(classes_dir,name)
            #建类别路径文件夹
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            picture_save_full_name = os.path.join(class_dir,picture_name)
            # 图片保存
            cv.imwrite(picture_save_full_name,box)


def f2():
    xml_path = r'C:\Users\pc\Desktop\tmp\make_data\packing_bag\xml_cut_p1'
    xmls = os.listdir(xml_path)
    for xml_file in xmls:
        if xml_file == 'desktop.ini':
            continue
        xml_file_path = os.path.join(xml_path,xml_file)
        # 使用minidom解析器来打开XML文档
        print(xml_file_path)
        DOMTree = xml.dom.minidom.parse(xml_file_path)

        # 获取当前xml的整个数据集
        collection = DOMTree.documentElement

        # 获取集合中对应的文件名称filename
        filename = collection.getElementsByTagName('filename')[0].childNodes[0].data

        # 获取集合中所有的object
        objects = collection.getElementsByTagName('object')

        with open('./ob.txt', 'a', encoding='utf-8') as writer:
            dir_path = collection.getElementsByTagName('path')[0].childNodes[0].data
            writer.write('{}'.format(dir_path))
            # 遍历所有object
            for obj in objects:
                # 获取类别
                name = obj.getElementsByTagName("name")[0].childNodes[0].data
                # 获取坐标
                bndbox = obj.getElementsByTagName("bndbox")[0]
                xmin = bndbox.getElementsByTagName("xmin")[0].childNodes[0].data
                ymin = bndbox.getElementsByTagName("ymin")[0].childNodes[0].data
                xmax = bndbox.getElementsByTagName("xmax")[0].childNodes[0].data
                ymax = bndbox.getElementsByTagName("ymax")[0].childNodes[0].data

                # 构造输出对象
                values = "\t{},{},{},{},{}".format(xmin, ymin, xmax, ymax,name)
                writer.write(values)
            writer.write('\n')


if __name__ == '__main__':
    f1()