#coding:utf-8
#!/usr/bin/env python3.7

import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')   #解决cv2没法导入的问题

import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
from config import opt
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# from crnn import crnn
import crnn
# from warpctc_pytorch import CTCLoss
import torch.optim as optim
from torchvision import transforms
import torchvision
import collections
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

from torch.autograd import Variable
import torch.onnx as torch_onnx

writer = SummaryWriter()
img_h = opt.img_h
batch_size = opt.batch_size
use_gpu = opt.use_gpu
max_epoch = opt.max_epoch

'''
准确率近似评价
计算两字符串的最长公共字段长度
'''
def get_num(str1, str2):
	lstr1 = len(str1)
	lstr2 = len(str2)
	record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]  # 多一位
	maxNum = 0  # 最长匹配长度
	p = 0
	q = 0

	for i in range(lstr1):
		for j in range(lstr2):
			if str1[i] == str2[j]:
				# 相同则累加
				record[i + 1][j + 1] = record[i][j] + 1
				if record[i + 1][j + 1] > maxNum:
					# 获取最大匹配长度
					maxNum = record[i + 1][j + 1]
					p = i + 1
					q = i + i
				# 记录最大匹配长度的终止位置
	return str1[p - maxNum:p], maxNum


def delete_substr_method(in_str, in_substr):
	start_loc = in_str.find(in_substr)
	len_substr = len(in_substr)
	res_str = in_str[:start_loc] + in_str[start_loc + len_substr:]
	return res_str


def get_accuracy(str1_1, str2_1):
	alphabetChinese = r' 1234567890qwerTYUIOPASDFGHtyuiopasdfghjklzxcvbnmQWERJKLZXCVBNM/-!.'
	str1 = [alphabetChinese[i] for i in str1_1]
	str1 = ''.join(str1)
	str2 = [alphabetChinese[i] for i in str2_1]
	str2 = ''.join(str2)
	le = max(len(str1),len(str2))
	max_steps = 4
	i = 0
	num = 0
	while len(str1) > 0 and len(str2) > 0 and i < max_steps:
		same_str, num1 = get_num(str1, str2)
		str1 = delete_substr_method(str1, same_str)
		str2 = delete_substr_method(str2, same_str)
		num += num1
		i += 1

	return (num / le) * 100, num
	# if len(str1) > 0 and len(str2) > 0:
	# 	same_str, num2 = get_num(str1, str2)
	# 	return ((num1 + num2) / le) * 100 , num1 + num2
	# return (num1 / le) * 100 , num1

#读取标签文件，生成字典。{图片名称：标签值列表}
def readfile(filename):

	res = []
	with open(filename, 'r',encoding='utf-8') as f:
		lines = f.readlines()
		for i in lines:
			res.append(i.strip())
			# strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
	dic = {}
	for i in res:
		p = i.split(' ')
		dic[p[0]] = p[1:]
	return dic

#数据标准化
class resizeNormalize(object):
	def __init__(self, size, interpolation=Image.BILINEAR):  # Image.BILINEAR：双线性插值对图像进行缩放
		self.size = size
		self.interpolation = interpolation
		self.toTensor = transforms.ToTensor()

	def __call__(self, img):
		img = img.resize(self.size, self.interpolation)
		img = self.toTensor(img)
		# img.sub_(0.5).div_(0.5)
		return img



class Chineseocr(Dataset):

	def __init__(self, imageroot, labelroot):
		self.image_dict = readfile(labelroot)
		self.image_root = imageroot
		self.image_name = [filename for filename, _ in self.image_dict.items()]
		# items()以列表形式返回可遍历的（键，值）元组数据

	def __getitem__(self, index):

		img_path = os.path.join(self.image_root, self.image_name[index])
		keys = self.image_dict.get(self.image_name[index])
		label = [int(x) for x in keys]

		Data = Image.open(img_path).convert('L')
		(w,h) = Data.size
		size_h = 32
		ratio = 32 / float(h)
		size_w = int(w * ratio)
		transform = resizeNormalize((size_w,size_h))
		Data = transform(Data)
		label=torch.IntTensor(label)

		return Data,label

	def __len__(self):
		return len(self.image_name)



train_data = Chineseocr(
	imageroot = opt.train_image_path,
	labelroot = opt.train_data_root
)
train_loader = DataLoader(
	train_data,
	batch_size = opt.batch_size,
	shuffle = True,
	num_workers = opt.num_workers
)

val_data = Chineseocr(
		imageroot = opt.valid_image_path,
		labelroot = opt.valid_data_root
	)
val_loader = DataLoader(
	val_data,
	batch_size = opt.batch_size,
	shuffle = True,
	num_workers = opt.num_workers
)

def decode(preds):
	pred = []
	for i in range(len(preds)):
		if preds[i] != 0 and ((i == 0) or (i != 0 and preds[i] != preds[i-1])):
			pred.append(int(preds[i]))
	return pred

def val(net,loss_func,max_iter = 50):
	print('start val')
	net.eval()
	totalloss = 0.0
	k = 0
	correct_num = 0
	total_num = 0
	val_iter = iter(val_loader)
	max_iter = min(max_iter, len(val_loader))
	for i in range(max_iter):
		k = k + 1
		(data,label) = val_iter.next()
		print()
		labels = torch.IntTensor([])
		for j in range(label.size(0)):
			labels = torch.cat((labels, label[j]),0)
		if torch.cuda.is_available and use_gpu:
			data = data.cuda()
		output = net(data)
		print('output.size():',output.size())
		print('label.size():',label.size())

		output_size = torch.IntTensor([output.size(0)] * int(output.size(1)))
		label_size = torch.IntTensor([label.size(1)] * int(label.size(0)))
		loss = loss_func(output, labels, output_size, label_size) / label.size(0)
		totalloss += float(loss)
		pred_label = output.max(2)[1]
		pred_label = pred_label.transpose(1, 0).contiguous().view(-1)
		pred = decode(pred_label)
		total_num += len(pred)
		# for x,y in zip(pred, labels):
		# 	if int(x) == int(y):
		# 		correct_num += 1
		'''改写准确率评价方法'''
		acc, num = get_accuracy(pred, labels)
		correct_num += num
		print('total_num:{},correct_num:{}'.format(total_num,correct_num))
		print('len:{},num:{}'.format(len(pred),num))
		print(acc)

	#防止除数为0
	try:
		accuracy = correct_num / float(total_num) * 100
	except:
		accuracy = correct_num / float(total_num+0.1) * 100
	test_loss = totalloss / k
	print('Test loss : %.3f , accuary : %.3f%%' % (test_loss , accuracy))
	return accuracy



import argparse   #外部输入参数

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-lr', type=float, default=opt.learning_rate, help='initial learning rate')
	args = parser.parse_args()

	# char_set = open('char_std_5990.txt', 'r', encoding='utf-8').readlines()
	char_set = open('./char.txt', 'r', encoding='utf-8').readlines()
	char_set = ''.join([ch.strip('\n') for ch in char_set[1:]] + ['卍'])
	n_class = len(char_set)
	model = crnn.CRNN(img_h, 1, n_class, 256)
	if torch.cuda.is_available and use_gpu:
		model.cuda()

	# modelpath = opt.modelpath
	modelpath = opt.model_path

	# learning_rate = opt.learning_rate
	learning_rate = args.lr
	# loss_func = CTCLoss()
	loss_func = torch.nn.CTCLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=opt.weight_decay)
	# optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=opt.weight_decay)

	if os.path.exists(modelpath):
		print('Load model from "%s" ...' % modelpath)
		model.load_state_dict(torch.load(modelpath))
		print('Done!')
	k = 0
	losstotal = 0.0
	# printinterval = opt.printinterval
	printinterval = opt.print_interval
	# valinterval = opt.valinterval
	valinterval = opt.valid_interval
	numinprint = 0

	train_accuracy = 0
	test_accuracy = 0
	# train
	for epoch in range(max_epoch):

		for i,(data,label) in enumerate(train_loader):
			total_num = 0
			correct_num = 0

			# learning_rate = learning_rate * 0.9 ** (k // 100)
			k = k + 1
			numinprint = numinprint + 1
			if torch.cuda.is_available and use_gpu:
				data = data.cuda()
				loss_func = loss_func.cuda()
			model.train()
			labels = torch.IntTensor([])
			for j in range(label.size(0)):
				labels = torch.cat((labels, label[j]),0)

			output = model(data)
			output_size = torch.IntTensor([output.size(0)] * int(output.size(1)))

			label_size = torch.IntTensor([label.size(1)] * int(label.size(0)))

			loss = loss_func(output,labels,output_size,label_size) / label.size(0)
			# loss = loss_func(output,labels,torch.tensor([32,32,200]),torch.tensor([0,0,0])) / label.size(0)
			losstotal += float(loss)

			#增加准确率判断
			pred_label = output.max(2)[1]
			pred_label = pred_label.transpose(1, 0).contiguous().view(-1)
			pred = decode(pred_label)
			total_num += len(pred)

			# for x, y in zip(pred, labels):
			# 	if int(x) == int(y):
			# 		correct_num += 1
			#防止除数为0
			# try:
			# 	train_accuracy = correct_num / float(total_num) * 100
			# except:
			# 	train_accuracy = correct_num / float(total_num+0.01) * 100
			'''改写准确率评价方法'''
			train_accuracy,_ = get_accuracy(pred,labels)

			# '''按准确率，选择展示图片'''
			# if train_accuracy <= 20:
			# 	alphabetChinese = r' 1234567890qwerTYUIOPASDFGHtyuiopasdfghjklzxcvbnmQWERJKLZXCVBNM/-!.'
			# 	print('长度：',len(pred))
			# 	for i in range(len(pred)):
			# 		print(alphabetChinese[pred[i]],end='')
			# 		if (i+1) % 12 == 0:
			# 			print()
			# 	img = torchvision.utils.make_grid(data.cpu(),nrow=2).numpy()
			# 	plt.imshow(np.transpose(img, (1, 2, 0)))
			# 	plt.show()
			# if train_accuracy == 100:
			# 	print('{}:label:{}'.format(i,len(label)))
			# 	print('{}:pred:{}'.format(i,len(pred)))

			if k % printinterval == 0:
				# display
				print(len(pred), len(labels))
				print("[%d/%d] || [%d/%d] || Loss:%.6f  || Acc:%.4f" % (
					epoch, max_epoch, i + 1, len(train_loader), losstotal / numinprint, train_accuracy))
				losstotal = 0.0
				numinprint = 0
				# torch.save(model.state_dict(), opt.modelpath)
				torch.save(model.state_dict(), opt.model_path)
			writer.add_scalar('loss', loss, k)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if k % valinterval == 0:
				# val
				test_accuracy = val(model, loss_func)


	if train_accuracy >= 90 and test_accuracy >= 90:
		torch.save(model.state_dict(), opt.model_path)
		input_shape = (1, 32, 280)
		model_onnx_path = "torch_model_train_Acc{:.3f}_test_Acc{:.3f}.onnx".format(train_accuracy, test_accuracy)
		model.train(False)
		# 导出onnx模型
		dummy_input = Variable(torch.randn(1, *input_shape))
		if torch.cuda.is_available():
			dummy_input = dummy_input.cuda()
		output = torch_onnx.export(model,
								   dummy_input,
								   model_onnx_path,
								   verbose=False)
		print('onnx转出成功')
		# print('epoch : %05d || loss : %.3f' % (epoch, losstotal/numinepoch))

	# writer.export_scalars_to_json("./all_scalars.json")
	# writer.close()