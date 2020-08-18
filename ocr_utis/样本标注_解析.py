path1 = './char.txt'   #字典exel
test_path_tmp = r'F:\pc\work\test_python\test\OCR\Train\data\lv_generate_txt\test_label_20200702.txt'  #测试的标签中间文件
test_path = r'F:\pc\work\test_python\test\OCR\Train\data\lv_generate_txt\char_test.txt'                    #测试的标签文件
train_path_tmp = r'F:\pc\work\test_python\test\OCR\Train\data\lv_generate_txt\train_label_20200702.txt'#训练的标签中间文件
train_path = r'F:\pc\work\test_python\test\OCR\Train\data\lv_generate_txt\char_train.txt'                  #训练的标签文件


flag = 'train'      #输入：'train'  or  'test'

path2 = path3 = ''
if flag == 'train':
    path2 = train_path_tmp
    path3 = train_path
elif flag == 'test':
    path2 = test_path_tmp
    path3 = test_path


"""将字符与索引对应，生成键值匹配的字典"""
dic1 = {}
with open(path1,'r') as f:
    lines = f.readlines()
    for idx,line in enumerate(lines):
        line = line.strip('\n')
        dic1[idx] = line
# print(dic1)


"""读取标签文件,存入字典列表中"""
dic2 = {}
with open(path2,'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        key,value = line.split(' ')
        dic2[key] = value
# print(dic2)


"""根据字典，将列表中的值对应到字典中"""
all_str = []
for key in dic2:
    # str_tmp = key + '.jpg'
    str_tmp = key
    str_dic2 = dic2[key]
    for char1 in str_dic2:
        #将a替换成'/'
        if char1 == '#':        #需要替换的字符
            char1 = '/'
        for idx,char2 in dic1.items():
            if char1 == char2:
                str_tmp += ' ' + str(idx)
    all_str.append(str_tmp)


"""将键值转换后的内容写入exel中，用于训练或测试读取的目录"""
with open(path3,'w') as f:
    for line in all_str:
        f.write(line+'\n')