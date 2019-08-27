import csv   #操作csv文件的
import os    #读写文件
import numpy as np  #操作数组
from PIL import Image   #操纵图像的

data_path = './fer2013.csv'
train_csv = './train.csv'
val_csv = './val.csv'
test_csv = './test.csv'
train_set = r'./data/train'
test_set = r'./data/test'
val_set = r'./data/val'

with open(data_path) as f:    #开始操纵文档
    csvr = csv.reader(f)
    rows = [row for row in csvr]

    train = [row[:-1] for row in rows if row[-1] == 'Training']
    csv.writer(open(train_csv, 'w+'), lineterminator='\n').writerows(train)
    print(len(train))   #28709

    evaluate = [row[:-1] for row in rows if row[-1] == 'PublicTest']
    csv.writer(open(val_csv, 'w+'), lineterminator='\n').writerows(evaluate)
    print(len(evaluate))   #3589

    test = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
    csv.writer(open(test_csv, 'w+'), lineterminator='\n').writerows(test)
    print(len(test))    #3589


for save_path, csv_file in [(train_set, train_csv), (test_set, test_csv), (val_set, val_csv)]:

    if not os.path.exists(save_path):
        os.makedirs(save_path)         #创建文件夹

    with open(csv_file) as f:
        csvr = csv.reader(f)
        for i, (label, pixel) in enumerate(csvr):
            pixel = np.array(pixel).reshape(48, 48).astype('float32')
            im = Image.fromarray(pixel).convert('L')
            subfolder = os.path.join(save_path, label)
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            image_name = os.path.join(subfolder, '{:05d}.jpg'.format(i))
            im.save(image_name)
