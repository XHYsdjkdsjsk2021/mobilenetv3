import sys
sys.path.append('./data')
sys.path.append('./model')

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from dataset import flowerDataset
from model import MobileNetV3_large
from model import MobileNetV3_small
import torchvision
from torch.autograd import Variable
import cv2
from PIL import Image
import pandas as pd
from tqdm import tqdm
import time

# 创建一个检测器类，包含了图片的读取，检测等方法

class Detector(object):
    # netkind为'large'或'small'可以选择加载MobileNetV3_large或MobileNetV3_small
    # 需要事先训练好对应网络的权重
    def __init__(self,net_kind,num_classes=143):
        super(Detector, self).__init__()
        kind=net_kind.lower()
        if kind=='large':
            self.net = MobileNetV3_large(num_classes=num_classes)
        elif kind=='small':
            self.net = MobileNetV3_small(num_classes=num_classes)
        self.net.eval()
        if torch.cuda.is_available():
            self.net.cuda()

    def load_weights(self,weight_path):
        self.net.load_state_dict(torch.load(weight_path))

    # 检测器主体
    def detect(self,weight_path,pic_path):
        # 先加载权重
        self.load_weights(weight_path=weight_path)
        # 读取图片
        img=Image.open(pic_path).convert('RGB')
        start_time = time.time()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0)
        if torch.cuda.is_available():
            img_tensor=img_tensor.cuda()
        with torch.no_grad():
            net_output = self.net(img_tensor)
            prob = torch.softmax(net_output, dim=1)
            confident, max_index = torch.max(prob, dim=1)
            # print(prob,"confidence：",confident[0].item(),"result：",max_index[0].item())
        stop_time = time.time()
        time_ = stop_time-start_time
        return confident[0].item(),max_index[0].item(),time_
            
        # print(net_output)
        # _, predicted = torch.max(net_output.data, 1)
        # result = predicted[0].item()
        # print("预测的结果为：",result)

# if __name__=='__main__':

#     detector=Detector('large',num_classes=143)
#     path = '/data/haoyuan/yolov5_test_result/'
#     cols = [ "pre_c","pre_r","now_c","now_r"]
#     df = pd.DataFrame(columns=cols)
#     df['path'] = [path + i for i in os.listdir(path)]
#     df.fillna(0.0, inplace=True)
#     for i in tqdm(df.iterrows()):
#         idx = i[0]
#         # frame = cv2.imread(i[1]['path'])
#         # df.iloc[idx, 0],df.iloc[idx, 1] = detector.detect('./weights/best_pre.pkl',i[1]['path'])
#         df.iloc[idx, 2],df.iloc[idx, 3] = detector.detect('./weights/best.pkl',i[1]['path'])
#     # df.to_csv('result.csv', index=None)
    
if __name__=='__main__':

    detector=Detector('large',num_classes=143)
    
    a,b,time_= detector.detect('./weights/best_large.pkl','/data/haoyuan/MobileNetV3-master/MobileNetV3-master/526.jpg')
    
    print("time:",time_)
    print(a,b)


        



