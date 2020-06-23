#coding:utf-8
# code for week2,recognize_computer_vision.py
# houchangligong,zhaomingming,20200602,
import torch
from torch import  nn
from itertools import product
import sys
from mnist import MNIST
import cv2
import numpy as np

def pdb():
    import pdb
    pdb.set_trace()
    pass

def model(feature,layers):
    y=-1
    #import pdb
    #pdb.set_trace()
    B = len(feature)
    fea=torch.tensor(feature).view(B,1,28,28).float()
    fea= torch.relu(layers[0](fea))
    # 完成lenet前向计算部分
    fea= torch.relu(layers[1](fea))
    fea= torch.relu(layers[2](fea))
    fea=torch.relu(layers[3](fea))
    fea=torch.relu(layers[4](fea))
    fea=fea.view(B,120)
    fea= torch.relu(layers[5](fea))
    #output= torch.relu(layers[6](fea))
    output= torch.sigmoid(layers[6](fea))
    y=output
    #pdb()
    #y=torch.softmax(output,1)
    #y = 1.0/(1.0+torch.exp(-1.*h))
    return y
def get_acc(image_data,image_label,layers,start_i,end_i):
    correct=0
    for i in range(start_i,end_i):
             y = model(image_data[i:i+1],layers)
             gt = image_label[i]
             pred = torch.argmax(y).item()
             if gt==pred:
                 correct+=1
    #print("acc=%s"%(float(correct/20.0)))
    return  float(correct/float(end_i-start_i))
def train_model(image_data,image_label,layers,lr):
    loss_value_before=1000000000000000.
    loss_value=10000000000000.
    #import pdb
    #pdb.set_trace()
    for epoch in range(0,1000):
        loss_value_before=loss_value
        loss_value=0
        #print(image_label[i])
        B = len(image_data)
        B = 80
        y = model(image_data[0:B],layers)
        gt=torch.tensor(image_label[0:B]).view(B,1)
        # get one_hot
        gt_vector = torch.zeros(B,10).scatter_(1,gt,1)
        #pdb.set_trace()
        # 关心所有值
        loss = torch.sum((y-gt_vector).mul(y-gt_vector))
        # 优化loss，正样本接近1，负样本远离1
        #loss1 = (y-1.0).mul(y-1.0)
        #loss = loss1[0,gt]+torch.sum(1.0/(loss1[0,0:gt]))+torch.sum(1.0/(loss1[0,gt:-1]))
        loss_value += loss.data.item()
        # 更新公式
        # w  = w - (y-y1)*x*lr
        loss.backward()
        for i in [0,2,4,5,6]: 
            layers[i].weight.data.sub_(layers[i].weight.grad.data*lr)
            layers[i].weight.grad.data.zero_()
            layers[i].bias.data.sub_(layers[i].bias.grad.data*lr)
            layers[i].bias.grad.data.zero_()
        train_acc=get_acc(image_data,image_label,layers,0,80)
        test_acc =get_acc(image_data,image_label,layers,80,100)
        print("epoch=%s,loss=%s/%s,train/test_acc:%s/%s"%(epoch,loss_value,loss_value_before,train_acc,test_acc))
    return layers 

if __name__=="__main__":
    # 从输入中获取学习率
    lr = float(sys.argv[1])
    
    layers=[]
    # add conv1 
    conv1=nn.Conv2d(1,6,kernel_size=(5,5),padding=(2,2))
    layers.append(conv1)
    pool2=nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    layers.append(pool2)
    # add conv3 
    conv3=nn.Conv2d(6,16,kernel_size=(5,5),padding=(0,0))
    layers.append(conv3)
    pool4=nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    layers.append(pool4)
    # add conv5 
    conv5=nn.Conv2d(16,120,kernel_size=(5,5),padding=(0,0))
    layers.append(conv5)
    f6 = nn.Linear(120, 84)
    layers.append(f6)
    output=nn.Linear(84,10)
    layers.append(output)
    # 记载数据
    # minst 2828 dataset 60000 samples
    mndata = MNIST('./python-mnist/data/')
    image_data_all, image_label_all = mndata.load_training()
    image_data=image_data_all[0:100]
    image_label=image_label_all[0:100]
    # 使用未训练的模型处理数据
    y = model(image_data,layers)
    # 使用为训练得模型测试 
    print("初始的未训练时模型的acc=%s"%(get_acc(image_data,image_label,layers,80,100)))
    # 对模型进行训练：
    train_model(image_data,image_label,layers,lr)
    # 训练完成，对模型进行测试，给出测试结果：
    print("训练完成后模型的acc=%s"%(get_acc(image_data,image_label,layers,80,100)))
