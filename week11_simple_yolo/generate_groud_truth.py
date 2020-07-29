# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 23:16:40 2020

@author: 16600
"""
import json
import xml.dom.minidom
import os 
import numpy as np
anno=[]
img=[]
onehot=[[0]*14 for _ in range(14)]
for i in range(14):
    onehot[i][i]=1
encoder={'农夫山泉':0,'冰露':1,'娃哈哈':2,'可口可乐':3,'天府可乐':4,'其他':5,
         '康师傅':6,'百事可乐':7,'怡宝':8,'百岁山':9,'苏打水':10,'景甜':11,'恒大冰泉':12,'今麦郎':13}
f1=open("./V006.manifest",'r', encoding='UTF-8')
for line in f1:
    dic = json.loads(line)
    anno.append(dic["annotation"][0]["annotation-loc"].split('V006')[-1])
    img.append(dic["source"].split('image')[-1])

label=[]
c=0
dim=7
for i in anno:
    sample=[[[0]*(5+14) for _ in range(dim)] for _ in range(dim)]
    DomTree = xml.dom.minidom.parse('.'+i)
    annotation = DomTree.documentElement
    objectlist = annotation.getElementsByTagName('object')
    size = annotation.getElementsByTagName('size')
    widl = size[0].getElementsByTagName('width')
    wid = int(widl[0].childNodes[0].data)
    heil = size[0].getElementsByTagName('height')
    hei = int(heil[0].childNodes[0].data)
    for objects in objectlist:
        namelist = objects.getElementsByTagName('name')
        objectname = namelist[0].childNodes[0].data
        class_=onehot[encoder[objectname]]
        bndbox = objects.getElementsByTagName('bndbox')
        for box in bndbox:
            x1_list = box.getElementsByTagName('xmin')
            x1 = int(x1_list[0].childNodes[0].data)*1.0/wid
            y1_list = box.getElementsByTagName('ymin')
            y1 = int(y1_list[0].childNodes[0].data)*1.0/hei
            x2_list = box.getElementsByTagName('xmax')
            x2 = int(x2_list[0].childNodes[0].data)*1.0/wid
            y2_list = box.getElementsByTagName('ymax')
            y2 = int(y2_list[0].childNodes[0].data)*1.0/hei
            w = x2 - x1
            h = y2 - y1
            c_x=(x2+x1)/2
            c_y=(y2+y1)/2
            i,j=0,0
            for n in range(1,dim):
                if c_x>n*1.0/dim and c_x<=(n+1)*1.0/dim:i=n
                if c_y>n*1.0/dim and c_y<=(n+1)*1.0/dim:j=n
            if sample[i][j]==[0]*19:
                sample[i][j]=[x1,y1,w,h,1]+class_
    label.append(sample)
            
label=np.array(label)
np.save("label.npy",label)
np.save("anno.npy",np.array(anno))
np.save("img.npy",np.array(img))
