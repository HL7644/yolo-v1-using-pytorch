import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as dsets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import random

import json
import pandas as pd
import os.path
import xml.etree.ElementTree as ET
import PIL
from google.colab import drive

#import other python files
from image_loader import *
from pre_train import *
from layers import *
from ftns_for_yolo_loss import *
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#create class dictionary for PASCAL-VOC images (20 classes)
N_class=20
cls_to_clslabel=dict()
clslabel_to_cls=dict()
cls_to_clslabel['aeroplane']=1
cls_to_clslabel['bicycle']=2
cls_to_clslabel['bird']=3
cls_to_clslabel['boat']=4
cls_to_clslabel['bottle']=5
cls_to_clslabel['bus']=6
cls_to_clslabel['car']=7
cls_to_clslabel['cat']=8
cls_to_clslabel['chair']=9
cls_to_clslabel['cow']=10
cls_to_clslabel['diningtable']=11
cls_to_clslabel['dog']=12
cls_to_clslabel['horse']=13
cls_to_clslabel['motorbike']=14
cls_to_clslabel['person']=15
cls_to_clslabel['pottedplant']=16
cls_to_clslabel['sheep']=17
cls_to_clslabel['sofa']=18
cls_to_clslabel['train']=19
cls_to_clslabel['tvmonitor']=20
for val, key in enumerate(cls_to_clslabel):
  clslabel_to_cls[val+1]=key

#collect image data from p-voc directory in colab
images_data=get_pvoc_images(N=64)

#pretrain first 20 convs  (using: imagenet)
#imagenet data: too large for colab storage

pretrain_layers=Pretrain_Layers()
#optimizer_pt=optim.SGD(pretrain_layers.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
#Pre-Train Network
#iter=0
#for epoch in range(1, iter+1):
#  cost=pretrain_layers.train(images, cls_labels)

#  optimizer_pt.zero_grad()
#  cost.backward() 
#  optimizer_pt.step()
#  if epoch%10==0:
#    print("Epoch: {:d}, Cost: {:.2f}".format(epoch, cost.item()))

class YOLOv1(nn.Module):
  def __init__(self):
    super(YOLOv1, self).__init__()
    self.conv_layers=Conv_Layers(pretrain_layers)
    self.fc_layers=FC_Layers()
  
  def train(self, images, bboxes, cls_labels):
    #getting total loss
    B=2
    S=7
    C=20
    yolo_loss=torch.FloatTensor([0]).to(device)
    fm=self.conv_layers(images)
    featmap=self.fc_layers(fm)
    featmap=torch.nan_to_num(featmap)
    batch,feat_h,feat_w,_=featmap.size()
    grid_boxes=get_yolo_grids(images.size(), feat_h, feat_w)
    for b_idx in range(batch):
      bboxes_img=bboxes[b_idx]
      cls_labels_img=cls_labels[b_idx]
      #per image yolo loss
      #for each grid cell
      for row in range(feat_h):
        for col in range(feat_w):
          grid_box=grid_boxes[row,col]
          feature_vector=featmap[b_idx,row,col] #30 dimensional
          yolo_loss=yolo_loss+torch.nan_to_num(get_loss(bboxes_img, cls_labels_img, grid_box, feature_vector))
    return yolo_loss

  def test(self, images):
    B=2
    S=7
    C=20
    objects=[]
    classes=[]
    fm=self.conv_layers(images)
    featmap=self.fc_layers(fm)
    batch,feat_h,feat_w,_=featmap.size()
    grids=get_yolo_grids(images.size(), feat_h, feat_w)
    box_vectors=torch.zeros(batch, feat_h, feat_w, B, 5).to(device)
    class_vectors=torch.zeros(batch, feat_h, feat_w, B, C).to(device)
    for b_idx in range(batch):
      object_vectors_batch=torch.Tensor([]).to(device)
      class_batch=[]
      for row in range(feat_h):
        for col in range(feat_w):
          grid_box=grids[row,col]
          feature_vector=featmap[b_idx,row,col]
          #box vectors
          for pred_idx in range(B):
            #5 dimensional: 4 coordinates + confidence
            box_vectors[b_idx,row,col,pred_idx,:]=feature_vector[pred_idx*5:(pred_idx+1)*5]
          #class vectors
          class_vector=feature_vector[5*B:]
          cls=torch.argmax(class_vector, dim=0)
          #analyze conf
          conf_vector=box_vectors[b_idx,row,col,:,4] #compare confidence
          resp_idx=torch.argmax(conf_vector, dim=0)
          resp_conf=conf_vector[resp_idx]
          #get resp_box if resp_conf is above certain threshold
          if resp_conf>0.05:
            resp_box_vector=box_vectors[b_idx,row,col,resp_idx]
            prediction=get_prediction(grid_box, resp_box_vector)
            object_vectors_batch=torch.cat((object_vectors_batch, prediction.unsqueeze(dim=0)), dim=0)
            class_batch.append(cls)
      class_batch=torch.LongTensor(class_batch)
      classes.append(class_batch)
      #perform nms within predictions of an image
      objects_batch, classes_batch=nms(object_vectors_batch, class_batch)
      objects.append(objects_batch)
      classes.append(classes_batch)
    return objects, classes

yolov1=YOLOv1()
optimizer=optim.SGD(yolov1.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
dataloader=torch.utils.data.DataLoader(images_data, batch_size=64, shuffle=True)

iter=20
for epoch in range(1,iter+1):
    for batch_data in dataloader:
        images, bboxes, cls_labels=batch_data
        cost=yolov1.train(images, bboxes, cls_labels)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        #change learning rate
    if epoch==5:
        optimizer.param_groups[0]['lr']=1e-2
    elif epoch==75:
        optimizer.param_groups[0]['lr']=1e-3
    elif epoch==105:
        optimizer.param_groups[0]['lr']=1e-4
    if epoch%1==0:
        print('Epoch: {:d}, Cost: {:.2f}'.format(epoch, cost.item()))