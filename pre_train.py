import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_class=20

#ImageNet Pre-Training
class Pretrain_Layers(nn.Module):
  def __init__(self):
    super(Pretrain_Layers, self).__init__()
    self.conv1=nn.Conv2d(3, 64, (7,7), stride=2, padding=3).to(device)
    self.maxpool1=nn.MaxPool2d((2,2), 2).to(device)

    self.conv2=nn.Conv2d(64, 192, (3,3), stride=1, padding=1).to(device)
    self.maxpool2=nn.MaxPool2d((2,2), 2).to(device)

    self.conv3=nn.Conv2d(192, 128, (1,1), stride=1, padding=0).to(device)
    self.conv4=nn.Conv2d(128, 256, (3,3), stride=1, padding=1).to(device)
    self.conv5=nn.Conv2d(256, 256, (1,1), stride=1, padding=0).to(device)
    self.conv6=nn.Conv2d(256, 512, (3,3), stride=1, padding=1).to(device)
    self.maxpool3=nn.MaxPool2d((2,2), 2).to(device)

    self.conv7=nn.Conv2d(512, 256, (1,1), stride=1, padding=0).to(device)
    self.conv8=nn.Conv2d(256, 512, (3,3), stride=1, padding=1).to(device)
    self.conv9=nn.Conv2d(512, 256, (1,1), stride=1, padding=0).to(device)
    self.conv10=nn.Conv2d(256, 512, (3,3), stride=1, padding=1).to(device)
    self.conv11=nn.Conv2d(512, 256, (1,1), stride=1, padding=0).to(device)
    self.conv12=nn.Conv2d(256, 512, (3,3), stride=1, padding=1).to(device)
    self.conv13=nn.Conv2d(512, 256, (1,1), stride=1, padding=0).to(device)
    self.conv14=nn.Conv2d(256, 512, (3,3), stride=1, padding=1).to(device)
    self.conv15=nn.Conv2d(512, 512, (1,1), stride=1, padding=0).to(device)
    self.conv16=nn.Conv2d(512, 1024, (3,3), stride=1, padding=1).to(device)
    self.maxpool4=nn.MaxPool2d((2,2), 2).to(device)

    self.conv17=nn.Conv2d(1024, 512, (1,1), stride=1, padding=0).to(device)
    self.conv18=nn.Conv2d(512, 1024, (3,3), stride=1, padding=1).to(device)
    self.conv19=nn.Conv2d(1024, 512, (1,1), stride=1, padding=0).to(device)
    self.conv20=nn.Conv2d(512, 1024, (3,3), stride=1, padding=1).to(device)

    self.pt_fc=nn.Linear(1024, N_class).to(device)

    self.element_init()
  
  def element_init(self):
    for element in self.children():
      if isinstance(element, nn.Conv2d):
        nn.init.xavier_uniform_(element.weight)
        nn.init.constant_(element.bias, 0.)
      elif isinstance(element, nn.Linear):
        nn.init.xavier_uniform_(element.weight)
        nn.init.constant_(element.bias, 0.)
  
  def train(self, images, cls_labels):
    lr=nn.LeakyReLU(0.1, inplace=True).to(device)
    dropout=nn.Dropout(p=0.5)
    first_20_convs=nn.Sequential(self.conv1, lr, self.maxpool1, 
                        self.conv2, lr, self.maxpool2, 
                        self.conv3, lr, self.conv4, lr, self.conv5, lr, self.conv6, lr, self.maxpool3, 
                        self.conv7, lr, self.conv8, lr, self.conv9, lr, self.conv10, self.conv11, lr, self.conv12, lr, self.conv13, lr, 
                        self.conv14, lr, self.conv15, lr, self.conv16, lr, self.maxpool4,
                        self.conv17, lr, self.conv18, lr, self.conv19, lr, self.conv20, lr)
    fm=first_20_convs(images)
    batch,_,f_h, f_w=fm.size()
    gt_cls_vector=torch.zeros(batch, N_class).to(device)
    for idx, cls_label in enumerate(cls_labels):
      gt_cls_vector[idx,cls_label]=1.
    avg_pool=nn.AvgPool2d((14,14), stride=14)
    fm=avg_pool(fm)
    cv=fm.reshape(batch,-1)
    linear_layer=nn.Sequential(self.pt_fc, dropout, lr)
    cls_vector=linear_layer(cv)
    pt_loss=F.cross_entropy(cls_vector, gt_cls_vector)
    return pt_loss
  
  def test(self, images):
    lr=nn.LeakyReLU(0.1, inplace=True).to(device)
    dropout=nn.Dropout(p=0.5)
    first_20_convs=nn.Sequential(self.conv1, lr, self.maxpool1, 
                        self.conv2, lr, self.maxpool2, 
                        self.conv3, lr, self.conv4, lr, self.conv5, lr, self.conv6, lr, self.maxpool3, 
                        self.conv7, lr, self.conv8, lr, self.conv9, lr, self.conv10, self.conv11, lr, self.conv12, lr, self.conv13, lr, 
                        self.conv14, lr, self.conv15, lr, self.conv16, lr, self.maxpool4,
                        self.conv17, lr, self.conv18, lr, self.conv19, lr, self.conv20, lr)
    fm=first_20_convs(images)
    batch,_,f_h, f_w=fm.size()
    avg_pool=nn.AvgPool2d((14,14),stride=14)
    cv=fm.reshape(batch,-1)
    linear_layer=nn.Sequential(self.pt_fc, dropout, lr)
    cls_vector=linear_layer(cv)
    return cls_vector