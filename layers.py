import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Conv_Layers(nn.Module):
  def __init__(self, pretrain_layers):
    super(Conv_Layers, self).__init__()
    #inherit weight, bias of conv1~20 from pretrain_layers
    self.conv1=nn.Conv2d(3, 64, (7,7), stride=2, padding=3).to(device)
    self.conv1.weight=nn.Parameter(pretrain_layers.conv1.weight)
    self.conv1.bias=nn.Parameter(pretrain_layers.conv1.bias)
    self.maxpool1=nn.MaxPool2d((2,2), 2).to(device)

    self.conv2=nn.Conv2d(64, 192, (3,3), stride=1, padding=1).to(device)
    self.conv2.weight=nn.Parameter(pretrain_layers.conv2.weight)
    self.conv2.bias=nn.Parameter(pretrain_layers.conv2.bias)
    self.maxpool2=nn.MaxPool2d((2,2), 2).to(device)

    self.conv3=nn.Conv2d(192, 128, (1,1), stride=1, padding=0).to(device)
    self.conv3.weight=nn.Parameter(pretrain_layers.conv3.weight)
    self.conv3.bias=nn.Parameter(pretrain_layers.conv3.bias)

    self.conv4=nn.Conv2d(128, 256, (3,3), stride=1, padding=1).to(device)
    self.conv4.weight=nn.Parameter(pretrain_layers.conv4.weight)
    self.conv4.bias=nn.Parameter(pretrain_layers.conv4.bias)

    self.conv5=nn.Conv2d(256, 256, (1,1), stride=1, padding=0).to(device)
    self.conv5.weight=nn.Parameter(pretrain_layers.conv5.weight)
    self.conv5.bias=nn.Parameter(pretrain_layers.conv5.bias)

    self.conv6=nn.Conv2d(256, 512, (3,3), stride=1, padding=1).to(device)
    self.conv6.weight=nn.Parameter(pretrain_layers.conv6.weight)
    self.conv6.bias=nn.Parameter(pretrain_layers.conv6.bias)
    self.maxpool3=nn.MaxPool2d((2,2), 2).to(device)

    self.conv7=nn.Conv2d(512, 256, (1,1), stride=1, padding=0).to(device)
    self.conv7.weight=nn.Parameter(pretrain_layers.conv7.weight)
    self.conv7.bias=nn.Parameter(pretrain_layers.conv7.bias)
    self.conv8=nn.Conv2d(256, 512, (3,3), stride=1, padding=1).to(device)
    self.conv8.weight=nn.Parameter(pretrain_layers.conv8.weight)
    self.conv8.bias=nn.Parameter(pretrain_layers.conv8.bias)
    self.conv9=nn.Conv2d(512, 256, (1,1), stride=1, padding=0).to(device)
    self.conv9.weight=nn.Parameter(pretrain_layers.conv9.weight)
    self.conv9.bias=nn.Parameter(pretrain_layers.conv9.bias)
    self.conv10=nn.Conv2d(256, 512, (3,3), stride=1, padding=1).to(device)
    self.conv10.weight=nn.Parameter(pretrain_layers.conv10.weight)
    self.conv10.bias=nn.Parameter(pretrain_layers.conv10.bias)
    self.conv11=nn.Conv2d(512, 256, (1,1), stride=1, padding=0).to(device)
    self.conv11.weight=nn.Parameter(pretrain_layers.conv11.weight)
    self.conv11.bias=nn.Parameter(pretrain_layers.conv11.bias)
    self.conv12=nn.Conv2d(256, 512, (3,3), stride=1, padding=1).to(device)
    self.conv12.weight=nn.Parameter(pretrain_layers.conv12.weight)
    self.conv12.bias=nn.Parameter(pretrain_layers.conv12.bias)
    self.conv13=nn.Conv2d(512, 256, (1,1), stride=1, padding=0).to(device)
    self.conv13.weight=nn.Parameter(pretrain_layers.conv13.weight)
    self.conv13.bias=nn.Parameter(pretrain_layers.conv13.bias)
    self.conv14=nn.Conv2d(256, 512, (3,3), stride=1, padding=1).to(device)
    self.conv14.weight=nn.Parameter(pretrain_layers.conv14.weight)
    self.conv14.bias=nn.Parameter(pretrain_layers.conv14.bias)
    self.conv15=nn.Conv2d(512, 512, (1,1), stride=1, padding=0).to(device)
    self.conv15.weight=nn.Parameter(pretrain_layers.conv15.weight)
    self.conv15.bias=nn.Parameter(pretrain_layers.conv15.bias)
    self.conv16=nn.Conv2d(512, 1024, (3,3), stride=1, padding=1).to(device)
    self.conv16.weight=nn.Parameter(pretrain_layers.conv16.weight)
    self.conv16.bias=nn.Parameter(pretrain_layers.conv16.bias)
    self.maxpool4=nn.MaxPool2d((2,2), 2).to(device)

    self.conv17=nn.Conv2d(1024, 512, (1,1), stride=1, padding=0).to(device)
    self.conv17.weight=nn.Parameter(pretrain_layers.conv17.weight)
    self.conv17.bias=nn.Parameter(pretrain_layers.conv17.bias)
    self.conv18=nn.Conv2d(512, 1024, (3,3), stride=1, padding=1).to(device)
    self.conv18.weight=nn.Parameter(pretrain_layers.conv18.weight)
    self.conv18.bias=nn.Parameter(pretrain_layers.conv18.bias)
    self.conv19=nn.Conv2d(1024, 512, (1,1), stride=1, padding=0).to(device)
    self.conv19.weight=nn.Parameter(pretrain_layers.conv19.weight)
    self.conv19.bias=nn.Parameter(pretrain_layers.conv19.bias)
    self.conv20=nn.Conv2d(512, 1024, (3,3), stride=1, padding=1).to(device)
    self.conv20.weight=nn.Parameter(pretrain_layers.conv20.weight)
    self.conv20.bias=nn.Parameter(pretrain_layers.conv20.bias)

    self.conv21=nn.Conv2d(1024, 1024, (3,3), stride=1, padding=1).to(device)
    nn.init.xavier_uniform_(self.conv21.weight)
    nn.init.constant_(self.conv21.bias, 0.)
    self.conv22=nn.Conv2d(1024, 1024, (3,3), stride=2, padding=1).to(device)
    nn.init.xavier_uniform_(self.conv22.weight)
    nn.init.constant_(self.conv22.bias, 0.)
    self.conv23=nn.Conv2d(1024, 1024, (3,3), stride=1, padding=1).to(device)
    nn.init.xavier_uniform_(self.conv23.weight)
    nn.init.constant_(self.conv23.bias, 0.)
    self.conv24=nn.Conv2d(1024, 1024, (3,3), stride=1, padding=1).to(device)
    nn.init.xavier_uniform_(self.conv24.weight)
    nn.init.constant_(self.conv24.bias, 0.)
    
  def forward(self, images):
    lr=nn.LeakyReLU(0.1, inplace=True).to(device)
    convs=nn.Sequential(self.conv1, lr, self.maxpool1, 
                        self.conv2, lr, self.maxpool2, 
                        self.conv3, lr, self.conv4, lr, self.conv5, lr, self.conv6, lr, self.maxpool3, 
                        self.conv7, lr, self.conv8, lr, self.conv9, lr, self.conv10, self.conv11, lr, self.conv12, lr, self.conv13, lr, 
                        self.conv14, lr, self.conv15, lr, self.conv16, lr, self.maxpool4,
                        self.conv17, lr, self.conv18, lr, self.conv19, lr, self.conv20, lr, self.conv21, lr, self.conv22, lr,
                        self.conv23, lr, self.conv24, lr)
    featmap=convs(images)
    return featmap

