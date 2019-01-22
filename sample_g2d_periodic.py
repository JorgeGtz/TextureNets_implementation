# SAMPLE GENERATOR PYRAMID 2D PERIODIC
#
# Code for the texture synthesis method in:
# Ulyanov et al. Texture Networks: Feed-forward Synthesis of Textures and Stylized Images
# https://arxiv.org/abs/1603.03417
# Generator architecture fixed to 6 scales!
#
# Author: Jorge Gutierrez
# Creation:  22 Jan 2019
# Last modified: 22 Jan 2019
# Based on https://github.com/leongatys/PytorchNeuralStyleTransfer
import math
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms


#generator's convolutional blocks 2D
class Conv_block2D(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, m=0.1):
        super(Conv_block2D, self).__init__()

        self.conv1 = nn.Conv2d(n_ch_in, n_ch_out, 3, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(n_ch_out, momentum=m)
        self.conv2 = nn.Conv2d(n_ch_out, n_ch_out, 3, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(n_ch_out, momentum=m)
        self.conv3 = nn.Conv2d(n_ch_out, n_ch_out, 1, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(n_ch_out, momentum=m)

    def forward(self, x):
        x = torch.cat((x[:,:,-1,:].unsqueeze(2),x,x[:,:,0,:].unsqueeze(2)),2)
        x = torch.cat((x[:,:,:,-1].unsqueeze(3),x,x[:,:,:,0].unsqueeze(3)),3)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = torch.cat((x[:,:,-1,:].unsqueeze(2),x,x[:,:,0,:].unsqueeze(2)),2)
        x = torch.cat((x[:,:,:,-1].unsqueeze(3),x,x[:,:,:,0].unsqueeze(3)),3)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        return x

#Up-sampling + batch normalization block
class Up_Bn2D(nn.Module):
    def __init__(self, n_ch):
        super(Up_Bn2D, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.bn = nn.BatchNorm2d(n_ch)

    def forward(self, x):
        x = self.bn(self.up(x))
        return x

class Pyramid2D(nn.Module):
    def __init__(self, ch_in=3, ch_step=8):
        super(Pyramid2D, self).__init__()

        self.cb1_1 = Conv_block2D(ch_in,ch_step)
        self.up1 = Up_Bn2D(ch_step)

        self.cb2_1 = Conv_block2D(ch_in,ch_step)
        self.cb2_2 = Conv_block2D(2*ch_step,2*ch_step)
        self.up2 = Up_Bn2D(2*ch_step)

        self.cb3_1 = Conv_block2D(ch_in,ch_step)
        self.cb3_2 = Conv_block2D(3*ch_step,3*ch_step)
        self.up3 = Up_Bn2D(3*ch_step)

        self.cb4_1 = Conv_block2D(ch_in,ch_step)
        self.cb4_2 = Conv_block2D(4*ch_step,4*ch_step)
        self.up4 = Up_Bn2D(4*ch_step)

        self.cb5_1 = Conv_block2D(ch_in,ch_step)
        self.cb5_2 = Conv_block2D(5*ch_step,5*ch_step)
        self.up5 = Up_Bn2D(5*ch_step)

        self.cb6_1 = Conv_block2D(ch_in,ch_step)
        self.cb6_2 = Conv_block2D(6*ch_step,6*ch_step)
        self.last_conv = nn.Conv2d(6*ch_step, 3, 1, padding=0, bias=True)

    def forward(self, z):

        y = self.cb1_1(z[5])
        y = self.up1(y)
        y = torch.cat((y,self.cb2_1(z[4])),1)
        y = self.cb2_2(y)
        y = self.up2(y)
        y = torch.cat((y,self.cb3_1(z[3])),1)
        y = self.cb3_2(y)
        y = self.up3(y)
        y = torch.cat((y,self.cb4_1(z[2])),1)
        y = self.cb4_2(y)
        y = self.up4(y)
        y = torch.cat((y,self.cb5_1(z[1])),1)
        y = self.cb5_2(y)
        y = self.up5(y)
        y = torch.cat((y,self.cb6_1(z[0])),1)
        y = self.cb6_2(y)
        y = self.last_conv(y)
        return y

# post processing for images
postpa = transforms.Compose([
        transforms.Lambda(lambda x: x.mul_(1./255)),
        #add imagenet mean
        transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                            std=[1,1,1]),
        #turn to RGB
        transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
        ])

postpb = transforms.Compose([transforms.ToPILImage()])
def postp(tensor): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1
    t[t<0] = 0
    img = postpb(t)
    return img


model_folder = 'Trained_models/2019-01-22_red-peppers256_2D_1347'

#load model
generator = Pyramid2D(ch_step=8)
generator.load_state_dict(torch.load('./' + model_folder + '/params.pytorch'))
generator.cuda()
generator.eval()
for param in generator.parameters():
    param.requires_grad = False


#draw sample
n_input_ch = 3
sample_size = 1024
n_samples = 2

sz = [sample_size /1,sample_size /2,sample_size /4,sample_size /8,sample_size /16,sample_size /32]
zk = [torch.rand(n_samples,n_input_ch,int(szk),int(szk)) for szk in sz]
z_samples = [Variable(z.cuda()) for z in zk ]
sample = generator(z_samples)

for n in range(n_samples):
    single_sample = sample[n,:,:,:]
    out_img = postp(single_sample.data.cpu().squeeze())
    out_img.save('./' + model_folder + '/offline_sample_' + str(n) + '.jpg', "JPEG")
