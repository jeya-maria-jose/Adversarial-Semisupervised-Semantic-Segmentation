import torch.nn as nn
import math
import torch
import numpy as np
import torch.nn.functional as F
affine_par = True

import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
       

       	if bottleneck:
            nDenseBlocks //= 2
        print nDenseBlocks
        
        nChannels = 2*growthRate
        #print nDenseBlocks
        
        #INIT
        self.conv1a = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1,
                               bias=False)
        self.conv1b = nn.Conv2d(32 , 32 , kernel_size=3, padding=1,stride=1,
                               bias=False)
        self.conv1c = nn.Conv2d(32 , 32, kernel_size=3, padding=1, stride=1,
                               bias=False)
        self.bn1a = nn.BatchNorm2d(32)
        self.bn1b = nn.BatchNorm2d(32)
        
        #Trans1

        self.bn1c = nn.BatchNorm2d(32)
        self.convdown1 = nn.Conv2d(32, 32 , kernel_size =2 , stride =2 , padding = 0)

        #DB1 starts



        self.bn1 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32,64,kernel_size=1,stride=1,padding=0)

        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1)
        self.drop1 = nn.Dropout(p=0.2)

        #cat1

        self.bn3 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48,64,kernel_size=1,stride=1,padding=0)

        self.bn4 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1)
        self.drop1 = nn.Dropout(p=0.2)#cat2
        #cat2
        self.bn5 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64,64,kernel_size=1,stride=1,padding=0)

        self.bn6 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1)
        self.drop1 = nn.Dropout(p=0.2)
        #cat3
        self.bn7 = nn.BatchNorm2d(80)
        self.conv7 = nn.Conv2d(80,64,kernel_size=1,stride=1,padding=0)

        self.bn8 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1)
        self.drop1 = nn.Dropout(p=0.2)

        #cat4
        #trans2
        self.bn9 = nn.BatchNorm2d(96)
        self.conv9 = nn.Conv2d(96, 48 , kernel_size=1,stride=1,padding=0)


        self.bn10 = nn.BatchNorm2d(48)
        self.convdown2 = nn.Conv2d(48,48, kernel_size=2,stride=2,padding=0)

        #DB2 starts

        self.bn10 = nn.BatchNorm2d(48)
        self.conv10 = nn.Conv2d(48,64,kernel_size=1,stride=1,padding=0)

        self.bn11 = nn.BatchNorm2d(64)
        self.conv11 = nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1)
        self.drop1 = nn.Dropout(p=0.2)

        #cat1

        self.bn12 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64,64,kernel_size=1,stride=1,padding=0)

        self.bn13 = nn.BatchNorm2d(64)
        self.conv13 = nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1)
        self.drop1 = nn.Dropout(p=0.2)#cat2
        #cat2
        self.bn14 = nn.BatchNorm2d(80)
        self.conv14 = nn.Conv2d(80,64,kernel_size=1,stride=1,padding=0)

        self.bn15 = nn.BatchNorm2d(64)
        self.conv15 = nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1)
        self.drop1 = nn.Dropout(p=0.2)
        #cat3
        self.bn16 = nn.BatchNorm2d(96)
        self.conv16 = nn.Conv2d(96,64,kernel_size=1,stride=1,padding=0)

        self.bn17 = nn.BatchNorm2d(64)
        self.conv17 = nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1)
        self.drop1 = nn.Dropout(p=0.2)
        print "hello"

        #cat4
        #Trans3
        self.bn19 = nn.BatchNorm2d(112)
        self.conv19 = nn.Conv2d(112,56,kernel_size=1,stride=1,padding=0)
        self.bn20 = nn.BatchNorm2d(56)
        self.convdown3 = nn.Conv2d(56,56,kernel_size=2,stride=2,padding=0)

        #DB3 begins

        self.bn21 = nn.BatchNorm2d(56)
        self.conv21 = nn.Conv2d(56,64,kernel_size=1,stride=1,padding=0)

        self.bn22 = nn.BatchNorm2d(64)
        self.conv22 = nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1)
        self.drop1 = nn.Dropout(p=0.2)

        #cat1

        self.bn23 = nn.BatchNorm2d(72)
        self.conv23 = nn.Conv2d(72,64,kernel_size=1,stride=1,padding=0)

        self.bn24 = nn.BatchNorm2d(64)
        self.conv24 = nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1)
        self.drop1 = nn.Dropout(p=0.2)#cat2
        #cat2
        self.bn25 = nn.BatchNorm2d(88)
        self.conv25 = nn.Conv2d(88,64,kernel_size=1,stride=1,padding=0)

        self.bn26 = nn.BatchNorm2d(64)
        self.conv26 = nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1)
        self.drop1 = nn.Dropout(p=0.2)
        #cat3
        self.bn27 = nn.BatchNorm2d(104)
        self.conv27 = nn.Conv2d(104,64,kernel_size=1,stride=1,padding=0)

        self.bn28 = nn.BatchNorm2d(64)
        self.conv28 = nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1)
        self.drop1 = nn.Dropout(p=0.2)

        #cat4

        #Trans 4 

        self.bn29 = nn.BatchNorm2d(120)
        self.conv29 = nn.Conv2d(120,60,kernel_size=1,stride=1,padding=0)

        self.bn30 = nn.BatchNorm2d(60)
        self.convdown4 = nn.Conv2d(60,60,kernel_size=2,stride=2,padding=0)
        #DB4 starts here
        
        self.bn31 = nn.BatchNorm2d(60)
        self.conv31 = nn.Conv2d(60,64,kernel_size=1,stride=1,padding=0)

        self.bn32 = nn.BatchNorm2d(64)
        self.conv32 = nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1)
        self.drop1 = nn.Dropout(p=0.2)

        #cat1

        self.bn33 = nn.BatchNorm2d(76)
        self.conv33 = nn.Conv2d(76,64,kernel_size=1,stride=1,padding=0)

        self.bn34 = nn.BatchNorm2d(64)
        self.conv34 = nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1)
        self.drop1 = nn.Dropout(p=0.2)#cat2
        #cat2
        self.bn35 = nn.BatchNorm2d(92)
        self.conv35 = nn.Conv2d(92,64,kernel_size=1,stride=1,padding=0)

        self.bn36 = nn.BatchNorm2d(64)
        self.conv36 = nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1)
        self.drop1 = nn.Dropout(p=0.2)
        #cat3
        self.bn37 = nn.BatchNorm2d(108)
        self.conv37 = nn.Conv2d(108,64,kernel_size=1,stride=1,padding=0)

        self.bn38 = nn.BatchNorm2d(64)
        self.conv38 = nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1)
        self.drop1 = nn.Dropout(p=0.2)

        #cat4


        self.deconv1 = nn.ConvTranspose2d(96,4,kernel_size=3,stride=2, padding=0)
        self.deconv2 = nn.ConvTranspose2d(112,4,kernel_size=5, stride= 4,padding=0)
        self.deconv3 = nn.ConvTranspose2d(120,4,kernel_size=9, stride= 8,padding=0)
        self.deconv4 = nn.ConvTranspose2d(124,4,kernel_size=17, stride= 16, padding=0)

        self.bncon = nn.BatchNorm2d(48)
        self.conv39 = nn.Conv2d(48,2,kernel_size=1,padding=0)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def db(self):
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32,64,kernel_size=1,stride=1,padding=0)

        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1)
        self.drop1 = nn.Dropout(p=0.2)


    def forward(self, x):
    	#layers=[]
        out = self.conv1a(x)
        out = F.relu(self.bn1a(out))
        out = self.conv1b(out)
        out = F.relu(self.bn1b(out))
        out = self.conv1c(out)

        j1 = out
        #DB1 
        out = F.relu(self.bn1c(out))
        out  = self.convdown1(out)

        n1 = out
        
        out = F.relu(self.bn1(out))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        
        out = self.drop1(out)
        out = torch.cat((n1,out),1)

        
        n2 = out
        
        out = F.relu(self.bn3(out))
        out = self.conv3(out)
        out = F.relu(self.bn4(out))
        out = self.conv4(out)
        
        out = self.drop1(out)
        out = torch.cat((n2,out),1)

        n3 = out
        
        out = F.relu(self.bn5(out))
        out = self.conv5(out)
        out = F.relu(self.bn6(out))
        out = self.conv6(out)
        
        out = self.drop1(out)
        out = torch.cat((n3,out),1)

        n4 = out
        
        out = F.relu(self.bn7(out))
        out = self.conv7(out)
        out = F.relu(self.bn8(out))
        out = self.conv8(out)
        
        out = self.drop1(out)
        out = torch.cat((n4,out),1)

        j2 = out

        out = F.relu(self.bn9(out))
        out = self.conv9(out)
        out = F.relu(self.bn10(out))

        out = self.convdown2(out)

        n1 = out

        out = F.relu(self.bn10(out))
        out = self.conv10(out)
        out = F.relu(self.bn11(out))
        out = self.conv11(out)
        
        out = self.drop1(out)
        out = torch.cat((n1,out),1)

        
        n2 = out
        
        out = F.relu(self.bn12(out))
        out = self.conv12(out)
        out = F.relu(self.bn13(out))
        out = self.conv13(out)
        
        out = self.drop1(out)
        out = torch.cat((n2,out),1)

        n3 = out
        
        out = F.relu(self.bn14(out))
        out = self.conv14(out)
        out = F.relu(self.bn15(out))
        out = self.conv15(out)
        
        out = self.drop1(out)
        out = torch.cat((n3,out),1)

        n4 = out
        
        out = F.relu(self.bn16(out))
        out = self.conv16(out)
        out = F.relu(self.bn17(out))
        out = self.conv17(out)
        
        out = self.drop1(out)
        out = torch.cat((n4,out),1)

        j3 = out

        out = F.relu(self.bn19(out))
        out= self.conv19(out)
        out = F.relu(self.bn20(out))
        out = self.convdown3(out)

        n1= out


        out = F.relu(self.bn21(out))
        out = self.conv21(out)
        out = F.relu(self.bn22(out))
        out = self.conv22(out)
        
        out = self.drop1(out)
        out = torch.cat((n1,out),1)

        
        n2 = out
        
        out = F.relu(self.bn23(out))
        out = self.conv23(out)
        out = F.relu(self.bn24(out))
        out = self.conv24(out)
        
        out = self.drop1(out)
        out = torch.cat((n2,out),1)

        n3 = out
        
        out = F.relu(self.bn25(out))
        out = self.conv25(out)
        out = F.relu(self.bn26(out))
        out = self.conv26(out)
        
        out = self.drop1(out)
        out = torch.cat((n3,out),1)

        n4 = out
        
        out = F.relu(self.bn27(out))
        out = self.conv27(out)
        out = F.relu(self.bn28(out))
        out = self.conv28(out)
        
        out = self.drop1(out)
        out = torch.cat((n4,out),1)

        j4 = out

        out = F.relu(self.bn29(out))
        out= self.conv29(out)
        out = F.relu(self.bn30(out))
        out = self.convdown4(out)

        n1 = out

        out = F.relu(self.bn31(out))
        out = self.conv31(out)
        out = F.relu(self.bn32(out))
        out = self.conv32(out)
        
        out = self.drop1(out)
        out = torch.cat((n1,out),1)

        
        n2 = out
        
        out = F.relu(self.bn33(out))
        out = self.conv33(out)
        out = F.relu(self.bn34(out))
        out = self.conv34(out)
        
        out = self.drop1(out)
        out = torch.cat((n2,out),1)

        n3 = out
        
        out = F.relu(self.bn35(out))
        out = self.conv35(out)
        out = F.relu(self.bn36(out))
        out = self.conv36(out)
        
        out = self.drop1(out)
        out = torch.cat((n3,out),1)

        n4 = out
        
        out = F.relu(self.bn37(out))
        out = self.conv37(out)
        out = F.relu(self.bn38(out))
        out = self.conv38(out)
        
        out = self.drop1(out)
        out = torch.cat((n4,out),1)

        jn4 = self.deconv4(out)
        jn3 = self.deconv3(j4)
        jn2 = self.deconv2(j3)
        jn1 = self.deconv1(j2)


        out = torch.cat((j1,jn1,jn2,jn3,jn4),1)

        out = F.relu(self.bncon(out))
        out = self.conv39(out)





        return out


def Denselab(NoLabels=2):
    return DenseNet(growthRate=8, depth=50, reduction=0.5,bottleneck=True, nClasses=2)