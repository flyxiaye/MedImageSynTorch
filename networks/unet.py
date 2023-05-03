import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.activation import ReLU


class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = F.relu(x)
        return x


class Gen(nn.Module):
    def __init__(self, in_channels=1):
        super(Gen, self).__init__()
        self.convbn11 = ConvBNLayer(in_channels, 32)
        self.convbn12 = ConvBNLayer(32, 64)
        self.pool1 = nn.MaxPool3d(2, 2)

        self.convbn21 = ConvBNLayer(64, 64)
        self.convbn22 = ConvBNLayer(64, 128)
        self.pool2 = nn.MaxPool3d(2, 2)

        self.convbn31 = ConvBNLayer(128, 128)
        self.convbn32 = ConvBNLayer(128, 256)
        self.pool3 = nn.MaxPool3d(2, 2)

        self.convbn41 = ConvBNLayer(256, 256)
        self.convbn42 = ConvBNLayer(256, 512)

        self.upconv1 = nn.ConvTranspose3d(512, 512, 2, 2)

        self.convbn51 = ConvBNLayer(512 + 256, 256)
        self.convbn52 = ConvBNLayer(256, 256)
        
        self.upconv2 = nn.ConvTranspose3d(256, 256, 2, 2)

        self.convbn61 = ConvBNLayer(256 + 128, 128)
        self.convbn62 = ConvBNLayer(128, 128)

        self.upconv3 = nn.ConvTranspose3d(128, 128, 2, 2)
        
        self.convbn71 = ConvBNLayer(128 + 64, 64)
        self.convbn72 = ConvBNLayer(64, 64)

        self.conv = nn.Conv3d(64, 1, 1, 1)

    def forward(self, input):
        x = self.convbn11(input)
        x1 = self.convbn12(x)
        x = self.pool1(x1)
        x = self.convbn21(x)
        x2 = self.convbn22(x)
        x = self.pool2(x2)
        x = self.convbn31(x)
        x3 = self.convbn32(x)
        x = self.pool3(x3)
        x = self.convbn41(x)
        x = self.convbn42(x)
        x = self.upconv1(x)
        yshape, xshape = np.array(x3.shape[2:]), np.array(x.shape[2:])
        starts = (yshape - xshape) // 2
        x3 = x3[:, :, starts[0]:starts[0]+xshape[0], starts[1]:starts[1]+xshape[1], starts[2]:starts[2]+xshape[2]]
        x = torch.cat((x, x3), dim=1)
        x = self.convbn51(x)
        x = self.convbn52(x)
        x = self.upconv2(x)
        yshape, xshape = np.array(x2.shape[2:]), np.array(x.shape[2:])
        starts = (yshape - xshape) // 2
        x2 = x2[:, :, starts[0]:starts[0]+xshape[0], starts[1]:starts[1]+xshape[1], starts[2]:starts[2]+xshape[2]]
        x = torch.cat((x, x2), dim=1)
        x = self.convbn61(x)
        x = self.convbn62(x)
        x = self.upconv3(x)
        yshape, xshape = np.array(x1.shape[2:]), np.array(x.shape[2:])
        starts = (yshape - xshape) // 2
        x1 = x1[:, :, starts[0]:starts[0]+xshape[0], starts[1]:starts[1]+xshape[1], starts[2]:starts[2]+xshape[2]]
        x = torch.cat((x, x1), dim=1)
        x = self.convbn71(x)
        x = self.convbn72(x)
        x = self.conv(x)
        return x

class Dsc(nn.Module):
    def __init__(self, in_channels):
        super(Dsc, self).__init__()
        self.convbn1 = ConvBNLayer(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.convbn2 = ConvBNLayer(32, 64)
        self.pool2 = nn.MaxPool3d(2, 2)
        # self.convbn3 = ConvBNLayer(64, 128)
        # self.pool3 = nn.MaxPool3d(2, 2)
        self.conv4 = nn.Conv3d(64, 128, 3, 1)
        self.fc1 = nn.Linear(16000, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, input):
        x = self.convbn1(input)
        x = self.pool1(x)
        x = self.convbn2(x)
        x = self.pool2(x)
        # x = self.convbn3(x)
        # x = self.pool3(x)
        x = self.conv4(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

if __name__ == '__main__':
    network = Gen(1)

    img = torch.ones([1, 1, 128, 128, 128])
    # img = img.cuda()
    # network = network.cuda()
    outs = network(img)
    print(outs.shape)
    """
    1,1,128,128,128 -> 1,1,36,36,36
    """
        

