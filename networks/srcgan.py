import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
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
        x = F.leaky_relu(x, 0.2)
        return x

class UpConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(UpConvBNLayer, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels=in_channels,
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
    def __init__(self, in_channels=1, res=False):
        super(Gen, self).__init__()
        self.resnet = res
        self.conv1 = nn.Conv3d(in_channels, 64, 4, 2, 1)
        self.convbn2 = ConvBNLayer(64, 128, 4, 2)
        self.convbn3 = ConvBNLayer(128, 256, 4, 2)
        self.convbn4 = ConvBNLayer(256, 512, 4, 2)
        self.convbn5 = ConvBNLayer(512, 512, 4, 2)
        # self.convbn6 = ConvBNLayer(512, 512, 4, 2)
        # self.convbn7 = ConvBNLayer(512, 512, 4, 2)

        # self.upconvbn1 = UpConvBNLayer(512, 512, 4, 2)
        # self.upconvbn2 = UpConvBNLayer(512, 512, 4, 2)
        self.upconvbn3 = UpConvBNLayer(512, 512, 4, 2)
        self.upconvbn4 = UpConvBNLayer(1024, 512, 4, 2)
        self.upconvbn5 = UpConvBNLayer(256*3, 256, 4, 2)
        self.upconvbn6 = UpConvBNLayer(128*3, 128, 4, 2)
        self.upconvbn7 = UpConvBNLayer(64*3, 128, 4, 2)
        self.conv2 = nn.Conv3d(128, 1, 1, 1)

    def forward(self, input):
        x = self.conv1(input)
        x1 = F.leaky_relu(x, 0.2)
        x2 = self.convbn2(x1)
        # print(x2.shape)
        x3 = self.convbn3(x2)
        x4 = self.convbn4(x3)
        x = self.convbn5(x4)
        # print(x5.shape)
        # x = self.convbn6(x5)
        # print(x6.shape)
        # x = self.convbn7(x6)
        # print(x.shape)
        # x = self.upconvbn1(x)
        
        # x = torch.cat((x, x6), dim=1)
        # print(x.shape)
        # x = self.upconvbn2(x)
        # x = torch.cat((x, x5), dim=1)
        x = self.upconvbn3(x)
        x = torch.cat((x, x4), dim=1)
        x = self.upconvbn4(x)
        x = torch.cat((x, x3), dim=1)
        x = self.upconvbn5(x)
        x = torch.cat((x, x2), dim=1)
        x = self.upconvbn6(x)
        x = torch.cat((x, x1), dim=1)
        x = self.upconvbn7(x)
        x = self.conv2(x)
        if self.resnet:
            x = torch.add(x, input)
        return x

class Dsc(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, 64, 4, 2, 1)
        self.convbn1 = ConvBNLayer(64, 128, 4, 2)
        self.convbn2 = ConvBNLayer(128, 256, 4, 2)
        self.convbn3 = ConvBNLayer(256, 512, 4, 2)
        self.conv2 = nn.Conv3d(512, 1, 1, 1)
        # self.linear = nn.Linear(512, 1)
    
    def forward(self, input):
        x = self.conv(input)
        x = F.leaky_relu(x, 0.2)
        x = self.convbn1(x)
        x = self.convbn2(x)
        x = self.convbn3(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return x

if __name__ == '__main__':
    network = Gen(1, True)

    img = torch.ones([1, 1, 32, 32, 32])
    # img = img.cuda()
    # network = network.cuda()
    outs = network(img)
    print(outs.shape)
