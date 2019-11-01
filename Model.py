import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch


class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Conv2d(64, 1, 1)
        vgg16 = models.vgg16(pretrained=True)
        self._initialize_weights()
        for i in range(len(self.frontend.state_dict().items())):
            list(self.frontend.state_dict().items())[i][1].data[:] = list(vgg16.state_dict().items())[i][1].data[:]

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)  # 是否应该加入relu以限制输出结果在范围[0,1]？
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, 2, stride=2),

            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 2, stride=2),

            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 2, stride=2),

            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(64, 1, 1)
        vgg16 = models.vgg16(pretrained=True)
        self._initialize_weights()
        for i in range(len(self.frontend.state_dict().items())):
            list(self.frontend.state_dict().items())[i][1].data[:] = list(vgg16.state_dict().items())[i][1].data[:]

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)  # 是否应该加入relu以限制输出结果在范围[0,1]？
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Net2(nn.Module):  # padding = int((kernel_size - 1) / 2) if same_padding else 0
    def __init__(self):
        super(Net2, self).__init__()
        self.frontend = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=64, is_first_layer=True),
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=256),
            ConvBlock(in_channels=256, out_channels=512),
        )
        self.backend = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Sequential(
            nn.Conv2d(64, 1, 1), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, instance_norm=False, is_first_layer=False):
        super(ConvBlock, self).__init__()
        brand_in_chans = in_channels
        brand_out_chans = int(out_channels / 4)
        if is_first_layer:  # 第一个ConvBlock未经过1x1降维
            self.brand1 = nn.Sequential(
                nn.Conv2d(brand_in_chans, brand_out_chans, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True)
            )
            self.brand2 = nn.Sequential(
                nn.Conv2d(brand_in_chans, brand_out_chans, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )
            self.brand3 = nn.Sequential(
                nn.Conv2d(brand_in_chans, brand_out_chans, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True)
            )
            self.brand4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(brand_in_chans, brand_out_chans, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True)
            )
        else:
            inner_channels = int(brand_out_chans / 2)  # to reduce the feature dim by half
            self.brand1 = nn.Conv2d(brand_in_chans, brand_out_chans, kernel_size=1, stride=1)
            self.brand2 = nn.Sequential(
                nn.Conv2d(brand_in_chans, inner_channels, kernel_size=1, stride=1), nn.ReLU(inplace=True),
                nn.Conv2d(inner_channels, brand_out_chans, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)
            )
            self.brand3 = nn.Sequential(
                nn.Conv2d(brand_in_chans, inner_channels, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(inner_channels, brand_out_chans, kernel_size=5, stride=1, padding=2), nn.ReLU(inplace=True)
            )
            self.brand4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(brand_in_chans, brand_out_chans, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x1 = self.brand1(x)
        x2 = self.brand2(x)
        x3 = self.brand3(x)
        x4 = self.brand4(x)
        x = torch.cat((x1, x2, x3, x4), 1)
        return x


# test
model = Net2()
# print(model)
x = torch.rand((1, 3, 240, 240))
print(model(x).shape)  # torch.Size([1, 1, 96, 128])
