import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        # print("Mish activation loaded...")

    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class SRNet(nn.Module):
    def __init__(self, in_channels, n_features):
        super(SRNet, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=n_features, kernel_size=[2, 2], padding=0, bias=True, dilation=1),
            nn.BatchNorm2d(n_features), 
            Mish(),

            nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=1, padding=0, bias=True, dilation=1),
            nn.BatchNorm2d(n_features), 
            Mish(),

            nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=1, padding=0, bias=True, dilation=1),
            nn.BatchNorm2d(n_features), 
            Mish(),

            nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=1, padding=0, bias=True, dilation=1),
            nn.BatchNorm2d(n_features), 
            Mish(),

            nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=1, padding=0, bias=True, dilation=1),
            nn.BatchNorm2d(n_features), 
            Mish(),

            nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=1, padding=0, bias=True, dilation=1),
            nn.BatchNorm2d(n_features), 
            Mish(),

            nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=1, padding=0, bias=True, dilation=1),
            nn.BatchNorm2d(n_features), 
            Mish(),

            nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=1, padding=0, bias=True, dilation=1),
            nn.BatchNorm2d(n_features), 
            Mish(),

            nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=1, padding=0, bias=True, dilation=1),
            nn.BatchNorm2d(n_features), 
            Mish(),

            nn.Conv2d(in_channels=n_features, out_channels=in_channels, kernel_size=1, padding=0, bias=True, dilation=1),
        )

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x_in):
        x = self.backbone(x_in)
        return x