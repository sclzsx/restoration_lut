import torch.nn as nn
import torch.nn.functional as F

class SRNet(nn.Module):
    def __init__(self, in_channels, n_features):
        super(SRNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_features, [2, 2], stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(n_features, n_features, 1, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(n_features, n_features, 1, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(n_features, n_features, 1, stride=1, padding=0, dilation=1)
        self.conv5 = nn.Conv2d(n_features, n_features, 1, stride=1, padding=0, dilation=1)
        self.conv6 = nn.Conv2d(n_features, in_channels, 1, stride=1, padding=0, dilation=1)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv4(F.relu(x))
        x = self.conv5(F.relu(x))
        x = self.conv6(F.relu(x))
        return x