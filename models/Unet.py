import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        
        #n1 = 8    #not sufficient to simulate BM3D
        n1 = 16
        #n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Avgpool  = nn.AvgPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_1 = conv_block(in_ch, filters[0])
        self.Conv1_2 = conv_block(filters[0], filters[0])
        self.Conv2_1 = conv_block(filters[0], filters[1])
        self.Conv2_2 = conv_block(filters[1], filters[1])
        self.Conv3_1 = conv_block(filters[1], filters[2])
        self.Conv3_2 = conv_block(filters[2], filters[2])
        self.Conv4_1 = conv_block(filters[2], filters[3])
        self.Conv4_2 = conv_block(filters[3], filters[3])
        self.Conv5_1 = conv_block(filters[3], filters[4])
        self.Conv5_2 = conv_block(filters[4], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5_1 = conv_block(filters[4], filters[3])
        self.Up_conv5_2 = conv_block(filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4_1 = conv_block(filters[3], filters[2])
        self.Up_conv4_2 = conv_block(filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3_1 = conv_block(filters[2], filters[1])
        self.Up_conv3_2 = conv_block(filters[1], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2_1 = conv_block(filters[1], filters[0])
        self.Up_conv2_2 = conv_block(filters[0], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    
    def forward(self, x):
        e1 = self.Conv1_1(x)
        e1 = self.Conv1_2(e1)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2_1(e2)
        e2 = self.Conv2_2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3_1(e3)
        e3 = self.Conv3_2(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4_1(e4)
        e4 = self.Conv4_2(e4)
        
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5_1(e5)
        e5 = self.Conv5_2(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5_1(d5)
        d5 = self.Up_conv5_2(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4_1(d4)
        d4 = self.Up_conv4_2(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3_1(d3)
        d3 = self.Up_conv3_2(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2_1(d2)
        d2 = self.Up_conv2_2(d2)

        out = self.Conv(d2)

        return out
    

if __name__ == '__main__':
    with torch.no_grad():
        x = torch.randn(2, 3, 64, 64).cuda()
        net = Unet(3, 3).cuda()
        x = net(x)
    print(x.shape)