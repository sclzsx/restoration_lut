import torch
import torch.nn as nn
import torch.nn.functional as F


'''
感受野计算公式：
F(n) = F(n-1) + (K(n) - 1) * PI(S(1), S(2), ..., S(n - 1))
F(0) = 1

F:感受野大小
n:层号 (n >= 1)
K:卷积核或池化核大小
S:卷积或池化步长
PI:求积

当步长固定为1, K固定时:
F(n) = n * (K - 1) + 1

空洞卷积等效核尺寸公式:
K = K + (K - 1) * (D - 1)
D:空洞数

卷积输出尺寸公式：
N = (N - K + 2P) / S + 1
N:输出尺寸
P:Padding数

'''

class UnetTinyRF(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(UnetTinyRF, self).__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, dilation=1, stride=2, bias=True),
            nn.PReLU(num_parameters=32),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, dilation=1, stride=2, bias=True),
            nn.PReLU(num_parameters=32),
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, dilation=1, stride=2, bias=True),
            nn.PReLU(num_parameters=32),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0, bias=True),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0, bias=True),
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 3, 3, padding=1, dilation=1, stride=1, bias=True),
        )

        self.print_network()

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Total number of parameters: %d' % num_params)

    def forward(self, x):
        d1 = self.down1(x)
        # print('d1', d1.shape)
        d2 = self.down2(d1)
        # print('d2', d2.shape)
        d3 = self.down3(d2)
        # print('d3', d3.shape)
        u1 = self.up1(d3)
        # print('u1', u1.shape)
        u2 = self.up2(u1 + d2)
        # print('u2', u2.shape)
        u3 = self.up3(u2 + d1)
        return u3


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    import time

    with torch.no_grad():
        net = UnetTinyRF().cuda()

        f, p = get_model_complexity_info(net, (3, 256, 256), as_strings=True, print_per_layer_stat=True, verbose=True)
        print('FLOPs:', f, 'Parms:', p)

        x = torch.randn(1, 3, 256, 256).cuda()
        print('in', x.shape)

        s = time.time()
        x = net(x)
        print('out', x.shape, 1 / (time.time() - s))

# FLOPs: 173.47 MMac Parms: 28.2 k
# in torch.Size([1, 3, 256, 256])
# out torch.Size([1, 3, 256, 256]) 2027.2131464475592
