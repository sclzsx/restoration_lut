import torch
import torch.nn as nn
import torch.nn.functional as F


class UnetTiny(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(UnetTiny, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, 64, 1, padding=0, stride=1, bias=True)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=0, stride=1, bias=True)
        self.relu2 = nn.PReLU(num_parameters=32)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=0, stride=1, bias=True)
        self.relu3 = nn.PReLU(num_parameters=32)
        #self.conv2 = nn.Conv2d(64, 64, 3, padding=0, stride=1, bias=True)
        #self.relu2 = nn.PReLU(num_parameters=64)
        #self.conv3 = nn.Conv2d(64, 64, 3, padding=0, stride=1, bias=True)
        #self.relu3 = nn.PReLU(num_parameters=64)

        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=0)
        #self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0)
        #self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv_out = nn.Conv2d(64, out_ch, kernel_size=1, stride=1, padding=0)

        self.print_network()

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Total number of parameters: %d' % num_params)

    def forward(self, x):
        # print(x.shape)

        conv_out_1 = self.conv1(x)
        # print(conv_out_1.shape)

        conv_out_2 = self.relu2(self.conv2(conv_out_1))
        # print(conv_out_2.shape)

        conv_out_3 = self.relu3(self.conv3(conv_out_2))
        # print(conv_out_3.shape)

        deconv_out_1 = self.deconv1(conv_out_3)
        # print(deconv_out_1.shape)

        deconv_out_2 = self.deconv2(deconv_out_1 + conv_out_2)
        # print(deconv_out_2.shape)

        out = self.conv_out(deconv_out_2 + conv_out_1)
        # print(out.shape)

        return out

if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    import time

    with torch.no_grad():
        net = UnetTiny().cuda()

        f, p = get_model_complexity_info(net, (3, 256, 256), as_strings=True, print_per_layer_stat=True, verbose=True)
        print('FLOPs:', f, 'Parms:', p)

        x = torch.randn(1, 3, 256, 256).cuda()
        print('in', x.shape)

        s = time.time()
        x = net(x)
        print('out', x.shape, 1 / (time.time() - s))

# FLOPs: 3.62 GMac Parms: 55.97 k
# in torch.Size([1, 3, 256, 256])
# out torch.Size([1, 3, 256, 256]) 2106.631843294827
