import torch


def choose_model(model_name):
    if model_name == 'Unet':
        from models.Unet import Unet
        model = Unet(in_ch=3, out_ch=3)
    elif model_name == 'srlut':
        from models.srlut import SRNet
        model = SRNet(in_channels=3, n_features=64)
    elif model_name == 'UnetTiny':
        from models.UnetTiny import UnetTiny
        model = UnetTiny(3, 3)
    elif model_name == 'UnetTinyRF':
        from models.UnetTinyRF import UnetTinyRF
        model = UnetTinyRF(3, 3)
    else:
        model = None
        print('unknown model name.')
    return model

def choose_loss(loss_name):
    if loss_name == 'l1':
        criterion = torch.nn.L1Loss()
    elif loss_name == 'l2':
        criterion = torch.nn.MSELoss()
    elif loss_name == 'hist':
        from losses import HistogramLoss
        criterion = HistogramLoss(200)
    else:
        criterion = None
        print('unknown loss name.')
    return criterion