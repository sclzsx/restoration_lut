import torch


def choose_model(model_name):
    if model_name == 'unet':
        from models.unet import UNet
        model = UNet(in_ch=3, out_ch=3)
    elif model_name == 'srlut':
        from models.srlut import SRNet
        model = SRNet(in_channels=3, n_features=64)
    else:
        model = None
        print('unknown model name.')
    return model

def choose_loss(loss_name):
    if loss_name == 'l1':
        criterion = torch.nn.L1Loss()
    elif loss_name == 'l2':
        criterion = torch.nn.MSELoss()
    else:
        criterion = None
        print('unknown loss name.')
    return criterion