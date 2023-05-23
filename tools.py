import random
import torch
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])

def batch_SSIM(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    for i in range(Img.shape[0]):
        for j in range(Img.shape[1]):
            SSIM += compare_ssim(Iclean[i, j, :, :], Img[i, j, :, :], data_range=data_range)
    return (SSIM / Img.shape[0] / Img.shape[1])

def self_ensemble_rot4(source, model):
    source1 = F.pad(source, (0,1,0,1), mode='reflect')
    source2 = F.pad(source, (0,1,0,1), mode='reflect')
    source3 = F.pad(source, (0,1,0,1), mode='reflect')
    source4 = F.pad(source, (0,1,0,1), mode='reflect')

    source2 = torch.rot90(source2, 1, [2,3])
    source3 = torch.rot90(source3, 2, [2,3])
    source4 = torch.rot90(source4, 3, [2,3])

    source1 = model(source1)
    source2 = model(source2)
    source3 = model(source3)
    source4 = model(source4)

    source2 = torch.rot90(source2, 3, [2,3])
    source3 = torch.rot90(source3, 2, [2,3])
    source4 = torch.rot90(source4, 1, [2,3])

    source_out = (source1 + source2 + source3 + source4) / 4
    return source_out