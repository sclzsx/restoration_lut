import cv2
import random
import torch
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import DataLoader
import torch
import os
import shutil
import torch.nn.functional as F
import numpy as np
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def extract_path_pairs(source_paths_txt, target_paths_txt, shuffle=False):
    with open(source_paths_txt, 'r') as f:
        lines = f.readlines()
        source_paths = [i.strip() for i in lines]

    with open(target_paths_txt, 'r') as f:
        lines = f.readlines()
        target_paths = [i.strip() for i in lines]

    if shuffle:
        random.seed(0)
        random.shuffle(source_paths)
        random.seed(0)
        random.shuffle(target_paths)

    return source_paths, target_paths

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])

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