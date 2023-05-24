from torch.utils.data import DataLoader
import torch
import os
import shutil
import numpy as np
import cv2
import sys
import time
from pathlib import Path

from tools import self_ensemble_rot4, batch_PSNR, batch_SSIM
from dataset import REC_DATASET, extract_path_pairs
from choices import choose_model

########################################################### hyperparameters 


model_name = 'UnetTinyRF'
ckpt_path = 'results/TextDeblur/UnetTinyRF/max_test_psnr.pt'

patch_size_test = 192
patch_num_per_img_test = 1
test_image_num = None
source_paths_test_txt = '/data/datasets/TEXT_DEBLUR/blur_test.txt'
target_paths_test_txt = '/data/datasets/TEXT_DEBLUR/gt_test.txt'
fix_img_size_test = (200, 200)
extract_random_patch_test = False
augment = False
vis_test_iter_freq = 0.1
num_workers_test = 0

batch_size_test = 1  # 大于1可能测得不准bug
self_ensemble_test = False

########################################################### prepare 

save_dir = str(Path(ckpt_path).parent) + '/eval'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model = choose_model(model_name)

source_paths_test, target_paths_test = extract_path_pairs(source_paths_test_txt, target_paths_test_txt, shuffle=True)
print('dataset test images num:', len(source_paths_test))

if test_image_num is not None and test_image_num < len(source_paths_test):
    source_paths_test = source_paths_test[:test_image_num]
    target_paths_test = target_paths_test[:test_image_num]
print('selected test images num:', len(source_paths_test))

dataset_test = REC_DATASET(source_paths_test, target_paths_test, patch_size_test, patch_num_per_img_test,
                           fix_img_size_test, extract_random_patch_test, augment)
print('test patches num:', len(dataset_test))

loader_test = DataLoader(dataset=dataset_test, num_workers=num_workers_test, batch_size=batch_size_test, shuffle=False)

ep_test_iter_num = len(loader_test)
vis_test_iter_num = max(int(vis_test_iter_freq * ep_test_iter_num), 1)

if ckpt_path is not None:
    if not os.path.exists(ckpt_path):
        print('resume path not exist.')
        sys.exit()
    print('load', ckpt_path)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])

model.cuda()

save_img_dir = save_dir + '/save_vis'
if os.path.exists(save_img_dir):
    shutil.rmtree(save_img_dir)
os.makedirs(save_img_dir)

model.eval()

test_psnr = 0
test_ssim = 0

for i, (source, target) in enumerate(loader_test):
    with torch.no_grad():
        source = source.cuda()
        target = target.cuda()

        if self_ensemble_test:
            source_out = self_ensemble_rot4(source, model)
        else:
            source_out = model(source)

    source_out = torch.clamp(source_out, 0, 1)

    psnr_ori = batch_PSNR(source, target, 1.0)
    psnr_out = batch_PSNR(source_out, target, 1.0)
    print('psnr_ori:{}, psnr_out:{}'.format(psnr_ori, psnr_out))
    test_psnr += psnr_out

    ssim_ori = batch_SSIM(source, target, 1.0)
    ssim_out = batch_SSIM(source_out, target, 1.0)
    print('ssim_ori:{}, ssim_out:{}'.format(ssim_ori, ssim_out))
    test_ssim += ssim_out

    if (i + 1) % vis_test_iter_num == 0:
        source_ori = np.array(source[0, :, :, :].permute(1, 2, 0).cpu())
        source_out = np.array(source_out[0, :, :, :].permute(1, 2, 0).cpu())
        target = np.array(target[0, :, :, :].permute(1, 2, 0).cpu())

        source_ori = (source_ori * 255).astype('uint8')
        source_out = (source_out * 255).astype('uint8')
        target = (target * 255).astype('uint8')

        cv2.imwrite(save_img_dir + '/' + str(i) + '_source_' + str(psnr_ori) + '.jpg', source_ori)
        cv2.imwrite(save_img_dir + '/' + str(i) + '_restored_' + str(psnr_out) + '.jpg', source_out)
        cv2.imwrite(save_img_dir + '/' + str(i) + '_target.jpg', target)

test_psnr = test_psnr / ep_test_iter_num
test_ssim = test_ssim / ep_test_iter_num

print('-' * 50)
print('PSNR:', test_psnr)
print('SSIM:', test_ssim)
print('-' * 50)

with open(save_dir + '/eval_metrics.txt', 'w') as f:
    f.write('PSNR: {}\n'.format(test_psnr))
    f.write('SSIM: {}\n'.format(test_ssim))
