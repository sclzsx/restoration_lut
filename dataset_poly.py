from torch.utils.data import Dataset
import cv2
import random
import torch
from tqdm import tqdm
import os
import numpy as np
import sys
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image


def spatial_deconvolution(img, kernel, alpha, beta):
    a3 = alpha / 2 - beta + 2
    a2 = 3 * beta - alpha - 6
    a1 = 5 - 3 * beta + alpha / 2
    imout = a3 * img
    imout = cv2.filter2D(imout, -1, kernel) + a2 * img
    imout = cv2.filter2D(imout, -1, kernel) + a1 * img
    imout = cv2.filter2D(imout, -1, kernel) + beta * img
    return np.clip(imout, 0, 1)


class REC_DATASET(Dataset):
    def __init__(self, source_paths, target_paths, patch_size, patch_num_per_img, fix_img_size, extract_random_patch,
                 augment):
        super(REC_DATASET, self).__init__()
        patches_info = []
        image_idx = 0
        for path in tqdm(source_paths):

            if fix_img_size is None:
                img = cv2.imread(path)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                h, w, _ = img.shape
            else:
                if not os.path.exists(path):
                    print('Image is not existed.', path)
                    sys.exit()
                h, w = fix_img_size

            if not (patch_size < h and patch_size < w):
                print('patch_size is not suitable.', path, patch_size, h, w, path)
                sys.exit()

            if patch_num_per_img > 1:
                if extract_random_patch == 0:
                    for _ in range(patch_num_per_img):
                        i = np.random.randint(0, h - patch_size)
                        j = np.random.randint(0, w - patch_size)
                        patches_info.append([image_idx, i, i + patch_size, j, j + patch_size])
                else:
                    patch_stride = patch_size // 2
                    patch_cnt = 0
                    for i in range(0, h - patch_size + 1, patch_stride):
                        if patch_cnt > patch_num_per_img:
                            break
                        for j in range(0, w - patch_size + 1, patch_stride):
                            if patch_cnt > patch_num_per_img:
                                break
                            patches_info.append([image_idx, i, i + patch_size, j, j + patch_size])
                            patch_cnt += 1
            else:
                i = h // 2 - patch_size // 2
                j = w // 2 - patch_size // 2
                patches_info.append([image_idx, i, i + patch_size, j, j + patch_size])

            image_idx += 1

        # random.shuffle(patches_info)

        self.source_paths = source_paths
        self.target_paths = target_paths
        self.patches_info = patches_info
        self.augment = augment

    def __len__(self):
        return len(self.patches_info)

    def __getitem__(self, idx):
        image_idx = self.patches_info[idx][0]
        patch_cord_h0 = self.patches_info[idx][1]
        patch_cord_h1 = self.patches_info[idx][2]
        patch_cord_w0 = self.patches_info[idx][3]
        patch_cord_w1 = self.patches_info[idx][4]

        source_image = cv2.imread(self.source_paths[image_idx])
        target_image = cv2.imread(self.target_paths[image_idx])

        source_patch = source_image[patch_cord_h0:patch_cord_h1, patch_cord_w0:patch_cord_w1, :]
        target_patch = target_image[patch_cord_h0:patch_cord_h1, patch_cord_w0:patch_cord_w1, :]

        if self.augment:
            if np.random.rand() < 0.5:
                np.flipud(source_patch)
                np.flipud(target_patch)
            if np.random.rand() < 0.5:
                np.fliplr(source_patch)
                np.fliplr(target_patch)
            if np.random.rand() < 0.5:
                np.flipud(source_patch)
                np.flipud(target_patch)
            if np.random.rand() < 0.5:
                np.rot90(source_patch, 1)
                np.rot90(target_patch, 1)
            if np.random.rand() < 0.5:
                np.rot90(source_patch, 1)
                np.rot90(target_patch, 1)

        psf_path = self.target_paths[image_idx].replace('orig', 'psf')
        psf = np.array(Image.open(psf_path))
        psf = psf / np.sum(psf)

        # hist等措施能否让网络收敛得更快？尽管指标提升不了
        # 实验1：已知psf，粗步deconv，再用小网络细化结果。后面再验证psf的准确性对结果的影响。相当与DL的non-blind算法
        # 实验2：估计粗略的psf，当作预处理，再先deconv后用小网络细化结果。或者直接套入一个小型non-blind网络，即网络学习到的是对deconv的细化
        # 实验3：小网络仅用于预测psf，采用deonv与L1结合的新损失函数优化网络，相当于网络学习到的是estimation。像是PSNRLoss，目的使预测时的psnr更高
        # 应该多看最新的non-blind的方法

        source_patch = 255 - source_patch
        target_patch = 255 - target_patch

        source_patch = source_patch / 255.0
        target_patch = target_patch / 255.0

        source_patch = spatial_deconvolution(source_patch, psf, 6, 1)

        source_patch = torch.from_numpy(source_patch).permute(2, 0, 1).float()
        target_patch = torch.from_numpy(target_patch).permute(2, 0, 1).float()

        return source_patch, target_patch


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


if __name__ == '__main__':
    source_paths_test_txt = '/data/datasets/TEXT_DEBLUR/blur_test.txt'
    target_paths_test_txt = '/data/datasets/TEXT_DEBLUR/gt_test.txt'
    source_paths_test, target_paths_test = extract_path_pairs(source_paths_test_txt, target_paths_test_txt,
                                                              shuffle=False)
    print('dataset test images num:', len(source_paths_test))

    patch_size_test = 192
    patch_num_per_img_test = 1
    fix_img_size_test = (200, 200)
    extract_random_patch_test = False
    augment = True

    dataset_test = REC_DATASET(source_paths_test, target_paths_test, patch_size_test, patch_num_per_img_test,
                               fix_img_size_test, extract_random_patch_test, augment)
    loader_test = DataLoader(dataset=dataset_test, num_workers=0, batch_size=8, shuffle=False)

    criterion = torch.nn.L1Loss()

    for i, (source_tensor, target_tensor) in enumerate(loader_test):
        # print(source_tensor.shape, target_tensor.shape, torch.min(source_tensor), torch.max(source_tensor))

        source_tensor, target_tensor = source_tensor.cuda(), target_tensor.cuda()

        loss = criterion(source_tensor, target_tensor)

        # print(loss.shape, type(loss), loss.dtype, loss.requires_grad, loss.item())

        source_b = (np.array(source_tensor[0, 0, :, :].cpu()) * 255).astype('uint8')
        target_b = (np.array(target_tensor[0, 0, :, :].cpu()) * 255).astype('uint8')

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(source_b)
        plt.subplot(1, 2, 2)
        plt.imshow(target_b)
        plt.savefig('tmp/' + str(i) + '_deconv.png')
