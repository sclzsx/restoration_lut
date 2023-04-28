from torch.utils.data import Dataset
import cv2
import random
import torch
from tqdm import tqdm
import os
import numpy as np
import sys

class REC_DATASET(Dataset):
    def __init__(self, source_paths, target_paths, patch_size, patch_num_per_img, fix_img_size, extract_random_patch, augment):
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

        random.shuffle(patches_info)

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

        source_patch = source_patch / 255.0
        target_patch = target_patch / 255.0

        source_patch = torch.from_numpy(source_patch).permute(2, 0, 1).float()
        target_patch = torch.from_numpy(target_patch).permute(2, 0, 1).float()

        return source_patch, target_patch