from torch.utils.data import Dataset
import cv2
import random
import torch
from tqdm import tqdm

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

class REC_DATASET(Dataset):
    def __init__(self, source_paths, target_paths, patch_size, patch_stride):
        super(REC_DATASET, self).__init__()
        print('Generating patches_idx')
        patches_idx = []
        image_idx = 0
        for path in tqdm(source_paths):
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            h, w, _ = img.shape
            if patch_size > 0 and patch_size > 0:
                try:
                    assert patch_size < h and patch_size < w
                except:
                    print(patch_size, h, w, path)
                for i in range(0, h - patch_size + 1, patch_stride):
                    for j in range(0, w - patch_size + 1, patch_stride):
                        patches_idx.append([image_idx, i, i + patch_size, j, j + patch_size])
            else:
                patches_idx.append([image_idx, 0, h, 0, w])
            image_idx += 1
        random.shuffle(patches_idx)
        print('Generate patches_idx done. Num of patches_idx:', len(patches_idx))

        self.source_paths = source_paths
        self.target_paths = target_paths
        self.patches_idx = patches_idx

    def __len__(self):
        return len(self.patches_idx)

    def __getitem__(self, idx):
        image_idx = self.patches_idx[idx][0]
        patch_cord_h0 = self.patches_idx[idx][1]
        patch_cord_h1 = self.patches_idx[idx][2]
        patch_cord_w0 = self.patches_idx[idx][3]
        patch_cord_w1 = self.patches_idx[idx][4]

        source_image = cv2.imread(self.source_paths[image_idx])
        target_image = cv2.imread(self.target_paths[image_idx])

        source_patch = source_image[patch_cord_h0:patch_cord_h1, patch_cord_w0:patch_cord_w1, :]
        target_patch = target_image[patch_cord_h0:patch_cord_h1, patch_cord_w0:patch_cord_w1, :]

        source_patch = source_patch / 255.0
        target_patch = target_patch / 255.0

        source_patch = torch.from_numpy(source_patch).permute(2, 0, 1).float()
        target_patch = torch.from_numpy(target_patch).permute(2, 0, 1).float()

        return source_patch, target_patch