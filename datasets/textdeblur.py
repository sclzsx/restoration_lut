from pathlib import Path
import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

def process(data_root):
    gt_paths_train = [str(i) for i in Path(data_root + '/train').glob('*_orig.png')]
    blur_paths_train = [i.replace('_orig', '_blur') for i in gt_paths_train]

    gt_paths_test = [str(i) for i in Path(data_root + '/test/orig').glob('*_orig.png')]
    blur_paths_test = [i.replace('_orig', '_blur').replace('orig', 'n_00') for i in gt_paths_test]

    with open(data_root + '/blur_train.txt', 'w') as f:
        for path in blur_paths_train:
            assert os.path.exists(path) == 1
            f.write(path)
            f.write('\n')

    with open(data_root + '/gt_train.txt', 'w') as f:
        for path in gt_paths_train:
            assert os.path.exists(path) == 1
            f.write(path)
            f.write('\n')

    with open(data_root + '/blur_test.txt', 'w') as f:
        for path in blur_paths_test:
            assert os.path.exists(path) == 1
            f.write(path)
            f.write('\n')

    with open(data_root + '/gt_test.txt', 'w') as f:
        for path in gt_paths_test:
            assert os.path.exists(path) == 1
            f.write(path)
            f.write('\n')


if __name__ =='__main__':
    data_root = '/data/datasets/TEXT_DEBLUR'
    process(data_root)
