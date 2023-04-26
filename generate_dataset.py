from pathlib import Path
import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil


def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def generate_kernel(s0, rho, theta, ksize):
    theta = np.pi / 2 + (np.pi - theta)

    s0 = s0 + 1e-8
    rho = rho + 1e-8
    assert ksize % 2 == 1
    a0 = (np.cos(theta) * np.cos(theta) / (2 * s0 * s0)) + (np.sin(theta) * np.sin(theta) / (2 * rho * rho * s0 * s0))
    a1 = (np.sin(2 * theta) / (4 * rho * rho * s0 * s0)) - (np.sin(2 * theta) / (4 * s0 * s0))
    a2 = (np.sin(theta) * np.sin(theta) / (2 * s0 * s0)) + (np.cos(theta) * np.cos(theta) / (2 * rho * rho * s0 * s0))
    kradius = ksize // 2
    kernel = np.zeros((ksize, ksize))
    X = [i - kradius for i in range(ksize)]
    Y = X
    for x in range(ksize):
        for y in range(ksize):
            xx = X[x]
            yy = Y[y]
            kernel[x, y] = np.exp(-(a0 * xx * xx + 2 * a1 * xx * yy + a2 * yy * yy))
    kernel = kernel / np.sum(kernel)
    return kernel

def add_gasuss_noise_float(image, mean, std):
    noise = np.random.normal(mean, std, image.shape)
    out = image + noise
    out = np.clip(out, 0, 1.0)
    return out


def cal_mean_grad_val(data_root, crop_size):
    G = []
    paths = [path for path in Path(data_root).rglob('*.*')]
    for path in tqdm(paths):
        hr = cv2.imread(str(path))
        if crop_size is not None:
            H, W, _  = hr.shape
            hr = hr[(H - crop_size) // 2:(H - crop_size) // 2 + crop_size, (W - crop_size) // 2:(W - crop_size) // 2 + crop_size, :]
        hr = hr / 255.0
        
        gray = (hr[:,:,0] + hr[:,:,1] + hr[:,:,2]) / 3
        gx, gy = np.gradient(gray)
        g = np.mean(np.sqrt(gx * gx + gy * gy))
        G.append(g)

    mG = sum(G) / len(G)
    return mG

def save_file_paths(data_root):
    for data_dir in os.listdir(data_root):
        data_dir = data_root + '/' + data_dir

        if not os.path.isdir(data_dir):
            continue

        blur_paths = [str(i) for i in Path(data_dir).glob('*.*') if '_hr.png' not in i.name]

        with open(data_dir + '_blur.txt', 'w') as f:
            for path in blur_paths:
                f.write(path)
                f.write('\n')

        with open(data_dir + '_hr.txt', 'w') as f:
            for path in blur_paths:
                gt_path = data_dir + '/' + Path(path).name.split('_')[0] + '_hr.png'
                f.write(gt_path)
                f.write('\n')

def generate_dataset_deblur(data_root, save_root, crop_size, stimu_num, add_noise=True):
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
        
    mG = 0.039515385207199924
    if mG is None:
        print('calculating mean gradient')
        mG = cal_mean_grad_val(data_root)
    print('mean gradient:', mG)

    all_paths = [i for i in Path(data_root).rglob('*.*')]

    for i, path in enumerate(all_paths):
        hr_name = path.name[:-4]

        hr = cv2.imread(str(path))

        H, W, _  = hr.shape
        if H < 256 or W < 256:
            continue

        if crop_size is not None:
            if H < crop_size or W < crop_size:
                continue
            hr = hr[(H - crop_size) // 2:(H - crop_size) // 2 + crop_size, (W - crop_size) // 2:(W - crop_size) // 2 + crop_size, :]
        
        hr = hr / 255.0

        gray = (hr[:,:,0] + hr[:,:,1] + hr[:,:,2]) / 3
        gy, gx = np.gradient(gray)
        g = np.mean(np.sqrt(gx * gx + gy * gy))
        if g < mG * 0.25:
            continue

        save_dir = save_root + '/' + path.parent.name
        mkdir(save_dir)
        cv2.imwrite(save_dir + '/' + hr_name + '_hr.png', (hr * 255).astype('uint8'))

        for _ in range(stimu_num):
            # k = np.random.choice([3, 5])
            k = 3

            s = np.random.randint(3, 15) / 10                           # sigma0    [0.1, 3.0]
            r = np.random.randint(3, 10) / 10                           # rho       [0.1, 1.0]
            t = np.round(np.random.randint(0, 180) / 180 * np.pi, 2)    # theta     [0.0, 3.14]
            
            v = np.random.randint(50, 500) / 1000000                    # var       [0.000001, 0.0005]
            n = np.round(v ** 0.5, 3)

            kernel = generate_kernel(s, r, t, k)

            blur = cv2.filter2D(hr, -1, kernel)
            blur = np.clip(blur, 0, 1.0)

            if add_noise:
                blur = add_gasuss_noise_float(blur, mean=0, std=n)

            blur_name = hr_name + '_k' + str(k) + '_s' + str(s) + '_r' + str(r) + '_t' + str(t) + '_n' + str(n)
            # print(blur_name)

            cv2.imwrite(save_dir + '/' + blur_name + '.png', (blur * 255).astype('uint8'))

        print('[%3d / %3d] done' % (i + 1, len(all_paths)))

    save_file_paths(save_root)


def generate_dataset_sharpen(data_root, save_root, crop_size):
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
        
    mG = 0.039515385207199924
    if mG is None:
        print('calculating mean gradient')
        mG = cal_mean_grad_val(data_root)
    print('mean gradient:', mG)

    all_paths = [i for i in Path(data_root).rglob('*.*')]

    for i, path in enumerate(all_paths):
        hr_name = path.name[:-4]

        hr = cv2.imread(str(path))

        H, W, _  = hr.shape
        if H < 256 or W < 256:
            continue

        if crop_size is not None:
            hr = hr[(H - crop_size) // 2:(H - crop_size) // 2 + crop_size, (W - crop_size) // 2:(W - crop_size) // 2 + crop_size, :]
        hr = hr / 255.0

        gray = (hr[:,:,0] + hr[:,:,1] + hr[:,:,2]) / 3
        gy, gx = np.gradient(gray)
        g = np.mean(np.sqrt(gx * gx + gy * gy))
        if g < mG * 0.25:
            continue

        save_dir = save_root + '/' + path.parent.name
        mkdir(save_dir)
        cv2.imwrite(save_dir + '/' + hr_name + '_hr.png', (hr * 255).astype('uint8'))

        blur = cv2.blur(hr, (5,5))
        sharpen = hr + (hr - blur) * 1.0
        sharpen = np.clip(sharpen, 0, 1)

        sharpen_name = hr_name + '_sharpen'

        cv2.imwrite(save_dir + '/' + sharpen_name + '.png', (sharpen * 255).astype('uint8'))

        print('[%3d / %3d] done' % (i + 1, len(all_paths)))

    save_file_paths(save_root)

if __name__ =='__main__':
    # data_root = '/data/datasets/DIV2K'
    # save_root = '/data/datasets/DEBLUR_DIV2K_50'
    # stimu_num = 50
    # crop_size = 1024
    # generate_dataset_deblur(data_root, save_root, crop_size, stimu_num)

    # data_root = '/data/datasets/SMALL'
    # save_root = '/data/datasets/DEBLUR_SMALL_50'
    # stimu_num = 50
    # crop_size = None
    # generate_dataset_deblur(data_root, save_root, crop_size, stimu_num)

    # data_root = '/data/datasets/SMALL'
    # save_root = '/data/datasets/DEBLUR_SMALL_2'
    # stimu_num = 2
    # crop_size = None
    # generate_dataset_deblur(data_root, save_root, crop_size, stimu_num)

    # data_root = '/data/datasets/SMALL'
    # save_root = '/data/datasets/SHARPEN_SMALL'
    # crop_size = None
    # generate_dataset_sharpen(data_root, save_root, crop_size)

    data_root = '/data/datasets/DIV2K'
    save_root = '/data/datasets/DEBLUR_DIV2K_NONOISE'
    stimu_num = 5
    crop_size = 1024
    generate_dataset_deblur(data_root, save_root, crop_size, stimu_num, add_noise=False)
