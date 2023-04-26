from torch.utils.data import DataLoader
import torch
import os
import shutil
import torch.nn.functional as F
from dataset import REC_DATASET, extract_path_pairs
from model3 import UNet
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter
import sys
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


def model_rot4_invrot4(source, model):
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
    

end_epoch = 10
batch_size = 16
patch_size = 256
patch_stride = patch_size // 2
lr = 1e-3
resume_path = ''
save_dir = './resultsUNet'
train_image_num = 1000
milestones = [i * end_epoch // 10 for i in range(4, 10, 2)]
TRAIN_SOURCE_PATHS_TXT = '/data/datasets/DEBLUR_DIV2K_NONOISE/DIV2K_train_HR_blur.txt'
TRAIN_TARGET_PATHS_TXT = '/data/datasets/DEBLUR_DIV2K_NONOISE/DIV2K_train_HR_hr.txt'


patch_size_test = 0
patch_stride_test = 0
test_image_num = 20
save_vis = True
TEST_SOURCE_PATHS_TXT = '/data/datasets/DEBLUR_DIV2K_NONOISE/DIV2K_valid_HR_blur.txt'
TEST_TARGET_PATHS_TXT = '/data/datasets/DEBLUR_DIV2K_NONOISE/DIV2K_valid_HR_hr.txt'


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

source_paths_train, target_paths_train = extract_path_pairs(TRAIN_SOURCE_PATHS_TXT, TRAIN_TARGET_PATHS_TXT, shuffle=True)
source_paths_test, target_paths_test = extract_path_pairs(TEST_SOURCE_PATHS_TXT, TEST_TARGET_PATHS_TXT, shuffle=True)

if train_image_num is not None and train_image_num < len(source_paths_train):
    source_paths_train = source_paths_train[:train_image_num]
    target_paths_train = target_paths_train[:train_image_num]

if test_image_num is not None and test_image_num < len(source_paths_test):
    source_paths_test = source_paths_test[:test_image_num]
    target_paths_test = target_paths_test[:test_image_num]

dataset_train = REC_DATASET(source_paths_train, target_paths_train, patch_size, patch_stride)
dataset_test = REC_DATASET(source_paths_test, target_paths_test, patch_size_test, patch_stride_test)

# model = SRNet(in_channels=3, n_features=64)
model = UNet(in_ch=3, out_ch=3)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

if os.path.exists(resume_path):
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    start_epoch = checkpoint['epoch']
    print('resume from epoch', start_epoch)
else:
    start_epoch = 1

criterion = torch.nn.MSELoss()

log_dir = save_dir + '/log'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
writer = SummaryWriter(log_dir)

model.cuda()
min_test_loss = 1000
max_test_psnr = -1000
loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=batch_size, shuffle=True)
iter_n_per_epoch = len(loader_train)
n_test_set = len(dataset_test)
for epoch in range(start_epoch, end_epoch + 1):

    model.train()
    train_loss = 0
    for i, (source, target) in enumerate(loader_train):
        source = source.cuda()
        target = target.cuda()

        # source_out = model_rot4_invrot4(source, model)
        source_out = model(source)
    
        source_out = torch.clamp(source_out, 0, 1)
        
        loss = criterion(source_out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            psnr_ori = batch_PSNR(source, target, 1.0)
            psnr_out = batch_PSNR(source_out, target, 1.0)
            print('Epoch:{}, Iter:[{}/{}], loss:{}, psnr_ori:{}, psnr_out:{}'.format(epoch, i + 1, iter_n_per_epoch, loss.item(), psnr_ori, psnr_out))

        writer.add_scalar('train_loss_iter', loss.item(), i + 1)
        train_loss += loss.item()

    train_loss /= iter_n_per_epoch
    writer.add_scalar('train_loss_ep', train_loss, epoch)
    writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

    model.eval()
    test_loss = 0
    test_psnr = 0
    save_img_dir = save_dir + '/save_vis'
    if os.path.exists(save_img_dir):
        shutil.rmtree(save_img_dir)
    os.makedirs(save_img_dir)
    with torch.no_grad():
        for i, (source, target) in enumerate(dataset_test):
            source = source.unsqueeze(0).cuda()
            target = target.unsqueeze(0).cuda()

            # source_out = model_rot4_invrot4(source, model)
            source_out = model(source)

            source_out = torch.clamp(source_out, 0, 1)

            test_loss += criterion(source, target).item()

            psnr_ori = batch_PSNR(source, target, 1.0)
            psnr_out = batch_PSNR(source_out, target, 1.0)
            print('psnr_ori:{}, psnr_out:{}'.format(psnr_ori, psnr_out))
            
            test_psnr += psnr_out

            if save_vis:
                source_ori = np.array(source.squeeze(0).permute(1, 2, 0).cpu())
                source_out = np.array(source_out.squeeze(0).permute(1, 2, 0).cpu())
                target = np.array(target.squeeze(0).permute(1, 2, 0).cpu())

                source_ori = (source_ori * 255).astype('uint8')
                source_out = (source_out * 255).astype('uint8')
                target = (target * 255).astype('uint8')

                cv2.imwrite(save_img_dir + '/' + str(i) + '_source_' + str(psnr_ori) + '.jpg', source_ori)
                cv2.imwrite(save_img_dir + '/' + str(i) + '_restored_' + str(psnr_out) + '.jpg', source_out)
                cv2.imwrite(save_img_dir + '/' + str(i) + '_target.jpg', target)

        test_loss = test_loss / n_test_set
        test_psnr = test_psnr / n_test_set

        writer.add_scalar('test_loss_ep', test_loss, epoch)
        writer.add_scalar('test_psnr', test_psnr, epoch)

        if test_loss < min_test_loss:
            min_test_loss = test_loss

            checkpoint = {
                "state_dict": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, save_dir + '/min_test_loss.pt')
            print('-' * 50)
            print('Saved checkpoint as min_test_loss:', min_test_loss)
            print('-' * 50)
    
        if test_psnr > max_test_psnr:
            max_test_psnr = test_psnr

            checkpoint = {
                "state_dict": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, save_dir + '/max_test_psnr.pt')
            print('-' * 50)
            print('Saved checkpoint as max_test_psnr:', max_test_psnr)
            print('-' * 50)
