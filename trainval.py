from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
import os
import shutil
import numpy as np
import cv2
import sys
import time

from tools import self_ensemble_rot4, batch_PSNR, extract_path_pairs
from dataset import REC_DATASET
from choices import choose_model, choose_loss

########################################################### hyperparameters 
end_epoch = 10
batch_size = 128
patch_size = 128
patch_num_per_img = 4
lr = 5e-4
resume_path = 'results/text_deblur/unetResume/max_test_psnr.pt'
save_dir = './results/text_deblur/unetResume2'
train_image_num = None
source_paths_train_txt = '/data/datasets/TEXT_DEBLUR/blur_train.txt'
target_paths_train_txt = '/data/datasets/TEXT_DEBLUR/gt_train.txt'
fix_img_size = (300, 300)
extract_random_patch = True
augment = True
print_train_iter_freq = 0.005
model_name = 'unet'
loss_name = 'l1'
self_ensemble = False
resume_optim = False
num_workers = 0

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

########################################################### prepare 

model = choose_model(model_name)

criterion = choose_loss(loss_name)

if fix_img_size is not None:
    torch.backends.cudnn.benchmark = True

source_paths_train, target_paths_train = extract_path_pairs(source_paths_train_txt, target_paths_train_txt, shuffle=True)
source_paths_test, target_paths_test = extract_path_pairs(source_paths_test_txt, target_paths_test_txt, shuffle=True)
print('dataset train images num:', len(source_paths_train))
print('dataset test images num:', len(source_paths_test))

if train_image_num is not None and train_image_num < len(source_paths_train):
    source_paths_train = source_paths_train[:train_image_num]
    target_paths_train = target_paths_train[:train_image_num]
if test_image_num is not None and test_image_num < len(source_paths_test):
    source_paths_test = source_paths_test[:test_image_num]
    target_paths_test = target_paths_test[:test_image_num]
print('selected train images num:', len(source_paths_train))
print('selected test images num:', len(source_paths_test))

dataset_train = REC_DATASET(source_paths_train, target_paths_train, patch_size, patch_num_per_img, fix_img_size, extract_random_patch, augment)
dataset_test = REC_DATASET(source_paths_test, target_paths_test, patch_size_test, patch_num_per_img_test, fix_img_size_test, extract_random_patch_test, augment)
print('train patches num:', len(dataset_train))
print('test patches num:', len(dataset_test))

loader_train = DataLoader(dataset=dataset_train, num_workers=num_workers, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(dataset=dataset_test, num_workers=num_workers_test, batch_size=batch_size, shuffle=False)

ep_train_iter_num = len(loader_train)
ep_test_iter_num = len(loader_test)
print_train_iter_num = max(int(print_train_iter_freq * ep_train_iter_num), 1)
vis_test_iter_num = max(int(vis_test_iter_freq * ep_test_iter_num), 1)

milestones = [i * end_epoch // 10 for i in range(4, 10, 2)]

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

if resume_path is not None:
    if not os.path.exists(resume_path):
        print('resume path not exist.')
        sys.exit()
    print('load', resume_path)
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['state_dict'])

    if resume_optim:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
    else:
        start_epoch = 1

    print('resume from epoch', start_epoch)
else:
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    start_epoch = 1
    print('begin training from scratch.')

########################################################### train

log_dir = save_dir + '/log'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
writer = SummaryWriter(log_dir)

min_test_loss = 1000
max_test_psnr = -1000
model.cuda()
train_iter_cnt = 1

for epoch in range(start_epoch, end_epoch + 1):

    model.train()
    ep_train_loss = 0
    time_pre = time.time()
    for ep_iter, (source, target) in enumerate(loader_train):
        ep_iter += 1
        time_cur = time.time()

        source = source.cuda()
        target = target.cuda()

        if self_ensemble:
            source_out = self_ensemble_rot4(source, model)
        else:
            source_out = model(source)
    
        source_out = torch.clamp(source_out, 0, 1)
        
        loss = criterion(source_out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()

        if ep_iter % print_train_iter_num == 0:
            psnr_ori = batch_PSNR(source, target, 1.0)
            psnr_out = batch_PSNR(source_out, target, 1.0)

            ep_remain_minutes = ((time_cur - time_pre) / print_train_iter_num) * (ep_train_iter_num - ep_iter) / 60
            time_pre = time_cur

            print('Epoch:[%d], Ep_iter:[%d/%d], Train_iter:[%d], loss:%f, psnr_ori:%.3f, psnr_out:%.3f, ep_remain_minutes:%.3f' % (epoch, ep_iter, ep_train_iter_num, train_iter_cnt, loss_value, psnr_ori, psnr_out, ep_remain_minutes))

            if psnr_out > 0 and psnr_out < 100:
                writer.add_scalar('train_iter_psnr', psnr_out, train_iter_cnt)
            writer.add_scalar('train_iter_loss', loss_value, train_iter_cnt)
            writer.add_scalar('train_iter_lr', optimizer.state_dict()['param_groups'][0]['lr'], train_iter_cnt)

        ep_train_loss += loss_value

        train_iter_cnt += 1

    ep_train_loss /= ep_train_iter_num
    writer.add_scalar('ep_train_loss', ep_train_loss, epoch)
    writer.add_scalar('ep_lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

    ########################################################### eval
    save_img_dir = save_dir + '/save_vis'
    if os.path.exists(save_img_dir):
        shutil.rmtree(save_img_dir)
    os.makedirs(save_img_dir)

    model.eval()
    test_loss = 0
    test_psnr = 0

    for i, (source, target) in enumerate(loader_test):
        with torch.no_grad():
            source = source.cuda()
            target = target.cuda()

            if self_ensemble:
                source_out = self_ensemble_rot4(source, model)
            else:
                source_out = model(source)
            
        source_out = torch.clamp(source_out, 0, 1)

        test_loss += criterion(source, target).item()

        psnr_ori = batch_PSNR(source, target, 1.0)
        psnr_out = batch_PSNR(source_out, target, 1.0)
        print('psnr_ori:{}, psnr_out:{}'.format(psnr_ori, psnr_out))
        
        test_psnr += psnr_out

        if (i + 1) % vis_test_iter_num == 0:
            source_ori = np.array(source[0,:,:,:].permute(1, 2, 0).cpu())
            source_out = np.array(source_out[0,:,:,:].permute(1, 2, 0).cpu())
            target = np.array(target[0,:,:,:].permute(1, 2, 0).cpu())

            source_ori = (source_ori * 255).astype('uint8')
            source_out = (source_out * 255).astype('uint8')
            target = (target * 255).astype('uint8')

            cv2.imwrite(save_img_dir + '/' + str(i) + '_source_' + str(psnr_ori) + '.jpg', source_ori)
            cv2.imwrite(save_img_dir + '/' + str(i) + '_restored_' + str(psnr_out) + '.jpg', source_out)
            cv2.imwrite(save_img_dir + '/' + str(i) + '_target.jpg', target)

    test_loss = test_loss / ep_test_iter_num
    test_psnr = test_psnr / ep_test_iter_num

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
