from choices import choose_model
import torch
from pathlib import Path
import cv2
import numpy as np
import os
import time

image_dir = './demo_in'

save_dir = './demo_out'

model_name = 'UnetTinyRF'

ckpt_path = './results/TextDeblur/UnetTinyRF/max_test_psnr.pt'

divide_image_rate = 2

save_tag = '_UnetTinyRF'

########################################

def predict_image_roi(image, model, factor):
    image_out = image.copy()
    h, w, _ = image.shape
    hh = h // factor * factor
    ww = w // factor * factor
    image_roi = image[:hh, :ww, :]

    input = image_roi / 255.0
    with torch.no_grad():
        input = torch.from_numpy(input).permute(2, 0, 1).float().unsqueeze(0).cuda()
        output = model(input)
        output = torch.clamp(output, 0, 1)
    output = (np.array(output.squeeze(0).permute(1, 2, 0).cpu()) * 255).astype('uint8')

    image_out[:hh, :ww, :] = output
    
    return image_out

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model = choose_model(model_name)

checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint['state_dict'])

model.cuda()
model.eval()

for image_path in Path(image_dir).glob('*.*'):
    print('predicting', image_path.name)
    image = cv2.imread(str(image_path))
    
    time0 = time.time()

    if divide_image_rate > 1:
        image_out = image.copy()
        h, w, _ = image.shape
        hh, ww = h // divide_image_rate, w // divide_image_rate
        for i in range(0, h - hh + 1, hh):
            for j in range(0, w - ww + 1, ww):
                patch = image[i:i+hh, j:j+ww, :]
                patch_out = predict_image_roi(patch, model, 16)
                image_out[i:i+hh, j:j+ww, :] = patch_out
    else:
        image_out = predict_image_roi(image, model, 16)
    
    time1 = time.time()
    print('Cost Time (s):', time1 - time0)

    cv2.imwrite(save_dir + '/' + image_path.name, image)
    cv2.imwrite(save_dir + '/' + image_path.name[:-4] + save_tag + '.png', image_out)

    break