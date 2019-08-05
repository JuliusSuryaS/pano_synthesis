import os, sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import compare_mse, compare_psnr, compare_ssim
from skimage import img_as_float as imFloat



vert_fake_data_path = '/home/juliussurya/workspace/360pano/incept_score/inception-score-pytorch/vertical_fake/pano'
vert_real_data_path = '/home/juliussurya/workspace/360pano/incept_score/inception-score-pytorch/vertical_real/pano'
horiz_real_data_path = '/home/juliussurya/workspace/360pano/incept_score/inception-score-pytorch/horizontal_real/pano'
horiz_fake_data_path = '/home/juliussurya/workspace/360pano/incept_score/inception-score-pytorch/horizontal_fake/pano'


def getImagesList(data_path):
    im_list = []
    for root, dirs, files in os.walk(data_path):
        for file in sorted(files):
            im_path = os.path.join(root, file)
            im_list.append(im_path)
    return im_list

def imRead(im_path):
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

def computeScore(fake_data, real_data, total_data=5000):
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    for i in range(total_data):
        im = imFloat(imRead(fake_data[i]))
        gt = imFloat(imRead(real_data[i]))

        mse_score = compare_mse(im, gt)
        psnr_score = compare_psnr(gt, im, data_range=im.max()-im.min())
        ssim_score = compare_ssim(im, gt, data_range=im.max()-im.min(),multichannel=True)

        total_mse = total_mse + mse_score
        total_psnr = total_psnr + psnr_score
        total_ssim = total_ssim + ssim_score

    return total_mse/total_data, total_psnr/total_data, total_ssim/total_data

# IMAGE LIST
vert_real = getImagesList(vert_real_data_path)
vert_fake = getImagesList(vert_fake_data_path)
horiz_fake = getImagesList(horiz_fake_data_path)
horiz_real = getImagesList(horiz_real_data_path)

total_data = len(vert_real)
v_mse, v_psnr, v_ssim = computeScore(vert_fake, vert_real)
h_mse, h_psnr, h_ssim = computeScore(horiz_fake, horiz_real)


print('Vertical - MSE:', v_mse, 'PSNR:', v_psnr, 'SSIM:', v_ssim)
print('Horizontal - MSE:', h_mse, 'PSNR:', h_psnr, 'SSIM:', h_ssim)

# Result
#Vertical - MSE: 0.025534946475992918 PSNR: 16.952006037939814 SSIM: 0.45972641654783586
#Horizontal - MSE: 0.018181969473755163 PSNR: 17.07570159328891 SSIM: 0.45889364436274716