import os, sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

# idx = sys.argv[1]

im_path = '../output/v3.5_2/'
save_path = '../output/consistency/'

# base_path = '/home/juliussurya/Dropbox (IVCL)/lab_affair/iccp19/figures/test_result_resized/'

def imRead(im_path):
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

def makeCubePano(im):
    # im_horiz = im[64:128, :, :]
    im_horiz = im[128:256, :, :]
    # im_top = im[0:64, 64:128, :]
    im_top = im[0:128, 128:256, :]
    im_bot = im[256:384, 128:256, :]
    # im_bot = im[128:192, 64:128, :]
    pad = np.zeros_like(im_bot) + 127

    cube_top = np.hstack((pad,im_top,pad,pad))
    cube_bot = np.hstack((pad,im_bot,pad,pad))
    im_cube = np.vstack((cube_top, im_horiz, cube_bot))

    return im_cube

def saveImg(im, im_path):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(im_path,im)

base_path = '/media/juliussurya/JULIUSSURYA/test_four_7/out_pix'
im = imRead(base_path + '.png')
# im = imRead(base_path + str(idx) + '.jpg')
# print(im.shape)
im_proc = makeCubePano(im)
# im_proc = im
saveImg(im_proc, base_path + '_.png')


# im_full = cv2.imread(im_path + 'output_full' + str(idx) +'.png')
# # im_full = cv2.cvtColor(im_full, cv2.COLOR_BGR2RGB)
#
# im = im_full[128:256,:,:]
# im_small = im[:,384:512,:]
# im_rest = im[:,0:384,:]
#
# im_proc = np.hstack((im_small, im_rest))
#
# im_h = cv2.imread(im_path + 'output_horizontal' + str(idx) +'.png')
# # im_h = cv2.cvtColor(im_h, cv2.COLOR_BGR2RGB)
#
# imh_small = im_h[:,384:512,:]
# imh_rest = im_h[:,0:384,:]
#
#
# imh_proc = np.hstack((imh_small, imh_rest))
#
#
# im_gt = cv2.imread(im_path + 'gt' + str(idx) +'.png')
# im_gt = cv2.cvtColor(im_gt, cv2.COLOR_BGR2RGB)
#
#
# cv2.imwrite(save_path + 'im'+ str(idx) +'.png', im_proc)
# cv2.imwrite(save_path + 'im_proc'+ str(idx) +'.png', imh_proc)
# # cv2.imwrite(save_path + 'im_gt'+ str(idx) +'.png')

# plt.imshow(im_gt)
# plt.show()
#
#
# fig = plt.figure(figsize=(2,2))
# fig.add_subplot(2,2,1)
# plt.imshow(im)
# fig.add_subplot(2,2,2)
# plt.imshow(im_proc)
#
# fig.add_subplot(2,2,3)
# plt.imshow(im_h)
# fig.add_subplot(2,2,4)
# plt.imshow(imh_proc)
#
#
# plt.show()


