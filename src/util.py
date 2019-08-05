# Helper function
import tensorflow as tf
import cv2
import imageio
import numpy as np

def dataParserFOV(dataset_path):
    features = tf.parse_single_example(dataset_path, features={
        'input_im' : tf.FixedLenFeature([], tf.string),
        'label_im' : tf.FixedLenFeature([], tf.string),
        'label_fov': tf.FixedLenFeature([], tf.int64),
        'pano_im' : tf.FixedLenFeature([], tf.string)
    })
    input_im = tf.decode_raw(features['input_im'], tf.uint8)
    label_im = tf.decode_raw(features['label_im'], tf.uint8)
    pano_im = tf.decode_raw(features['pano_im'], tf.uint8)
    
    input_im = tf.reshape(input_im, [256,256,3*6])
    label_im = tf.reshape(label_im, [256,256,3*6])
    pano_im = tf.reshape(pano_im,  [512,1024,3])
    
    input_im = tf.cast(input_im, tf.float32)/127.5 - 1
    label_im = tf.cast(label_im, tf.float32)/127.5 - 1
    pano_im = tf.cast(pano_im, tf.float32)/127.5 - 1
    
    
    # Set to one hot encoding
    label_fov = features['label_fov']
    label_fov = tf.cast(label_fov/2-1,tf.int64)# Resize label to half
    fov_hot = tf.one_hot(label_fov,128) 
    
    return input_im, label_im, fov_hot, pano_im

def initDataset(dataset_path, batch_sz=1, shuffle_buff=50, shuffle=True):
    dataset = tf.data.TFRecordDataset(dataset_path)
    dataset = dataset.map(dataParserFOV)
    if shuffle:
        dataset = dataset.batch(batch_sz).repeat().shuffle(shuffle_buff)
    else:
        dataset = dataset.batch(batch_sz).repeat()
    iterator = dataset.make_initializable_iterator()
    return iterator

def dataParserNew(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded,[128,512])
    label = tf.cast(label/2-1, tf.int64)
    return image_resized, label

def initDatasetNew(dataset_path, batch_sz=1, shuffle_buff=50, shuffle=True):
    data_path = '/home/juliussurya/work/360dataset/SUN360/scripts/dir_list_tf'
    im_file = []
    with open(data_path, 'r') as f:
        for line in f:
            l = line.strip()
            im_file.append(l)

    fov_path = '/home/juliussurya/work/360dataset/SUN360/scripts/dir_list_fov'
    fov_file = []
    with open(fov_path, 'r') as f:
        for line in f:
            l = line.strip()
            fov_file.append(float(l))

    dataset = tf.data.Dataset.from_tensor_slices((im_file, fov_file))
    dataset = dataset.map(dataParserNew)
    dataset = dataset.batch(batch_sz).repeat().shuffle(shuffle_buff)
    iterator = dataset.make_initializable_iterator()
    return iterator

def catResize(y_,resize=[128,512],name=None):
    y = tf.concat([y_[:,:,:,6:9],y_[:,:,:,0:3],y_[:,:,:,9:12],y_[:,:,:,3:6]],axis=2)
    y = tf.image.resize_bilinear(y,resize,name=name)
    return y

def catVerticalResize(y, resize=[384,128], name=None):
    y = tf.concat((y[:,:,:,12:15],y[:,:,:,0:3],y[:,:,:,15:18]),axis=1)
    y = tf.image.resize_bilinear(y,resize,name=name)
    return y

def catVerticalResizeV2(y, resize=[384,128], name=None):
    top_im = y[:,:,:,12:15]
    bot_im = y[:,:,:,15:18]

    y1 = tf.concat((rot90(top_im), y[:,:,:,6:9], rot270(bot_im)),axis=1)
    y1 = tf.image.resize_bilinear(y1,resize)
    y2 = tf.concat((top_im, y[:,:,:,0:3], bot_im),axis=1)
    y2 = tf.image.resize_bilinear(y2,resize)
    y3 = tf.concat((rot270(top_im), y[:,:,:,9:12], rot90(bot_im)),axis=1)
    y3 = tf.image.resize_bilinear(y3,resize)
    y4 = tf.concat((rot180(top_im), y[:,:,:,3:6], rot180(bot_im)),axis=1)
    y4 = tf.image.resize_bilinear(y4,resize)

    y_recon = tf.concat((y1,y2,y3,y4), axis=2)

    return y_recon

def catResizeV2(y_,resize=[128,640],name=None):
    y = tf.concat([y_[:,:,:,3:6],
                   y_[:,:,:,6:9],
                   y_[:,:,:,0:3],
                   y_[:,:,:,9:12],
                   y_[:,:,:,3:6]],
                  axis=2)
    y = tf.image.resize_bilinear(y,resize,name=name)
    return y

def catConst(y ):
    y_side = tf.image.crop_to_bounding_box(y,0,383,128,128)
    y_recon = tf.concat([y_side, y],axis=2)
    return y_recon

def constToNorm(y, bbox=[0,127,128,512]):
    b1, b2, b3, b4 = bbox
    y = tf.image.crop_to_bounding_box(y,b1,b2,b3,b4)
    return y

def constToNormBlend(y, bbox=[0,127,128,512]):
    y1 = tf.image.crop_to_bounding_box(y,0,0,32,32)
    y2 = tf.image.crop_to_bounding_box(y,0,127,32,32)
    y_avg = 0.5 * (y1 + y2)
    y = tf.image.crop_to_bounding_box(y,0,31,32,96)
    return tf.concat((y,y_avg), axis=2)

def procInputFov(x,crop,resize=[128,512], name=None):
    crop0 = crop[0]
    crop1 = crop[1]
    
    x0_0 =x[0,:,0:128,:]
    x0_0 = tf.reshape(x0_0,[1,128,128,3])
    x0_0 = tf.image.resize_bilinear(x0_0,[crop0,crop0])
    x0_0 = tf.image.resize_image_with_crop_or_pad(x0_0,128,128)
    x0_1 =x[1,:,0:128,:]
    x0_1 = tf.reshape(x0_1,[1,128,128,3])
    x0_1 = tf.image.resize_bilinear(x0_1,[crop1,crop1])
    x0_1 = tf.image.resize_image_with_crop_or_pad(x0_1,128,128)
    x0 = tf.concat((x0_0,x0_1),axis=0)
    
    x1_0 =x[0,:,128:256,:]
    x1_0 = tf.reshape(x1_0,[1,128,128,3])
    x1_0 = tf.image.resize_bilinear(x1_0,[crop0,crop0])
    x1_0 = tf.image.resize_image_with_crop_or_pad(x1_0,128,128)
    x1_1 =x[1,:,128:256,:]
    x1_1 = tf.reshape(x1_1,[1,128,128,3])
    x1_1 = tf.image.resize_bilinear(x1_1,[crop1,crop1])
    x1_1 = tf.image.resize_image_with_crop_or_pad(x1_1,128,128)
    x1 = tf.concat((x1_0,x1_1),axis=0)
    
    x2_0 =x[0,:,256:384,:]
    x2_0 = tf.reshape(x2_0,[1,128,128,3])
    x2_0 = tf.image.resize_bilinear(x2_0,[crop0,crop0])
    x2_0 = tf.image.resize_image_with_crop_or_pad(x2_0,128,128)
    x2_1 =x[1,:,256:384,:]
    x2_1 = tf.reshape(x2_1,[1,128,128,3])
    x2_1 = tf.image.resize_bilinear(x2_1,[crop1,crop1])
    x2_1 = tf.image.resize_image_with_crop_or_pad(x2_1,128,128)
    x2 = tf.concat((x2_0,x2_1),axis=0)
    
    x3_0 =x[0,:,384:512,:]
    x3_0 = tf.reshape(x3_0,[1,128,128,3])
    x3_0 = tf.image.resize_bilinear(x3_0,[crop0,crop0])
    x3_0 = tf.image.resize_image_with_crop_or_pad(x3_0,128,128)
    x3_1 =x[1,:,384:512,:]
    x3_1 = tf.reshape(x3_1,[1,128,128,3])
    x3_1 = tf.image.resize_bilinear(x3_1,[crop1,crop1])
    x3_1 = tf.image.resize_image_with_crop_or_pad(x3_1,128,128)
    x3 = tf.concat((x3_0,x3_1),axis=0)
    
    x_ = tf.concat((x0,x1,x2,x3), axis=2, name=name)
    
    return x_


def procInputFovV2(x,crop,resize=[128,512], name=None):
    crop0 = crop[0]
    crop1 = crop[1]

    x0_0 =x[0,:,0:128,:]
    x0_0 = tf.reshape(x0_0,[1,128,128,3])
    x0_0 = tf.image.resize_bilinear(x0_0,[crop0,crop0])
    x0_0 = tf.image.resize_image_with_crop_or_pad(x0_0,128,128)
    x0_1 =x[1,:,0:128,:]
    x0_1 = tf.reshape(x0_1,[1,128,128,3])
    x0_1 = tf.image.resize_bilinear(x0_1,[crop1,crop1])
    x0_1 = tf.image.resize_image_with_crop_or_pad(x0_1,128,128)
    x0 = tf.concat((x0_0,x0_1),axis=0)

    x1_0 =x[0,:,128:256,:]
    x1_0 = tf.reshape(x1_0,[1,128,128,3])
    x1_0 = tf.image.resize_bilinear(x1_0,[crop0,crop0])
    x1_0 = tf.image.resize_image_with_crop_or_pad(x1_0,128,128)
    x1_1 =x[1,:,128:256,:]
    x1_1 = tf.reshape(x1_1,[1,128,128,3])
    x1_1 = tf.image.resize_bilinear(x1_1,[crop1,crop1])
    x1_1 = tf.image.resize_image_with_crop_or_pad(x1_1,128,128)
    x1 = tf.concat((x1_0,x1_1),axis=0)

    x2_0 =x[0,:,256:384,:]
    x2_0 = tf.reshape(x2_0,[1,128,128,3])
    x2_0 = tf.image.resize_bilinear(x2_0,[crop0,crop0])
    x2_0 = tf.image.resize_image_with_crop_or_pad(x2_0,128,128)
    x2_1 =x[1,:,256:384,:]
    x2_1 = tf.reshape(x2_1,[1,128,128,3])
    x2_1 = tf.image.resize_bilinear(x2_1,[crop1,crop1])
    x2_1 = tf.image.resize_image_with_crop_or_pad(x2_1,128,128)
    x2 = tf.concat((x2_0,x2_1),axis=0)

    x3_0 =x[0,:,384:512,:]
    x3_0 = tf.reshape(x3_0,[1,128,128,3])
    x3_0 = tf.image.resize_bilinear(x3_0,[crop0,crop0])
    x3_0 = tf.image.resize_image_with_crop_or_pad(x3_0,128,128)
    x3_1 =x[1,:,384:512,:]
    x3_1 = tf.reshape(x3_1,[1,128,128,3])
    x3_1 = tf.image.resize_bilinear(x3_1,[crop1,crop1])
    x3_1 = tf.image.resize_image_with_crop_or_pad(x3_1,128,128)
    x3 = tf.concat((x3_0,x3_1),axis=0)

    x_ = tf.concat((x3,x0,x1,x2,x3), axis=2, name=name)

    return x_

def createMask(crop, pad=None, max_sz=128, name=None):
    crop0 = crop[0]
    crop1 = crop[1]
    
    # PER-CHANNEL
    mask0 = tf.ones([1,crop0,crop0,3])
    mask0 = tf.image.resize_image_with_crop_or_pad(mask0,128,128)
    mask0 = tf.concat((mask0,mask0,mask0,mask0),axis=2)
    
    # PER-CHANNEL
    mask1 = tf.ones([1,crop1,crop1,3])
    mask1 = tf.image.resize_image_with_crop_or_pad(mask1,128,128)
    mask1 = tf.concat((mask1,mask1,mask1,mask1),axis=2)
    
    mask = tf.concat((mask0,mask1), axis=0, name=name) #concat batch
    
    
    return mask

def dstackPano(x, x_in=None,  n_top=None, n_bot=None, size=[128,128]):
    x1 = tf.image.crop_to_bounding_box(x,0,0,128,128)
    x2 = tf.image.crop_to_bounding_box(x,0,127,128,128)
    x3 = tf.image.crop_to_bounding_box(x,0,255,128,128)
    x4 = tf.image.crop_to_bounding_box(x,0,383,128,128)

    x1 = tf.image.resize_bilinear(x1, size)
    x2 = tf.image.resize_bilinear(x2, size)
    x3 = tf.image.resize_bilinear(x3, size)
    x4 = tf.image.resize_bilinear(x4, size)

    xd = tf.concat((x1,x2,x3,x4),axis=2)

    pad = tf.zeros_like(xd)
    x_recon = tf.concat((pad, xd, pad),axis=1)

    return x_recon

def rot90(x):
    return tf.image.rot90(x,1)

def rot180(x):
    return tf.image.rot90(x,2)

def rot270(x):
    return tf.image.rot90(x,3)

def tanhToRGB(im):
    im = ((im+1)/2) * 255
    im = im.astype('uint8')
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return im

def loadImage(x, normalize=False):
    im = cv2.imread(x)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if normalize:
        im = im.astype(np.float32)
        im = im/127.5 - 1
    return im

def resizeImage(x, size=[256,256]):
    im = cv2.resize(x,(size[0],size[1]))
    return im

def saveImage(x, save_path, name_idx):
    x = tanhToRGB(x[0,:,:,:])
    cv2.imwrite(save_path + name_idx, x)
    # imageio.imwrite(save_path+name_idx,x)

def createVmask(size=[32,32,12]):
    zeros = tf.zeros(size)
    ones = zeros + 1

    m1 = tf.concat((ones,zeros,ones), axis=0)
    m1 = tf.expand_dims(m1, 0)

    return m1

def makeCubePano(ypred):
    yp_top = ypred[:,0:128,:,:]
    yp_bot = ypred[:,256:384,:,:]

    yt = yp_top[:,:,128:256,:] + rot180(yp_top[:,:,384:512,:]) + \
         rot270(yp_top[:,:,0:128,:]) + rot90(yp_top[:,:,256:384,:])
    yt = yt / 4.0
    # yt = yp_top[:,:,128:256,:]

    yb = yp_bot[:,:,128:256,:] + rot180(yp_bot[:,:,384:512,:]) + \
         rot90(yp_bot[:,:,0:128,:]) + rot270(yp_bot[:,:,256:384,:])
    yb = yb / 4.0
    # yb = yp_bot[:,:,128:256,:]

    pad = tf.zeros_like(yp_top[:,:,128:256,:])

    y_top_pad = tf.concat((pad,yt,pad,pad),axis=2)
    y_bot_pad = tf.concat((pad,yb,pad,pad),axis=2)
    y_horizontal = ypred[:,128:256,:,:]

    y_proc = tf.concat((y_top_pad, y_horizontal, y_bot_pad),axis=1)

    return y_proc


