import tensorflow as tf
from tensorflow.layers import *
import modlayers as ml
import bilinearSampler as bs
import util

# Name wrappers
sigmoidEnt = tf.nn.sigmoid_cross_entropy_with_logits  # type: function
sum = tf.add
deconv2d = conv2d_transpose
lrelu = tf.nn.leaky_relu
tanh = tf.nn.tanh
relu = tf.nn.relu
batchNorm = batch_normalization

def dstack(x1,x2):
    return tf.concat((x1,x2),axis=-1)

def fovNet(x,ksz=3,s=2):
    with tf.variable_scope('Fov'):
        enc1 = lrelu(conv2d(x,16,ksz,s,'same',name='f_enc1')) #(64,256)
        enc2 = lrelu(conv2d(enc1,32,ksz,s,'same',name='f_enc2')) #(32, 128)
        enc3 = lrelu(conv2d(enc2,64,ksz,s,'same', name='f_enc3')) #(16, 64)
        enc4 = lrelu(conv2d(enc3,128,ksz,s,'same', name='f_enc4')) #(8,32)
        enc5 = lrelu(conv2d(enc4,256,ksz,s,'same', name='f_enc5'))#(4,16)
        enc6 = lrelu(conv2d(enc5,512,ksz,s,'same', name='f_enc6')) #(2,8)
        fc = flatten(enc6)
        fc = lrelu(dense(fc,1024, name='f_fc1'))
        fc = lrelu(dense(fc,512, name='f_fc2'))
        fc = dense(fc,128, name='f_fc3')
        fc_max = tf.nn.softmax(fc)
        fov = tf.argmax(fc_max,axis=-1)
        fov = tf.cast(fov, tf.int32)
        return fc, tf.clip_by_value(fov, 1, 128)

def Ds(x,ksz=3,s=2,prob=0.5,reuse=False):
    init = tf.random_normal_initializer(stddev=0.02)
    with tf.variable_scope('DiscriminatorS', reuse=reuse):
        d = ml.convLrelu(x,32,ksz,s,'same',init=init,name='ds_enc1')#(16,64)
        d = ml.dropout(d, prob,name='drop1')
        d = ml.convLrelu(d,64,ksz,s,'same', init=init,name='ds_enc2')#(8,32)
        d = ml.dropout(d, prob,name='drop2')
        d = ml.convLrelu(d,64,ksz,s,'same',init=init,name='ds_enc3')#(4,16)
        d = ml.dropout(d, prob,name='drop3')
        d = ml.convLrelu(d,64,ksz,s,'same',init=init,name='ds_enc4')#(4,16)
        d = ml.dropout(d, prob,name='drop4')
        d = ml.fullConn(d, 1, init=init, name='d_fc1')
        return d, tf.reduce_mean(ml.sigmoid(d))
    
def Dm(x,ksz=5,s=2,prob=0.5,reuse=False):
    init = tf.random_normal_initializer(stddev=0.02)
    with tf.variable_scope('DiscriminatorM', reuse=reuse):
        d = ml.convLrelu(x,64,ksz,s,'same',init=init,name='dm_enc1')#(32,128)
        d = ml.convLrelu(d,128,ksz,s,'same',init=init,name='dm_enc2')#(16,64)
        d = ml.convLrelu(d,128,ksz,s,'same',init=init,name='dm_enc3')#(8,32)
        d = ml.convLrelu(d,128,ksz,s,'same',init=init,name='dm_enc4')#(4,16)
        #d = ml.flatten(d)
        d = ml.dropout(d, prob, name='drop1')
        d = ml.fullConn(d, 1, init=init, name='d_fc1')
        return d, tf.reduce_mean(ml.sigmoid(d))

def D(x,ksz=5,s=2,prob=0.5,reuse=False):
    init = tf.random_normal_initializer(stddev=0.02)
    with tf.variable_scope('DiscriminatorH', reuse=reuse):
        d = ml.convLrelu(x,64,ksz,s,'same', init=init,name='d_enc1')#(64,256)
        d = ml.convLrelu(d,128,ksz,s,'same', init=init, name='d_enc2')#(32,128)
        d = ml.convLrelu(d,128,ksz,s,'same',init=init,name='d_enc3')#(16,64)
        d = ml.convLrelu(d,128,ksz,s,'same',init=init,name='d_enc4')#(8,32)
        d = ml.convLrelu(d,256,ksz,s,'same',init=init,name='d_enc5')#(8,32)
        d = ml.dropout(d, prob, name='drop1')
        d = ml.fullConn(d,1,init=init,name='d_fc1')
        # d = ml.dropout(d,prob,name='drop4')
        # d = ml.convBNLrelu(d,256,ksz,s,'same',init=init,name='d_enc5')#(4,16)
        # d = ml.dropout(d,prob,name='drop5')
        # d = ml.conv(d,512,ksz,s,'same',init=init,name='d_enc6')#(2,8)
        return d, tf.reduce_mean(ml.sigmoid(d))


def Gs3(x,ksz=3,s=2,init='he_uniform',reuse=False):
    with tf.variable_scope('Generator',reuse=reuse):
        enc_h = ml.convLrelu(x,64,ksz,s,'same',name='g_enc1')#(64,256)
        enc_m = ml.convLrelu(enc_h,128,ksz,s,'same',name='g_enc2')#(32,128)
        enc1 = lrelu(conv2d(enc_m,256,ksz,s,'same',name='g_enc3'))#(16,64)
        enc2 = lrelu(conv2d(enc1,256,ksz,s,'same',name='g_enc4'))#(8,32)
        enc3 = lrelu(conv2d(enc2,256,ksz,s,'same',name='g_enc5'))#(4,16)
        enc4 = lrelu(conv2d(enc3,512,ksz,s,'same',name='g_enc6'))#(2,8)
        enc5 = lrelu(conv2d(enc4,512,ksz,s,'same',name='g_enc7'))#(1,4)
        dec1 = lrelu(conv2d(enc5,512,ksz,s,'same',name='g_dec1'))#(2,8)
        dec2 = lrelu(conv2d(dstack(dec1,enc4),512,ksz,s,'same',name='g_dec2'))#(4,16)
        dec3 = lrelu(conv2d(dstack(dec2,enc3),256,ksz,s,'same',name='g_dec3'))#(8,32)
        dec4 = lrelu(conv2d(dstack(dec3,enc2),256,ksz,s,'same',name='g_dec4'))#(16,64)
        dec5 = lrelu(conv2d(dstack(dec4,enc1),128,ksz,s,'same',name='g_dec5'))#(32,128)
        dec6 = lrelu(conv2d(dstack(dec5,enc_m),64,ksz,s,'same',name='g_dec6'))#(64,128)

    # Output high 
        dec_h = tanh(deconv2d(dstack(dec6,enc_h),3,ksz,s,'same',name='im_h'))#(128,256)
    # Output med #1x1 conv
        dec_m = tanh(deconv2d(dec6,3,1,1,'same',name='im_m'))
    # Output small #1x1 conv
        dec_s = tanh(deconv2d(dec5,3,1,1,'same',name='im_s'))

        return dec_s, dec_m, dec_h

def GRefine(x,prob,reuse=False):
    init = tf.random_normal_initializer(stddev=0.02)
    keep_prob = prob
    with tf.variable_scope('GeneratorRef', reuse=reuse):
        g1 = ml.lrelu(ml.conv(x,32,3,1,init=init,name='gconv1'))
        g2 = ml.lrelu(ml.conv(g1,64,3,1,init=init,name='gconv2'))
        g3 = ml.lrelu(ml.conv(g2,128,3,1,init=init,name='gconv3'))
        g4 = ml.tanh(ml.conv(g3,3,3,1,init=init,name='gconv4'))
        return g4

def Drefine(x,prob,reuse=False):
    init = tf.random_normal_initializer(stddev=0.02)
    keep_prob = prob
    with tf.variable_scope('DiscriminatorRef', reuse=reuse):
        d = ml.lrelu(ml.conv(x,32,3,2, init=init, name='dconv1'))
        d = ml.lrelu(ml.conv(d,64,3,2, init=init, name='dconv2'))
        d = ml.lrelu(ml.conv(d,128,3,2,init=init, name='dconv3'))
        d = ml.lrelu(ml.conv(d,128,3,2,init=init, name='dconv4'))
        d = ml.lrelu(ml.conv(d,128,3,2,init=init, name='dconv5'))
        # d = ml.leaky(ml.conv(d,256,3,1,name='dconv5'))
        # d = ml.leaky(ml.conv(d,256,3,2,name='dconv6'))
        # d = ml.leaky(ml.conv(d,512,3,2,name='dconv7'))
        # d = ml.leaky(ml.conv(d,512,3,2,name='dconv8'))
        d = ml.fullConn(d, 1, name='dconv6')
        return d, tf.reduce_mean(ml.sigmoid(d))


def Gs3Residual(x,ksz=5,s=2,prob=None,reuse=False):
    init = tf.random_normal_initializer(stddev=0.02)
    with tf.variable_scope('Generator'):
        enc_h = ml.convBNLrelu(x,64,7,2,'same', init=init, name='g_enc1')#(64,256) 320
        enc_m = ml.convBNLrelu(enc_h,128,5,s,'same', init=init, name='g_enc2')#(32,128) 160
        enc1 = ml.convBNLrelu(enc_m,256,5,s,'same', init=init, name='g_enc3')#(16,64) 80
        enc2 = ml.convBNLrelu(enc1,256,5,s,'same', init=init, name='g_enc4')#(8,32) 40
        enc3 = ml.convBNLrelu(enc2,256,5,s,'same', init=init, name='g_enc5')#(4,16) 20
        enc4 = ml.convBNLrelu(enc3,512,5,s,'same', init=init, name='g_enc6')#(2,8) 10
        enc5 = ml.convBNLrelu(enc4,512,5,s,'same', init=init, name='g_enc7')#(1,4) 5
        # Deconv
        dec1 = ml.deconvBNLrelu(enc5,512,5,s,'same', init=init, name='g_dec1')#(2,8) 10
        dec2 = ml.deconvBNLrelu(dstack(dec1,enc4),512,5,s, 'same', init=init, name='g_dec2')#(4,16) 20
        dec3 = ml.deconvBNLrelu(dstack(dec2,enc3),256,5,s, 'same', init=init, name='g_dec3')#(8,32) 40
        dec4 = ml.deconvBNLrelu(dstack(dec3,enc2),256,5,s, 'same', init=init, name='g_dec4')#(16,64) 80
        dec5 = ml.deconvBNLrelu(dstack(dec4,enc1),128,5,s, 'same', init=init, name='g_dec5')#(32,128) 160
        dec6 = ml.deconvBNLrelu(dstack(dec5,enc_m),64,5,s, 'same', init=init, name='g_dec6')#(64,128) 320

        # Output small 1x1 conv
        dec_s = deconv2d(dec5,3,1,1,'same', kernel_initializer=init, name='im_s')
        dec_s_up = tf.image.resize_bilinear(dec_s, [64,320], name='im_s_up')

        # Output med 1x1 conv
        dec_m = deconv2d(dec6,3,1,1,'same', kernel_initializer=init, name='im_m_res')
        dec_m = tf.add(dec_s_up, dec_m, name='im_m')
        dec_m_up = tf.image.resize_bilinear(dec_m, [128,640], name='im_m_up')

        # Output high
        dec_h = deconv2d(dstack(dec6,enc_h),3,7,s,'same',kernel_initializer=init, name='im_h_res')#(128,256)
        dec_h = tf.add(dec_m_up, dec_h, name='im_h')

        return tanh(dec_s), tanh(dec_m), tanh(dec_h)

def Gvertical(x, xs, xm, reuse=False):
    init = tf.random_normal_initializer(stddev=0.02)
    with tf.variable_scope('GeneratorV', reuse=reuse):
        e1 = ml.convBNLrelu(x,64,7,2,'same', init=init, name='g_enc1') # 192, 64
        e2 = ml.convBNLrelu(e1,128,7,2,'same', init=init, name='g_enc2') # 96, 32
        e3 = ml.convBNLrelu(e2,256,5,2,'same', init=init, name='g_enc3') # 48, 16
        e4 = ml.convBNLrelu(e3,256,5,2,'same', init=init, name='g_enc4') # 24, 8
        e5 = ml.convBNLrelu(e4,512,5,2,'same', init=init, name='g_enc5') # 12, 4
        e6 = ml.convBNLrelu(e5,512,5,2,'same', init=init, name='g_enc6') # 6, 2

        d1 = ml.deconvBNLrelu(e6,512,5,2,'same', init=init, name='g_dec1') # 12,4
        d2 = ml.deconvBNLrelu(dstack(d1,e5),512,5,2,'same', init=init, name='g_dec2') #24,8
        d3 = ml.deconvBNLrelu(dstack(d2,e4),256,5,2,'same', init=init, name='g_dec3') #48,16
        d4 = ml.deconvBNLrelu(dstack(d3,e3),256,5,2,'same', init=init, name='g_dec4') #96,32
        d5 = ml.deconvBNLrelu(dstack(d4,e2),128,5,2,'same', init=init, name='g_dec5') #192,64
        d6 = ml.deconvBNLrelu(dstack(d5,e1),64,7,2,'same', init=init, name='g_dec6')

        # small part
        # small 1
        ds = ml.deconv(d4,3,1,1,'same',init=init, name='imv_s')
        ds_big = ds[:,:,0:96,:]
        ds_small = ds[:,:,96:128,:]
        ds_ = tf.concat((ds_small,ds_big),axis=2)
        # small 2
        ds2 = ml.deconv(ds_,3,3,1,'same', name='imv_s2')
        ds2_small = ds2[:,:,0:32,:]
        ds2_big = ds2[:,:,32:128,:]
        ds2_ = tf.concat((ds2_big, ds2_small), axis=2)

        ds_comb = tf.add(ds,ds2_,name='imv_s_comb')
        ds_comb = tf.multiply(ds_comb,0.5,name='imv_s_out')
        ds_up = tf.image.resize_bilinear(ds_comb, [192,256], name='imv_s_up') #192

        # medium part
        # medium 1
        dm = ml.deconv(d5,3,1,1,'same', init=init, name='imv_m_res')
        dm_big = dm[:,:,0:192,:]
        dm_small = dm[:,:,192:256,:]
        dm_ = tf.concat((dm_small, dm_big),axis=2)
        # medium 2
        dm2 = ml.deconv(dm_,3,3,1,'same', name='imv_m_res2')
        dm2_small= dm2[:,:,0:64,:]
        dm2_big = dm2[:,:,64:256,:]
        dm2_ = tf.concat((dm2_big, dm2_small),axis=2)

        dm_comb = tf.add(dm,dm2_,name='imv_m_comb')
        dm_comb = tf.multiply(dm_comb,0.5, name='imv_m_out_res')
        dm_comb = tf.add(dm_comb, ds_up, name='imv_m_out')
        dm_up = tf.image.resize_bilinear(dm_comb, [384,512], name='imv_m_up')

        # high part
        # high 1
        dh = ml.deconv(d6,3,1,1,'same', init=init, name='imv_h_res')
        dh_big = dh[:,:,0:384,:]
        dh_small = dh[:,:,384:512,:]
        dh_ = tf.concat((dh_small,dh_big),axis=2)
        # high 2
        dh2 = ml.deconv(dh_,3,3,1,'same',name='imv_h_res2')
        dh2_small = dh2[:,:,0:128,:]
        dh2_big = dh2[:,:,128:512,:]
        dh2_ = tf.concat((dh2_big,dh2_small),axis=2)

        dh_comb = tf.add(dh,dh2_,name='imv_h_comb')
        dh_comb = tf.multiply(dh_comb,0.5,name='imv_h_out_res')
        dh_comb = tf.add(dh_comb, dm_up, name='imv_h_out')

        # dtop = d4[:,0:32,:,:]
        # dmid = d4[:,32:64,:,:]
        # dbot = d4[:,64:96,:,:]
        #
        #
        # d_1 = tf.concat((util.rot180(dtop),dmid,util.rot180(dbot)),1)
        # d_1 = (ml.deconv(d_1,3,3,1,'same', init=init, name='g_dec6'))
        #
        # d_2 = tf.concat((util.rot90(dtop),dmid,util.rot270(dbot)),1)
        # d_2 = (ml.deconv(d_2,3,3,1,'same', init=init, name='g_dec7'))
        #
        # d_3 = tf.concat((dtop,dmid,dbot),1)
        # d_3 = (ml.deconv(d_3,3,3,1,'same', init=init, name='g_dec8'))
        #
        # d_4 = tf.concat((util.rot270(dtop),dmid,util.rot90(dbot)),1)
        # d_4 = (ml.deconv(d_4,3,3,1,'same', init=init, name='g_dec9'))
        #
        # d_cat = tf.concat((d_1,d_2,d_3,d_4), axis=-1) + x
        #d_cat = ml.deconv(d_cat,12,1,1,'same',init=init,name='g_dec10')

        return ml.tanh(ds_comb), ml.tanh(dm_comb), ml.tanh(dh_comb)

def Gvertical2(x, reuse=False):
    init = tf.random_normal_initializer(stddev=0.02)
    with tf.variable_scope('GeneratorV', reuse=reuse):
        # Split input
        x1 = x[:,:,:,0:3]
        x2 = x[:,:,:,3:6]
        x3 = x[:,:,:,6:9]
        x4 = x[:,:,:,9:12]
        x5 = x[:,:,:,12:15]

        e1 = ml.convBNLrelu(x1,16,3,2,'same', init=init, name='x1_enc1')
        e2 = ml.convBNLrelu(x2,16,3,2,'same', init=init, name='x2_enc1')
        e3 = ml.convBNLrelu(x3,16,3,2,'same', init=init, name='x3_enc1')
        e4 = ml.convBNLrelu(x4,16,3,2,'same', init=init, name='x4_enc1')
        e5 = ml.convBNLrelu(x5,16,3,2,'same', init=init, name='x5_enc1')


def DverticalS(x,reuse=False):
    init = tf.random_normal_initializer(stddev=0.02)
    with tf.variable_scope('DiscriminatorVS', reuse=reuse):
        d = ml.convLrelu(x,32,3,2,'same', init=init, name='d_enc1')
        d = ml.convLrelu(d,64,3,2,'same', init=init, name='d_enc2')
        d = ml.convLrelu(d,64,3,2,'same', init=init, name='d_enc3')
        d = ml.convLrelu(d,64,3,2,'same', init=init, name='d_enc4')
        d = ml.dropout(d, 0.5, name='drop1')
        d = ml.fullConn(d,1,init=init, name='d_fc1')
        return d, tf.reduce_mean(ml.sigmoid(d))

def DverticalM(x,reuse=False):
    init = tf.random_normal_initializer(stddev=0.02)
    with tf.variable_scope('DiscriminatorVM', reuse=reuse):
        d = ml.convLrelu(x,32,3,2,'same', init=init, name='d_enc1')
        d = ml.convLrelu(d,64,3,2,'same', init=init, name='d_enc2')
        d = ml.convLrelu(d,64,3,2,'same', init=init, name='d_enc3')
        d = ml.convLrelu(d,128,3,2,'same', init=init, name='d_enc4')
        d = ml.convLrelu(d,128,3,2,'same', init=init, name='d_enc5')
        d = ml.dropout(d, 0.5, name='drop1')
        d = ml.fullConn(d,1,init=init, name='d_fc1')
        return d, tf.reduce_mean(ml.sigmoid(d))

def DverticalH(x, reuse=False):
    init = tf.random_normal_initializer(stddev=0.02)
    with tf.variable_scope('DiscriminatorVH', reuse=reuse):
        d = ml.convLrelu(x,32,3,2,'same', init=init, name='d_enc1')
        d = ml.convLrelu(d,64,3,2,'same', init=init, name='d_enc2')
        d = ml.convLrelu(d,64,3,2,'same', init=init, name='d_enc3')
        d = ml.convLrelu(d,128,3,2,'same', init=init, name='d_enc4')
        d = ml.convLrelu(d,128,3,2,'same', init=init, name='d_enc5')
        d = ml.convLrelu(d,256,3,2,'same', init=init, name='d_enc6')
        d = ml.dropout(d, 0.5, name='drop1')
        d = ml.fullConn(d,1,init=init, name='d_fc1')
        return d, tf.reduce_mean(ml.sigmoid(d))




def GfromNoise(x,init='random_normal', reuse=False):
    with tf.variable_scope('GeneratorNoise', reuse=reuse):
        leaky = tf.nn.leaky_relu
        g = tf.layers.dense(x, 1*4*512)
        g = tf.reshape(g,[-1,1,4,512])
        g = tf.layers.batch_normalization(g)
        g = leaky(g)

        g = tf.layers.conv2d_transpose(g, 256, 5, 2, 'same',name='conv1')#(2,8)
        g = tf.layers.batch_normalization(g)
        g = leaky(g)

        g = tf.layers.conv2d_transpose(g, 128, 5, 2,'same',name='conv2')#(4,16)
        g = tf.layers.batch_normalization(g)
        g = leaky(g)

        g = tf.layers.conv2d_transpose(g, 64, 5, 2,'same',name='conv3')#(4,16)
        g = tf.layers.batch_normalization(g)
        g = leaky(g)

        g = tf.layers.conv2d_transpose(g, 3, 5, 2, 'same',name='conv4')#(16,64)
        g = tf.nn.tanh(g)
        print(g.get_shape())

        return g

def DfromNoise(x, init='random_normal', reuse=False):
    with tf.variable_scope('DiscriminatorNoise',reuse=reuse):
        leaky = tf.nn.leaky_relu
        d = tf.layers.conv2d(x, 32, 3, 2, padding='same', name='d1')
        d = tf.layers.batch_normalization(d)
        d = leaky(d)
        d = tf.layers.dropout(d)

        d = tf.layers.conv2d(d, 64, 3, 2, padding='same', name='d2')
        d = tf.layers.batch_normalization(d)
        d = leaky(d)
        d = tf.layers.dropout(d)

        d = tf.layers.conv2d(d, 128, 3, 2, padding='same', name='d3')
        d = tf.layers.batch_normalization(d)
        d = leaky(d)
        d = tf.layers.dropout(d)

        d = tf.layers.conv2d(d, 128, 3, 2, padding='same', name='d4')
        d = tf.layers.batch_normalization(d)
        d = leaky(d)
        d = tf.layers.dropout(d)

        d = tf.layers.flatten(d)
        d = tf.layers.dense(d,1,name='d5')
        return d, tf.reduce_mean(ml.sigmoid(d))

def fullPano(x,ksz=3,s=2,init='he_uniform',reuse=False):
    with tf.variable_scope('360Pano'):
        enc1 = ml.leaky(ml.convLyr(x,32,ksz,s,init=init,name='pano_enc1'))#(64,256)
        enc2 = ml.leaky(ml.convLyr(enc1,64,ksz,s,init=init,name='pano_enc2'))#(32,128)
        enc3 = ml.leaky(ml.convLyr(enc2,128,ksz,s,init=init,name='pano_enc3'))#(16,64)
        enc4 = ml.leaky(ml.convLyr(enc3,128,ksz,s,init=init,name='pano_enc4'))#(8,32)
        enc5 = ml.leaky(ml.convLyr(enc4,256,ksz,s,init=init,name='pano_enc5'))#(4,16)
        enc6 = ml.leaky(ml.convLyr(enc5,512,ksz,s,init=init,name='pano_enc6'))#(2,8)
        dec1 = ml.leaky(ml.deconvLyr(enc6,512,ksz,s,init=init,name='pano_dec1'))#(4,16)
        dec2 = ml.leaky(ml.deconvLyr(dstack(dec1,enc5),256,ksz,s,init=init,name='pano_dec2'))#(8,32)
        dec3 = ml.leaky(ml.deconvLyr(dstack(dec2,enc4),128,ksz,s,init=init,name='pano_dec3'))#(16,64)
        dec4 = ml.leaky(ml.deconvLyr(dstack(dec3,enc3),128,ksz,s,init=init,name='pano_dec4'))#(32,128)
        dec5 = ml.leaky(ml.deconvLyr(dstack(dec4,enc2),64,ksz,s,init=init,name='pano_dec5'))#(64,256)
        dec6 = ml.deconvLyr(dstack(dec5,enc1),2,ksz,s,init=init,name='pano_dec6')#(128,512)
        dec7 = bs.bilinearSampler(x,dec6)
        return dec7

def getModelVars(var_scope):
    vars_dict = {}
    vars_to_save = [v for v in tf.global_variables() if var_scope in v.name ]
    for var in vars_to_save:
        vars_dict[var.name[:-2]] = var
    return vars_dict

def dLossGan(dy,dg,lbl_real,lbl_fake):
    loss_real = tf.reduce_mean(sigmoidEnt(logits=dy, labels=lbl_real))
    loss_fake = tf.reduce_mean(sigmoidEnt(logits=dg, labels=lbl_fake))
    loss_d = loss_real + loss_fake
    return loss_real, loss_fake, loss_d

def gLossGanMask(g,dg,y,mask,lbl_real,a1,a2):
    loss_adv = tf.reduce_mean(sigmoidEnt(logits=dg, labels=lbl_real))
    loss_hole = tf.reduce_mean(tf.abs((1-mask) * (y-g)))
    loss_valid = tf.reduce_mean(tf.abs(mask * (y-g)))
    loss_g = loss_adv + a1 * loss_hole + a2 * loss_valid
    return loss_adv, loss_hole, loss_valid, loss_g

def gLossGanRes(g,gd,dg,mask,y,lbl_real,a1,a2):
    loss_adv = tf.reduce_mean(sigmoidEnt(logits=dg, labels=lbl_real))
    loss_valid = tf.reduce_mean(tf.abs(mask * (y-g)))
    loss_res = tf.reduce_mean(tf.abs(y-gd))
    loss_g = loss_adv + a2 * loss_valid
    return loss_adv, loss_valid, loss_res, loss_g

def gLossGan(g,dg,y,lbl_real,a1,a2):
    loss_adv = tf.reduce_mean(sigmoidEnt(logits=dg, labels=lbl_real))
    loss_valid = tf.reduce_mean(tf.abs( (y-g)))
    loss_g = loss_adv +  a1 * loss_valid
    return loss_adv, loss_valid, loss_valid, loss_g

def panoLoss(g_pano: tf.Tensor, y: tf.Tensor):
    loss_pano = tf.reduce_mean(tf.abs(y-g_pano))
    return loss_pano

def interpolate(a,b):
    shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
    alpha = tf.random_uniform(shape=shape, minval=0.0, maxval=1.0)
    inter = a + alpha * (b - a)
    inter.set_shape(a.get_shape().as_list())
    return inter

def gradPenalty(g, y, batch_sz, d_func):
    epsilon = tf.random_uniform(shape=[batch_sz,1,1,1],minval=0.0,maxval=1.0)
    x_hat = y + epsilon * (g - y)
    D_x_hat = d_func(x_hat, reuse=True)
    grad_D_x_hat = tf.gradients(D_x_hat, [x_hat])[0]
    red_idx = tf.range(1, x_hat.shape.ndims)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_x_hat), reduction_indices=red_idx))
    gp = tf.reduce_mean((slopes - 1)**2)
    return gp

def lossGP(g, y, batch_sz):
    epsilon = tf.random_uniform(shape=[batch_sz,1,1,1],minval=0.0,maxval=1.0)
    x_diff = y + epsilon * (g - y)
    return x_diff

def consistentG(g):
    with tf.variable_scope('Consistency'):
        g_const = util.catConst(g)
        conv = ml.convLrelu(g_const,16,3,1,name='g_conv1')
        conv = ml.convLrelu(conv,32,3,1,name='g_conv2')
        conv = ml.convLrelu(conv,3,3,1,name='g_conv3')
        return ml.tanh(conv)

def gWGANLoss(g, y, dg, a1, a2, mask):
    loss_adv = -tf.reduce_mean(dg)
    #loss_hole = tf.reduce_mean(tf.abs(1-mask) * (y-g))
    #loss_valid = tf.reduce_mean(tf.abs( mask * (y-g)))
    #loss_g = loss_adv + a1 * loss_hole + a2 * loss_valid
    loss_hole = 0
    loss_valid = 0
    loss_g = loss_adv
    return loss_adv, loss_hole, loss_valid, loss_g

def dWGANLoss(dg, dy, g, y, batch_sz, Dfunc, a1):
    loss_real = tf.reduce_mean(dy)
    loss_fake = tf.reduce_mean(dg)
    loss_d = loss_fake - loss_real

    gp = gradPenalty(g, y, batch_sz, Dfunc)
    loss_d += a1 * gp

    return loss_real, loss_fake, loss_d

# Binary cross entropy loss
def bceLoss(logits,labels,smoothing=0.0):
    labels = labels + smoothing
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss

def l1Loss(ypred,y,mask=1):
    loss = tf.reduce_mean(tf.abs((mask) * (ypred-y)))
    return loss

def constLoss(ypred, y, im_type='high'):
    if im_type == 'small':
        # ypred = util.constToNormBlend(ypred)
        # y = util.constToNormBlend(y)
        ypred1 = tf.image.crop_to_bounding_box(ypred,0,0,32,32)
        ypred2 = tf.image.crop_to_bounding_box(ypred,0,127,32,32)
    elif im_type == 'medium':
        y1 = tf.image.crop_to_bounding_box(ypred,0,0,64,64)
        y2 = tf.image.crop_to_bounding_box(ypred,0,255,64,64)
    else:
        y1 = tf.image.crop_to_bounding_box(ypred,0,0,128,128)
        y2 = tf.image.crop_to_bounding_box(ypred,0,511,128,128)

    return l1Loss(ypred1, ypred2)

def vertConstLoss(ypred, y, net_type='small'):
    if net_type == 'small':
        yp_top = ypred[:,0:32,:,:]
        yp_bot = ypred[:,64:96,:,:]

        yt1 = l1Loss(yp_top[:,:,32:64,:], util.rot180(yp_top[:,:,96:128,:]))
        yt2 = l1Loss(yp_top[:,:,32:64,:], util.rot90(yp_top[:,:,0:32,:]))
        yt3 = l1Loss(yp_top[:,:,32:64,:], util.rot270(yp_top[:,:,64:96,:]))

        yb1 = l1Loss(yp_bot[:,:,32:64,:], util.rot180(yp_bot[:,:,96:128,:]))
        yb2 = l1Loss(yp_bot[:,:,32:64,:], util.rot270(yp_bot[:,:,0:32,:]))
        yb3 = l1Loss(yp_bot[:,:,32:64,:], util.rot90(yp_bot[:,:,64:96,:]))

    if net_type == 'medium':
        yp_top = ypred[:,0:64,:,:]
        yp_bot = ypred[:,128:192,:,:]

        yt1 = l1Loss(yp_top[:,:,64:128,:], util.rot180(yp_top[:,:,192:256,:]))
        yt2 = l1Loss(yp_top[:,:,64:128,:], util.rot90(yp_top[:,:,0:64,:]))
        yt3 = l1Loss(yp_top[:,:,64:128,:], util.rot270(yp_top[:,:,128:192,:]))

        yb1 = l1Loss(yp_bot[:,:,64:128,:], util.rot180(yp_bot[:,:,192:256,:]))
        yb2 = l1Loss(yp_bot[:,:,64:128,:], util.rot270(yp_bot[:,:,0:64,:]))
        yb3 = l1Loss(yp_bot[:,:,64:128,:], util.rot90(yp_bot[:,:,128:192,:]))

    if net_type == 'high':
        yp_top = ypred[:,0:128,:,:]
        yp_mid = ypred[:,128:256,:,:]
        yp_bot = ypred[:,256:384,:,:]

        yt1 = l1Loss(yp_top[:,:,128:256,:], util.rot180(yp_top[:,:,384:512,:]))
        yt2 = l1Loss(yp_top[:,:,128:256,:], util.rot270(yp_top[:,:,0:128,:]))
        yt3 = l1Loss(yp_top[:,:,128:256,:], util.rot90(yp_top[:,:,256:384,:]))

        yb1 = l1Loss(yp_bot[:,:,128:256,:], util.rot180(yp_bot[:,:,384:512,:]))
        yb2 = l1Loss(yp_bot[:,:,128:256,:], util.rot90(yp_bot[:,:,0:128,:]))
        yb3 = l1Loss(yp_bot[:,:,128:256,:], util.rot270(yp_bot[:,:,256:384,:]))
        # ypred = util.makeCubePano(ypred)
        # y = util.makeCubePano(y)
        # loss0 = l1Loss(ypred, y)
        loss = yt1 + yt2 + yt3 + yb1 + yb2 + yb3


    return loss
