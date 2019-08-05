import random, time
import tensorflow as tf
import model
import util
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class Network:
    def __init__(self, dataset_path, batch_sz, shuffle_buff):
        self.x = None
        self.y = None
        self.fov_gt = None
        self.g = None
        self.dy = None
        self.dg = None
        self.x_pred = None
        self.fov_pred = None
        self.mask_pred = None
        self.iterator = None
        self.epochs = 300000
        self.batch_sz =batch_sz
        self.shuffle_buff = shuffle_buff
        self.print_range = 200
        self.save_range = 10000
        self.sess = None
        self.dataset_path = dataset_path
        self.output_folder ='/home/juliussurya/workspace/360pano/output'

    def initDataset(self):
        self.iterator = util.initDataset(self.dataset_path, self.batch_sz, self.shuffle_buff)
        self.getDataset = self.iterator.get_next() # ** RUN THIS IN SESSION **

        # placeholder
        self.x_ = tf.placeholder(tf.float32,[None,256,256,18], name='x_in')
        self.y_ = tf.placeholder(tf.float32,[None,256,256,18], name='y_in')
        self.fov_ = tf.placeholder(tf.float32,[None,128], name='fov_in')
        self.pano_ = tf.placeholder(tf.float32, [None,512,1024,3], name='pano_in')


        # Process input
        self.fov_gt = self.fov_
        self.x = util.catResize(self.x_, name='x_proc') # concat and resize
        self.y = util.catResizeV2(self.y_, name='y_proc') # concat and resize
        self.ym = tf.image.resize_bilinear(self.y,[64,320], name='ym_proc')
        self.ys = tf.image.resize_bilinear(self.y,[32,160], name='ys_proc')
        return self.x, self.y, self.fov_gt, self.pano_

    def initDatasetFOV(self):
        self.iterator = util.initDatasetNew(self.dataset_path, self.batch_sz, self.shuffle_buff)
        self.getDataset = self.iterator.get_next()

        self.x_ = tf.placeholder(tf.float32,[None, 128, 512, 3], name='x_in')
        self.fov_single = tf.placeholder(tf.int64, [None], name='fov_in')
        self.fov_gt = tf.one_hot(self.fov_single, 128, axis=-1)

    # ------ BUILD MODEL ------ #
    def buildModelFOV(self):
        self.fov_pred, self.fov_scalar = model.fovNet(self.x_)
        # self.x = util.procInputFovV2(self.x,fov_scalar, name='x_pred')
        # self.xm = tf.image.resize_bilinear(self.x, [64,320], name='xm_pred')
        # self.xs = tf.image.resize_bilinear(self.x, [32,160], name='xs_pred')

        # loss
        softmaxEnt = tf.nn.softmax_cross_entropy_with_logits_v2
        self.loss_fov = tf.reduce_mean(softmaxEnt(logits=self.fov_pred, labels=self.fov_gt),
                                       name='loss_fov')

        gs_vars = [v for v in tf.trainable_variables()
                   if v.name.startswith('Fov')]
        print(gs_vars)

        adamOpt = tf.train.AdamOptimizer
        self.setLearningRate(lr_s=[0.0003, 0.0003])
        self.opt_fov = adamOpt(self.lr_fov).minimize(self.loss_fov, global_step=self.step_fov)
        tf.summary.scalar('fov', self.loss_fov)

    def buildModelSmall(self):
        # Function redefinition
        adamOpt = tf.train.AdamOptimizer

        # D real
        self.ds_real, self.dsr = model.Ds(model.dstack(self.xs, self.ys))
        self.loss_ds_real = model.bceLoss(self.ds_real, tf.ones_like(self.ds_real))

        # Generator and D fake
        self.gs, self.gm, self.g = model.Gs3Residual(self.x)
        self.ds_fake, self.dsf = model.Ds(model.dstack(self.xs, self.gs),reuse=True)
        self.loss_gs_adv = model.bceLoss(self.ds_fake, tf.ones_like(self.ds_fake))
        self.loss_gs_valid = model.l1Loss(self.gs, self.ys)
        self.loss_gs_hole = 0
        self.loss_gs_const = model.constLoss(self.gs, self.ys, 'small')
        self.loss_ds_fake = model.bceLoss(self.ds_fake, tf.zeros_like(self.ds_fake))

        # Total Loss
        self.loss_ds = self.loss_ds_real + self.loss_ds_fake
        self.loss_gs = self.loss_gs_adv + 100 * self.loss_gs_valid #+ 0.01 * self.loss_gs_const

        # Optimize
        self.setLearningRate(lr_s=[0.0003, 0.0003])

        # FOV
        self.opt_fov = adamOpt(self.lr_fov).minimize(self.loss_fov, global_step=self.step_fov)

        # GAN
        gs_vars = [v for v in tf.trainable_variables()
                  if v.name.startswith('Generator/g_') or v.name.startswith('Generator/im_s')]
        ds_vars = [v for v in tf.trainable_variables() if v.name.startswith('DiscriminatorS')]

        with tf.variable_scope('OptimizerS'):
            self.opt_ds = adamOpt(self.lr_ds,beta1=0.5).minimize(self.loss_ds,
                                                                 global_step=self.step_s,
                                                                 var_list=ds_vars)
            self.opt_gs = adamOpt(self.lr_gs,beta1=0.5).minimize(self.loss_gs,
                                                                 global_step=self.step_s,
                                                                 var_list=gs_vars)

        self.gs_crop = util.constToNorm(self.gs,[0,15,32,128])
        # Add graph summary
        self.mergeSummarySmall()

    def buildModelPreMed(self):
        self.iterator = util.initDataset(self.dataset_path, self.batch_sz, self.shuffle_buff)
        self.getDataset = self.iterator.get_next() # ** RUN THIS IN SESSION **

        # Network
        # Generator
        self.x_ = self.graph.get_tensor_by_name('x_in:0')
        self.y_ = self.graph.get_tensor_by_name('y_in:0')
        self.fov_ = self.graph.get_tensor_by_name('fov_in:0')
        self.pano_ = self.graph.get_tensor_by_name('pano_in:0')
        self.noise_ = self.graph.get_tensor_by_name('rand_noise:0')
        self.prob_ = self.graph.get_tensor_by_name('keep_prob:0')
        self.x = self.graph.get_tensor_by_name('x_pred:0')
        xm = tf.image.resize_bilinear(self.x, [64,256])
        self.mask_m = self.graph.get_tensor_by_name('mask_m:0')
        self.ym = self.graph.get_tensor_by_name('ym_proc:0')

        # D real
        self.dm_real, self.dmr  = model.Dm(model.dstack(xm,self.ym),prob=self.prob_)
        self.loss_dm_real = model.bceLoss(self.dm_real, tf.ones_like(self.dm_real))

        # G and D fake
        gm_up = self.graph.get_tensor_by_name('Generator/im_s_up:0')
        self.gm = self.graph.get_tensor_by_name('Generator/im_m:0')
        gm_res = tf.subtract(self.gm, gm_up, name='Generator/gm_d_res')
        self.dm_fake, self.dmf = model.Dm(model.dstack(xm,self.gm),prob=self.prob_,reuse=True)
        self.loss_gm_adv = model.bceLoss(self.dm_fake, tf.ones_like(self.dm_fake))
        self.loss_gm_valid = model.l1Loss(self.gm, self.ym)
        self.loss_gm_hole = 0
        self.loss_dm_fake = model.bceLoss(self.dm_fake, tf.zeros_like(self.dm_fake))

        # Total Loss
        self.loss_dm = self.loss_dm_real + self.loss_dm_fake
        self.loss_gm = self.loss_gm_adv + 100 * self.loss_gm_valid


        # Optimize
        self.setLearningRate(lr_m=[0.0002, 0.0002])
        adamOpt = tf.train.AdamOptimizer

        dm_var = [v for v in tf.trainable_variables() if v.name.startswith('DiscriminatorM')]
        gm_var = [v for v in tf.trainable_variables() if v.name.startswith('Generator/g_')
                  or v.name.startswith('Generator/im_s') or v.name.startswith('Generator/im_m')]
        print(gm_var)
        with tf.name_scope('OptimizerM'):
            self.opt_dm = adamOpt(self.lr_dm, beta1=0.5).minimize(self.loss_dm, global_step=self.step_m,
                                                                  var_list=dm_var)
            self.opt_gm = adamOpt(self.lr_gm, beta1=0.5).minimize(self.loss_gm, global_step=self.step_m,
                                                                  var_list=gm_var)

        self.mergeSummaryMed()

    def buildModelPreHigh(self):
        self.iterator = util.initDataset(self.dataset_path, self.batch_sz, self.shuffle_buff)
        self.getDataset = self.iterator.get_next() # ** RUN THIS IN SESSION **

        # Generator
        self.x_ = self.graph.get_tensor_by_name('x_in:0')
        self.y_ = self.graph.get_tensor_by_name('y_in:0')
        self.fov_ = self.graph.get_tensor_by_name('fov_in:0')
        self.pano_ = self.graph.get_tensor_by_name('pano_in:0')
        self.mask = self.graph.get_tensor_by_name('mask:0')
        self.noise_ = self.graph.get_tensor_by_name('rand_noise:0')
        self.prob_ = self.graph.get_tensor_by_name('keep_prob:0')
        self.x = self.graph.get_tensor_by_name('x_pred:0')
        self.y = self.graph.get_tensor_by_name('y_proc:0')

        # D real
        self.d_real, self.dr = model.D(model.dstack(self.x, self.y),prob=self.prob_) # Dreal
        self.loss_d_real = model.bceLoss(self.d_real, tf.ones_like(self.d_real))

        # G and D fake
        g_up = self.graph.get_tensor_by_name('Generator/im_m_up:0')
        self.g = self.graph.get_tensor_by_name('Generator/im_h:0')
        self.g = tf.nn.tanh(self.g)
        g_res = tf.subtract(self.g, g_up, name='Generator/g_res') #G
        # self.g_const = model.consistentG(self.g)
        # self.y_const = util.catConst(self.y)
        # self.g = util.constToNorm(self.g_const)

        self.d_fake, self.df = model.D(model.dstack(self.x, self.g),prob=self.prob_, reuse=True) #Dfake
        self.loss_g_adv = model.bceLoss(self.d_fake, tf.ones_like(self.d_fake))
        self.loss_g_valid = model.l1Loss(self.g, self.y)
        self.loss_g_hole = 0
        self.loss_g_const = model.l1Loss(self.g, self.y)
        self.loss_d_fake = model.bceLoss(self.d_fake, tf.zeros_like(self.d_fake))

        # Total Loss
        self.loss_d = self.loss_d_real + self.loss_d_fake
        self.loss_g = self.loss_g_adv +  (100 *  self.loss_g_valid)


        # Optimize
        self.setLearningRate(lr_h=[0.0003, 0.0003])
        adamOpt = tf.train.AdamOptimizer

        d_var = [v for v in tf.trainable_variables() if v.name.startswith('DiscriminatorH')]
        g_var = [v for v in tf.trainable_variables() if v.name.startswith('Generator')]
        print(g_var)
        with tf.variable_scope('OptimizerH'):
            self.opt_d = adamOpt(self.lr_d, beta1=0.5).minimize(self.loss_d, global_step=self.step,
                                                                var_list=d_var)
            self.opt_g = adamOpt(self.lr_g, beta1=0.5).minimize(self.loss_g, global_step=self.step,
                                                                var_list=g_var)

        self.mergeSummaryHigh()

    def buildVertModel(self, model_type):
        self.iterator = util.initDataset(self.dataset_path, self.batch_sz, self.shuffle_buff)
        self.getDataset = self.iterator.get_next()

        self.x_ = self.graph.get_tensor_by_name('x_in:0')
        self.y_ = self.graph.get_tensor_by_name('y_in:0')
        self.fov_ = self.graph.get_tensor_by_name('fov_in:0')
        self.pano_ = self.graph.get_tensor_by_name('pano_in:0')
        self.noise_ = self.graph.get_tensor_by_name('rand_noise:0')
        self.prob_ = self.graph.get_tensor_by_name('keep_prob:0')

        self.g = self.graph.get_tensor_by_name('Generator/im_h:0')
        self.g = tf.nn.tanh(self.g)
        self.yv = util.catVerticalResizeV2(self.y_) # Make y vert
        self.yv_s = util.catVerticalResizeV2(self.y_, [96,32]) # small yvert
        self.yv_m = util.catVerticalResizeV2(self.y_, [192,64]) # med yvert
        self.gv_in = util.dstackPano(self.g) # Make x vert
        self.gv_in_s = util.dstackPano(self.g, size=[32,32]) # small x vert
        self.gv_in_m = util.dstackPano(self.g, size=[64,64]) # med x vert


        adamOpt = tf.train.AdamOptimizer

        if model_type == 'small':
            # Loss Real
            self.dv_real, self.dv_r = model.DverticalS(model.dstack(self.gv_in_s, self.yv_s))
            self.loss_dv_real = model.bceLoss(self.dv_real, labels=tf.ones_like(self.dv_real))

            # Loss G and Fake
            self.gv_s,_,_ = model.Gvertical(self.gv_in, self.gv_in_s, self.gv_in_m)
            self.dv_fake, self.dv_f = model.DverticalS(model.dstack(self.gv_in_s, self.gv_s), reuse=True)

            # Loss G
            self.loss_gv_adv = model.bceLoss(self.dv_fake, tf.ones_like(self.dv_fake))
            self.loss_gv_valid = model.l1Loss(self.gv_s, self.yv_s)
            self.loss_gv_const = model.vertConstLoss(self.gv_s, self.yv_s)

            # Loss Fake
            self.loss_dv_fake = model.bceLoss(self.dv_fake, tf.zeros_like(self.dv_fake))

            # Total Loss
            self.loss_dv = self.loss_dv_real + self.loss_dv_fake
            self.loss_gv = self.loss_gv_adv + 100 * self.loss_gv_valid + self.loss_gv_const

            self.setLearningRate(lr_s=[0.0002, 0.0002])

            ds_vars = [v for v in tf.trainable_variables() if v.name.startswith('DiscriminatorVS')]
            gs_vars = [v for v in tf.trainable_variables() if v.name.startswith('GeneratorV/g_') or
                       v.name.startswith('GeneratorV/imv_s')]

            with tf.variable_scope('OptimizerVS'):
                self.opt_dv = adamOpt(self.lr_ds, beta1=0.5).minimize(self.loss_dv, var_list=ds_vars)
                self.opt_gv = adamOpt(self.lr_gs, beta1=0.5).minimize(self.loss_gv, var_list=gs_vars)

            self.mergeSummaryVertS()

        if model_type == 'medium':
            # Loss Real
            self.dv_real, self.dv_r = model.DverticalM(model.dstack(self.gv_in_m, self.yv_m))
            self.loss_dv_real = model.bceLoss(self.dv_real, labels=tf.ones_like(self.dv_real))

            # Loss G and Fake
            # self.gv_s,_,_ = model.Gvertical(self.gv_in, self.gv_in_s, self.gv_in_m)
            self.gv_m = self.graph.get_tensor_by_name('GeneratorV/imv_m_out:0')
            self.gv_m = tf.nn.tanh(self.gv_m)
            self.dv_fake, self.dv_f = model.DverticalM(model.dstack(self.gv_in_m, self.gv_m), reuse=True)

            # Loss G
            self.loss_gv_adv = model.bceLoss(self.dv_fake, tf.ones_like(self.dv_fake))
            self.loss_gv_valid = model.l1Loss(self.gv_m, self.yv_m)
            self.loss_gv_const = model.vertConstLoss(self.gv_m, self.yv_m, 'medium')

            # Loss Fake
            self.loss_dv_fake = model.bceLoss(self.dv_fake, tf.zeros_like(self.dv_fake))

            # Total Loss
            self.loss_dv = self.loss_dv_real + self.loss_dv_fake
            self.loss_gv = self.loss_gv_adv + 100 * self.loss_gv_valid + self.loss_gv_const

            self.setLearningRate(lr_s=[0.0002, 0.0002])

            ds_vars = [v for v in tf.trainable_variables() if v.name.startswith('DiscriminatorVM')]
            gs_vars = [v for v in tf.trainable_variables() if v.name.startswith('GeneratorV/g_') or
                       v.name.startswith('GeneratorV/imv_m')]

            with tf.variable_scope('OptimizerVM'):
                self.opt_dv = adamOpt(self.lr_ds, beta1=0.5).minimize(self.loss_dv, var_list=ds_vars)
                self.opt_gv = adamOpt(self.lr_gs, beta1=0.5).minimize(self.loss_gv, var_list=gs_vars)

            self.mergeSummaryVertM()

        if model_type == 'high':
            # Loss Real
            self.dv_real, self.dv_r = model.DverticalH(model.dstack(self.gv_in, self.yv))
            self.loss_dv_real = model.bceLoss(self.dv_real, labels=tf.ones_like(self.dv_real))

            # Loss G and Fake
            # self.gv_s,_,_ = model.Gvertical(self.gv_in, self.gv_in_s, self.gv_in_m)
            self.gv_h = self.graph.get_tensor_by_name('GeneratorV/imv_h_out:0')
            self.gv_h = tf.nn.tanh(self.gv_h)

            self.dv_fake, self.dv_f = model.DverticalH(model.dstack(self.gv_in, self.gv_h), reuse=True)

            # Loss G
            self.loss_gv_adv = model.bceLoss(self.dv_fake, tf.ones_like(self.dv_fake))
            self.loss_gv_valid = model.l1Loss(self.gv_h, self.yv)
            self.loss_gv_const = model.vertConstLoss(self.gv_h, self.yv, 'high')

            # Loss Fake
            self.loss_dv_fake = model.bceLoss(self.dv_fake, tf.zeros_like(self.dv_fake))

            # Total Loss
            self.loss_dv = self.loss_dv_real + self.loss_dv_fake
            # self.loss_gv = self.loss_gv_adv +  100 *  self.loss_gv_const
            self.loss_gv = self.loss_gv_adv + 100 * self.loss_gv_valid + 10 * self.loss_gv_const

            self.setLearningRate(lr_s=[0.0002, 0.0002])

            ds_vars = [v for v in tf.trainable_variables() if v.name.startswith('DiscriminatorVH')]
            gs_vars = [v for v in tf.trainable_variables() if v.name.startswith('GeneratorV')]

            with tf.variable_scope('OptimizerVH2'):
                self.opt_dv = adamOpt(self.lr_ds, beta1=0.5).minimize(self.loss_dv, var_list=ds_vars)
                self.opt_gv = adamOpt(self.lr_gs, beta1=0.5).minimize(self.loss_gv, var_list=gs_vars)

            self.mergeSummaryVertH()

    def buildVertModelSmall(self):
        self.iterator = util.initDataset(self.dataset_path, self.batch_sz, self.shuffle_buff)
        self.getDataset = self.iterator.get_next()

        self.x_ = self.graph.get_tensor_by_name('x_in:0')
        self.y_ = self.graph.get_tensor_by_name('y_in:0')
        self.fov_ = self.graph.get_tensor_by_name('fov_in:0')
        self.pano_ = self.graph.get_tensor_by_name('pano_in:0')
        self.ys = self.graph.get_tensor_by_name('ys_proc:0')

        self.n_top = tf.placeholder(tf.float32)
        self.n_bot = tf.placeholder(tf.float32)

        self.gs = self.graph.get_tensor_by_name('Generator/im_s/BiasAdd:0')
        self.gs = tf.nn.tanh(self.gs)

        self.ys_v = util.catVerticalResizeV2(self.y_, [96,32])
        g_in = util.dstackPano(self.gs, self.n_top, self.n_bot)
        # self.gs = tf.stop_gradient(self.gs)
        self.g_in = g_in
        maskv_s = util.createVmask()
        # self.ys_v = tf.image.resize_bilinear(self.y_v, [96,32])

        self.ds_v_real, self.dsr_v =  model.Dvertical(model.dstack(g_in, self.ys_v))
        self.loss_ds_v_real = model.bceLoss(self.ds_v_real, tf.ones_like(self.ds_v_real))

        self.gs_v = model.Gvertical(g_in)
        self.ds_v_fake, self.dsf_v = model.Dvertical(model.dstack(g_in, self.gs_v), reuse=True)
        self.loss_gs_v_adv = model.bceLoss(self.ds_v_fake, tf.ones_like(self.ds_v_fake))
        self.loss_gs_v_valid = model.l1Loss(self.gs_v , self.ys_v )
        # self.loss_gs_v_const = model.vertConstLoss(self.gs_v, self.ys_v)
        self.loss_ds_v_fake = model.bceLoss(self.ds_v_fake, tf.zeros_like(self.ds_v_fake))

        self.loss_ds_v = self.loss_ds_v_fake + self.loss_ds_v_real
        self.loss_gs_v = self.loss_gs_v_adv + 100 * self.loss_gs_v_valid #+ self.loss_gs_v_const

        self.setLearningRate(lr_s=[0.0002, 0.0002])

        ds_vars = [v for v in tf.trainable_variables() if v.name.startswith('DiscriminatorV')]
        gs_vars = [v for v in tf.trainable_variables() if v.name.startswith('GeneratorV')]
        adamOpt = tf.train.AdamOptimizer
        with tf.variable_scope('OptimizerVS'):
            self.opt_ds_v = adamOpt(self.lr_ds, beta1=0.5).minimize(self.loss_ds_v,
                                                                    global_step=self.step,
                                                                    var_list=ds_vars)
            self.opt_gs_v = adamOpt(self.lr_gs, beta1=0.5).minimize(self.loss_gs_v,
                                                                    global_step=self.step,
                                                                    var_list=gs_vars)
        self.mergeSummaryVertS()

    def buildTestModel(self):
        self.iterator = util.initDataset(self.dataset_path, self.batch_sz, self.shuffle_buff,False)
        self.getDataset = self.iterator.get_next() # ** RUN THIS IN SESSION **

        # Generator
        self.x_ = self.graph.get_tensor_by_name('x_in:0')
        self.y_ = self.graph.get_tensor_by_name('y_in:0')
        self.fov_ = self.graph.get_tensor_by_name('fov_in:0')
        self.pano_ = self.graph.get_tensor_by_name('pano_in:0')
        self.x_proc = self.graph.get_tensor_by_name('x_proc:0')
        self.x_pred = self.graph.get_tensor_by_name('x_pred:0')
        self.noise_ = self.graph.get_tensor_by_name('rand_noise:0')
        self.prob_ = self.graph.get_tensor_by_name('keep_prob:0')
        self.y = self.graph.get_tensor_by_name('y_proc:0')
        self.g = self.graph.get_tensor_by_name('Generator/im_h:0')

    def buildTestModelFull(self):
        self.iterator = util.initDataset(self.dataset_path, self.batch_sz, self.shuffle_buff, False)
        self.getDataset = self.iterator.get_next()

        self.x_ = self.graph.get_tensor_by_name('x_in:0')
        self.y_ = self.graph.get_tensor_by_name('y_in:0')
        self.fov_ = self.graph.get_tensor_by_name('fov_in:0')
        self.pano_ = self.graph.get_tensor_by_name('pano_in:0')
        self.x_proc = self.graph.get_tensor_by_name('x_proc:0')
        self.x_pred = self.graph.get_tensor_by_name('x_pred:0')
        self.noise_ = self.graph.get_tensor_by_name('rand_noise:0')
        self.prob_ = self.graph.get_tensor_by_name('keep_prob:0')
        self.y = self.graph.get_tensor_by_name('y_proc:0')
        self.g_horizontal = self.graph.get_tensor_by_name('Generator/im_h:0')
        self.g_horizontal = tf.nn.tanh(self.g_horizontal)
        self.yv = util.catVerticalResizeV2(self.y_) # Make y vert

        self.g_full = self.graph.get_tensor_by_name('GeneratorV/imv_h_out:0')
        self.g_cube = util.makeCubePano(self.g_full)

        self.g_full = tf.nn.tanh(self.g_full)
        self.g_cube = tf.nn.tanh(self.g_cube)

    def buildRefineModel(self):
        self.iterator = util.initDataset(self.dataset_path, self.batch_sz, self.shuffle_buff)
        self.getDataset = self.iterator.get_next() # ** RUN THIS IN SESSION **

        # Load Trained Generator
        self.x_ = self.graph.get_tensor_by_name('x_in:0')
        self.y_ = self.graph.get_tensor_by_name('y_in:0')
        self.fov_ = self.graph.get_tensor_by_name('fov_in:0')
        self.pano_ = self.graph.get_tensor_by_name('pano_in:0')
        self.mask = self.graph.get_tensor_by_name('mask:0')
        self.x = self.graph.get_tensor_by_name('x_pred:0')
        self.y = self.graph.get_tensor_by_name('y_proc:0')
        self.noise_ = self.graph.get_tensor_by_name('rand_noise:0')
        self.prob_ = self.graph.get_tensor_by_name('keep_prob:0')
        self.g_ = self.graph.get_tensor_by_name('Generator/im_h:0')
        self.g_ = tf.stop_gradient(self.g_, name='gradStop') # STOP GRADIENT
        self.g_ = util.catConst(self.g_)

        self.y = util.catConst(self.y)
        # D real
        self.d_real, self.dr = model.Drefine(model.dstack(self.g_,self.y),prob=self.prob_)
        self.loss_d_real = model.bceLoss(self.d_real, tf.ones_like(self.d_real))

        # Refine Generator and D fake
        self.g_res = model.GRefine(self.g_,prob=self.prob_)
        self.g = tf.add(self.g_res, self.g_)
        self.d_fake, self.df = model.Drefine(model.dstack(self.g_res,self.y),
                                             prob=self.prob_, reuse=True)
        self.loss_g_adv = model.bceLoss(self.d_fake, tf.ones_like(self.d_fake))
        self.loss_g_valid = model.l1Loss(self.g_res, self.y)
        self.loss_d_fake = model.bceLoss(self.d_fake, tf.zeros_like(self.d_fake))

        # Total Loss
        self.loss_d = self.loss_d_real + self.loss_d_fake
        self.loss_g = self.loss_g_adv +  100 * self.loss_g_valid

        self.g_crop = util.constToNorm(self.g)

        # Optimize
        self.setLearningRate(lr_h=[0.0003, 0.0003])
        adamOpt = tf.train.AdamOptimizer

        d_var = [v for v in tf.trainable_variables() if v.name.startswith('DiscriminatorRef')]
        g_var = [v for v in tf.trainable_variables() if v.name.startswith('GeneratorRef')]
        print(g_var)
        print(d_var)
        with tf.variable_scope('OptimizerRef'):
            self.opt_d = adamOpt(self.lr_d, beta1=0.5).minimize(self.loss_d, global_step=self.step,
                                                                var_list=d_var)
            self.opt_g = adamOpt(self.lr_g, beta1=0.5).minimize(self.loss_g, global_step=self.step,
                                                                var_list=g_var)

        self.mergeSummaryRef()

    def buildModelNoise(self):

        sigmoidEnt = tf.nn.sigmoid_cross_entropy_with_logits
        self.initDataset()

        #self.initDataset()
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        #self.ys = tf.placeholder(tf.float32, shape=[None,28,28,1])

        # Generator
        self.noise_  = tf.placeholder(tf.float32, shape=[None,100])

        # D small
        self.ys = tf.image.resize_bilinear(self.ys,[16,64])
        print(self.ys.get_shape())

        # Loss Small
        self.ds_real, self.dsr = model.DfromNoise(self.ys)
        self.loss_ds_real = tf.reduce_mean(sigmoidEnt(logits=self.ds_real,
                                                      labels=tf.ones_like(self.ds_real) - 0.1))

        self.gs = model.GfromNoise(self.noise_)
        print(self.gs.get_shape())
        self.ds_fake, self.dsf = model.DfromNoise(self.gs,reuse=True)
        self.loss_gs = tf.reduce_mean(sigmoidEnt(logits=self.ds_fake,
                                                 labels=tf.ones_like(self.ds_fake)))
        self.loss_ds_fake = tf.reduce_mean(sigmoidEnt(logits=self.ds_fake,
                                                      labels=tf.zeros_like(self.ds_fake)))
        self.loss_ds = self.loss_ds_real + self.loss_ds_fake


        # Optimize
        self.setLearningRate(lr_s=[0.0002, 0.0002])#learning rate for optimization
        adamOpt = tf.train.AdamOptimizer


        # GAN
        gs_var = [v for v in tf.trainable_variables() if v.name.startswith('GeneratorNoise')]
        ds_var = [v for v in tf.trainable_variables() if v.name.startswith('DiscriminatorNoise')]
        lr_ds = 0.0002
        lr_gs = 0.0002
        with tf.variable_scope('OptimizerNoise'):
            self.opt_ds = adamOpt(lr_ds,beta1=0.5).minimize(self.loss_ds, var_list=ds_var)
            self.opt_gs = adamOpt(lr_gs,beta1=0.5).minimize(self.loss_gs, var_list=gs_var)

        # Add graph summary
        self.mergeSummaryNoise()

    def buildNetworkFoV(self, sess, graph_path, model_path,  model_name):
        self.sess = sess
        self.model_name = model_name
        self.model_path = model_path
        self.graph_path = graph_path

        # Init network
        self.initDatasetFOV()
        self.buildModelFOV()

    # ------ BUILD NETWORK -----
    def buildNetworkSmall(self, sess, graph_path, model_path, model_name):
        self.sess = sess
        self.model_name = model_name
        self.model_path = model_path
        self.graph_path = graph_path

        # Init network
        self.initDataset()
        self.buildModelFOV()
        self.buildModelSmall()

    def buildNetworkMed(self, sess, graph_path, model_path, model_name, meta_name):
        self.saver = tf.train.import_meta_graph(model_path + meta_name + '.meta')
        self.graph = tf.get_default_graph()

        self.sess = sess
        self.model_name = model_name
        self.model_path = model_path
        self.graph_path = graph_path
        self.meta_name = meta_name

        # Network
        self.buildModelPreMed() #initDataset inside here (different with Netsmall)

    def buildNetworkHigh(self, sess, graph_path, model_path, model_name, meta_name):
        self.saver = tf.train.import_meta_graph(model_path + meta_name + '.meta')
        self.graph = tf.get_default_graph()

        self.sess = sess
        self.model_name = model_name
        self.model_path = model_path
        self.graph_path = graph_path
        self.meta_name = meta_name

        # Network
        self.buildModelPreHigh()

    def buildNetworkTest(self, sess, graph_path, model_path, meta_name):
        self.saver = tf.train.import_meta_graph(model_path + meta_name + '.meta')
        self.graph = tf.get_default_graph()

        self.sess = sess
        self.model_path = model_path
        self.graph_path = graph_path
        self.meta_name = meta_name

        # Network
        self.buildTestModel()

    def buildNetworkRef(self, sess, graph_path, model_path, model_name, meta_name):
        self.saver = tf.train.import_meta_graph(model_path + meta_name + '.meta')
        self.graph = tf.get_default_graph()

        self.sess = sess
        self.model_name = model_name
        self.model_path = model_path
        self.graph_path = graph_path
        self.meta_name = meta_name

        # Network
        self.buildRefineModel()

    def buildNetworkNoise(self, sess, graph_path, model_path, model_name):
        self.sess = sess
        self.model_name = model_name
        self.model_path = model_path
        self.graph_path = graph_path

        # Init network
        self.buildModelNoise()

    def buildNetworkVert(self, sess, graph_path, model_path, model_name, meta_name, net_size):
        self.saver = tf.train.import_meta_graph(model_path + meta_name + '.meta')
        self.graph = tf.get_default_graph()

        self.sess = sess
        self.model_name = model_name
        self.model_path = model_path
        self.meta_name = meta_name
        self.graph_path = graph_path

        self.buildVertModel(net_size)

    def buildNetworkTestFull(self, sess, graph_path, model_path, meta_name):
        self.saver = tf.train.import_meta_graph(model_path + meta_name + '.meta')
        self.graph = tf.get_default_graph()

        self.sess = sess
        self.model_path = model_path
        self.meta_name = meta_name
        self.graph_path = graph_path

        self.buildTestModelFull()

    def trainNetworkFov(self, iterations, restore=False):
        self.summary = tf.summary.merge_all()
        sess = self.sess
        fov_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Fov')
        print(fov_list)
        saver = tf.train.Saver(fov_list)
        writer = tf.summary.FileWriter(self.graph_path, sess.graph)
        sess.run([tf.global_variables_initializer(), self.iterator.initializer])
        saver.restore(sess, self.model_path + 'v3.4_small_' + '250000')
        for iter in range(iterations + 1):
            x, fov = sess.run(self.getDataset)
            #sess.run([self.opt_fov], feed_dict={self.x_:x, self.fov_single:fov})

            if iter % 100 == 0 :
                summary,loss_fov_out, fov_out = sess.run([self.summary, self.loss_fov, self.fov_scalar],
                                                feed_dict={self.x_:x,
                                                                  self.fov_single:fov})
                print(iter, ':', loss_fov_out, ":", fov_out)
                writer.add_summary(summary, iter)
            if iter % 10000 == 0:
                print('.....Saving checkpoint....')
                # saver.save(sess, self.model_path + 'fov_trained_' + str(iter))
                print('....Finshed saving checkpoint....')


    # ----- TRAIN NETWORK ----- #
    def trainNetworkSmall(self, iterations, restore=False):
        sess = self.sess
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(self.graph_path, sess.graph)

        # Init variables and iterator
        sess.run([tf.global_variables_initializer(), self.iterator.initializer])

        if restore == True:
            saver.restore(sess, self.model_path + self.model_name + '250000')

        # Training loop
        for iter in range(iterations + 1):
            x, y, fov, pano = sess.run(self.getDataset)

            # Optimize FOV
            sess.run([self.opt_fov], feed_dict={self.x_: x, self.y_: y,
                                                 self.fov_: fov, self.pano_: pano})
            if iter % 2 == 0:
                # Optimize Discriminator
                sess.run([self.opt_ds], feed_dict={self.x_: x, self.y_: y,
                                                   self.fov_: fov, self.pano_: pano})
            # Optimize Generator
            sess.run([self.opt_gs], feed_dict={self.x_: x, self.y_: y,
                                               self.fov_: fov, self.pano_: pano})

            if iter % 200 == 0:
                summary, loss_d, dr, df, loss_g = sess.run([self.summary,
                                                            self.loss_ds,
                                                            self.dsr,
                                                            self.dsf,
                                                            self.loss_gs],
                                                           feed_dict={self.x_: x,
                                                                      self.y_: y,
                                                                      self.fov_: fov,
                                                                      self.pano_: pano})

                writer.add_summary(summary, iter)
                print('[%d/%d]: %g ~ %g || %g (0.5) ~ %g (1)'
                      %(iter, iterations, loss_g, loss_d, df, dr))

            if iter % 10000 == 0 and iter > 0:
                print('======== Saving Checkpoint =========')
                saver.save(sess, self.model_path + self.model_name + str(iter))
                print('==== Finished Saving Checkpoint ====')

    def trainNetworkMed(self, iterations, restore=False):
        sess = self.sess
        saver = self.saver # ** defined in buildModel **
        writer = tf.summary.FileWriter(self.graph_path, sess.graph)

        # Init variables and iterator
        sess.run([tf.global_variables_initializer(), self.iterator.initializer])
        # Restore trained variables
        saver.restore(sess, self.model_path + self.meta_name)

        if restore == True:
            saver.restore(sess, self.model_path + self.model_name)

        # Training loop
        for iter in range(iterations + 1):
            x, y, fov, pano = sess.run(self.getDataset)

            prob = 0.5 # dropout probability
            noise = np.ones([self.batch_sz, 128,512,3]) - 1
            #noise = np.random.uniform(-1.0,1.0,[self.batch_sz,128,512,3]) #will be resized in ops

            # Optimize Discriminator
            sess.run([self.opt_dm], feed_dict={self.x_: x, self.y_: y, self.noise_: noise,
                                               self.fov_: fov, self.pano_: pano, self.prob_: prob})
            # Optimize Generator
            sess.run([self.opt_gm], feed_dict={self.x_: x, self.y_: y, self.noise_: noise,
                                                   self.fov_: fov, self.pano_: pano, self.prob_: prob})

            if iter % 200 == 0:
                summary, loss_d, dr, df, loss_g = sess.run([self.summary,
                                                            self.loss_dm,
                                                            self.dmr,
                                                            self.dmf,
                                                            self.loss_gm],
                                                           feed_dict={self.x_: x,
                                                                      self.y_: y,
                                                                      self.noise_: noise,
                                                                      self.fov_: fov,
                                                                      self.pano_: pano,
                                                                      self.prob_: prob})
                writer.add_summary(summary, iter)
                print('[%d/%d]: %g ~ %g || %g (0.5) ~ %g (1)'
                      %(iter, iterations, loss_g, loss_d, df, dr))

            if iter % 10000 == 0 and iter > 0:
                print('======== Saving Checkpoint =========')
                saver.save(sess, self.model_path + self.model_name + str(iter))
                print('==== Finished Saving Checkpoint ====')

    def trainNetworkHigh(self, iterations, restore=False):
        sess = self.sess
        saver = self.saver # ** defined in buildModel **
        writer = tf.summary.FileWriter(self.graph_path, sess.graph)

        # Init variables and iterator
        sess.run([tf.global_variables_initializer(), self.iterator.initializer])
        # Restore trained variables
        saver.restore(sess, self.model_path + self.meta_name)

        if restore == True:
            saver.restore(sess, self.model_path + self.model_name)

        # Training loop
        for iter in range(iterations + 1):
            x, y, fov, pano = sess.run(self.getDataset)

            prob = 0.5 # dropout probability
            noise = np.ones([self.batch_sz, 128,512,3]) - 1
            #noise = np.random.uniform(-1.0,1.0,[self.batch_sz,128,512,3]) #will be resized in ops

            # Optimize Discriminator
            sess.run([self.opt_d], feed_dict={self.x_: x, self.y_: y,
                                              self.noise_: noise, self.prob_: prob,
                                              self.fov_: fov, self.pano_: pano})
            # Optimize Generator
            sess.run([self.opt_g], feed_dict={self.x_: x, self.y_: y,
                                              self.noise_: noise, self.prob_: prob,
                                              self.fov_: fov, self.pano_: pano})

            if iter % 200 == 0:
                summary, loss_d, dr, df, loss_g = sess.run([self.summary,
                                                            self.loss_d,
                                                            self.dr,
                                                            self.df,
                                                            self.loss_g],
                                                           feed_dict={self.x_: x,
                                                                      self.y_: y,
                                                                      self.fov_: fov,
                                                                      self.noise_: noise,
                                                                      self.pano_: pano,
                                                                      self.prob_: prob})
                writer.add_summary(summary, iter)
                print('[%d/%d]: %g ~ %g || %g (0.5) ~ %g (1)'
                      %(iter, iterations, loss_g, loss_d, df, dr))

            if iter % 10000 == 0 and iter > 0:
                print('======== Saving Checkpoint =========')
                saver.save(sess, self.model_path + self.model_name + str(iter))
                print('==== Finished Saving Checkpoint ====')

    def trainNetworkVert(self, iterations, restore=False, params='train'):
        sess = self.sess
        saver = self.saver
        vert_saver = tf.train.Saver()
        writer = tf.summary.FileWriter(self.graph_path, sess.graph)

        # Init variables and iterator
        sess.run([tf.global_variables_initializer(), self.iterator.initializer])
        saver.restore(sess, self.model_path + self.meta_name)# restore trained model

        if restore == True:
            vert_saver.restore(sess, self.model_path + self.model_name + '300000')

        # Training loop
        for iter in range(iterations + 1):
            x, y, fov, pano = sess.run(self.getDataset)

            n_top = np.random.uniform(-1.0,1.0,[self.batch_sz,32,32,12]) #will be resized in ops
            n_bot = np.random.uniform(-1.0,1.0,[self.batch_sz,32,32,12]) #will be resized in ops


            prob = 0.5 # dropout probability
            noise = np.ones([self.batch_sz, 128,512,3]) - 1
            if iter % 2 == 0:
                # Optimize Discriminator
                sess.run([self.opt_dv], feed_dict={self.x_: x, self.y_: y,
                                                   self.noise_: noise, self.prob_: prob,
                                                   self.fov_: fov, self.pano_: pano })
            # Optimize Generator
            sess.run([self.opt_gv], feed_dict={self.x_: x, self.y_: y,
                                               self.noise_: noise, self.prob_: prob,
                                                 self.fov_: fov, self.pano_: pano })

            if iter % 100 == 0:
                summary, loss_d, dr, df, loss_g = sess.run([self.summary,
                                                            self.loss_dv,
                                                            self.dv_r,
                                                            self.dv_f,
                                                            self.loss_gv],
                                                           feed_dict={self.x_: x,
                                                                      self.y_: y,
                                                                      self.fov_: fov,
                                                                      self.noise_: noise, self.prob_: prob,
                                                                      self.pano_: pano
                                                                      })

                writer.add_summary(summary, iter)
                print('[%d/%d]: %g ~ %g || %g (0.5) ~ %g (1)'
                      %(iter, iterations, loss_g, loss_d, df, dr))

            if iter % 10000 == 0 and iter > 0:
                print('======== Saving Checkpoint =========')
                vert_saver.save(sess, self.model_path + self.model_name + str(iter))
                print('==== Finished Saving Checkpoint ====')

    def trainNetworkRef(self, iterations, restore=False):
        sess = self.sess
        saver = self.saver # ** defined in buildModel **
        writer = tf.summary.FileWriter(self.graph_path, sess.graph)

        # Init variables and iterator
        sess.run([tf.global_variables_initializer(), self.iterator.initializer])
        # Restore trained variables
        saver.restore(sess, self.model_path + self.meta_name)

        if restore == True:
            saver.restore(sess, self.model_path + self.model_name)

        # Training loop
        for iter in range(iterations + 1):
            x, y, fov, pano = sess.run(self.getDataset)

            prob = 0.5 # dropout probability
            noise = np.zeros([self.batch_sz, 128,512,3])

            # Optimize Discriminator
            sess.run([self.opt_d], feed_dict={self.x_: x, self.y_: y, self.prob_: prob,
                                              self.fov_: fov, self.pano_: pano, self.noise_: noise})
            # Optimize Generator
            sess.run([self.opt_g], feed_dict={self.x_: x, self.y_: y, self.prob_: prob,
                                                   self.fov_: fov, self.pano_: pano, self.noise_: noise})

            if iter % 200 == 0:
                summary, loss_d, dr, df, loss_g = sess.run([self.summary,
                                                            self.loss_d,
                                                            self.dr,
                                                            self.df,
                                                            self.loss_g],
                                                           feed_dict={self.x_: x,
                                                                      self.y_: y,
                                                                      self.fov_: fov,
                                                                      self.prob_: prob,
                                                                      self.noise_: noise,
                                                                      self.pano_: pano})
                writer.add_summary(summary, iter)
                print('[%d/%d]: %g ~ %g || %g (0.5) ~ %g (1)'
                      %(iter, iterations, loss_g, loss_d, df, dr))

            if iter % 10000 == 0 and iter > 0:
                print('======== Saving Checkpoint =========')
                saver.save(sess, self.model_path + self.model_name + str(iter))
                print('==== Finished Saving Checkpoint ====')

    def trainNetworkNoise(self, iterations, restore=False):
        sess = self.sess
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(self.graph_path, sess.graph)

        sess.run([tf.global_variables_initializer(), self.iterator.initializer])

        for iter in range(iterations+1):
            noise = np.random.uniform(-1,1,[2,100])
            x, y, fov, pano = sess.run(self.getDataset)
            #train_x = self.mnist.train.next_batch(2)
            #train_x = np.reshape(train_x[0],[2,28,28,1])

            # Optimize Discriminator
            sess.run([self.opt_ds], feed_dict={self.y_: y, self.noise_: noise})
            # Optimize Generator
            sess.run([self.opt_gs], feed_dict={self.noise_: noise})

            if iter % 100 == 0:
                summary, loss_d, dr, df, loss_g = sess.run([self.summary,
                                                            self.loss_ds,
                                                            self.dsr,
                                                            self.dsf,
                                                            self.loss_gs],
                                                           feed_dict={self.x_: x,
                                                                      self.y_: y,
                                                                      self.noise_: noise,
                                                                      self.fov_: fov,
                                                                      self.pano_: pano})
                writer.add_summary(summary, iter)
                print('[%d/%d]: %g ~ %g || %g (0.5) ~ %g (1)'
                      %(iter, iterations, loss_g, loss_d, df, dr))
        sess = self.sess
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(self.graph_path, sess.graph)

        # Init variables and iterator
        sess.run([tf.global_variables_initializer(), self.iterator.initializer])

        if restore == True:
            saver.restore(sess, self.model_path + self.model_name + '250000')

        # Training loop
        for iter in range(iterations + 1):
            x, y, fov, pano = sess.run(self.getDataset)

            # Optimize FOV
            sess.run([self.opt_fov], feed_dict={self.x_: x, self.y_: y,
                                                 self.fov_: fov, self.pano_: pano})
            if iter % 2 == 0:
                # Optimize Discriminator
                sess.run([self.opt_ds], feed_dict={self.x_: x, self.y_: y,
                                                   self.fov_: fov, self.pano_: pano})
            # Optimize Generator
            sess.run([self.opt_gs], feed_dict={self.x_: x, self.y_: y,
                                               self.fov_: fov, self.pano_: pano})

            if iter % 200 == 0:
                summary, loss_d, dr, df, loss_g = sess.run([self.summary,
                                                            self.loss_ds_v,
                                                            self.dsr_v,
                                                            self.dsf_v,
                                                            self.loss_gs_v],
                                                           feed_dict={self.x_: x,
                                                                      self.y_: y,
                                                                      self.fov_: fov,
                                                                      self.pano_: pano})

                writer.add_summary(summary, iter)
                print('[%d/%d]: %g ~ %g || %g (0.5) ~ %g (1)'
                      %(iter, iterations, loss_g, loss_d, df, dr))

            if iter % 10000 == 0 and iter > 0:
                print('======== Saving Checkpoint =========')
                saver.save(sess, self.model_path + self.model_name + str(iter))
                print('==== Finished Saving Checkpoint ====')

    def runTestNetwork(self, iterations, save_path, net_type='horizontal'):
        sess = self.sess
        saver = self.saver

        sess.run([self.iterator.initializer])
        saver.restore(sess, self.model_path + self.meta_name)

        im_path = '../test_data_real/data_real_4/'
        save_path = im_path
        if net_type == 'inference':
            x0 = util.resizeImage(util.loadImage(im_path + 'im1.jpg',normalize=True))
            print('loading image')
            x1 = util.resizeImage(util.loadImage(im_path + 'im2.jpg',normalize=True))
            x2 = util.resizeImage(util.loadImage(im_path + 'im3.jpg',normalize=True))
            x3 = util.resizeImage(util.loadImage(im_path + 'im4.jpg',normalize=True))
            zeros = np.zeros_like(x0)
            x_cont = np.dstack((x0,x1,x2,x3,zeros,zeros))
            x = np.expand_dims(x_cont,0)

            # Input
            x = np.concatenate((x,x),axis=0)
            y = np.zeros_like(x)
            noise = np.zeros([self.batch_sz, 128,512,3])
            pano = np.zeros([2,512,1024,3],dtype=np.float32)
            fov = np.zeros([2,128])

            im_full, im_cube, im_horiz, xpred = \
                sess.run([self.g_full, self.g_cube, self.g_horizontal, self.x_pred],
                         feed_dict={self.x_: x, self.y_: y, self.fov_: fov,
                                    self.noise_: noise, self.pano_: pano})

            util.saveImage(im_full, save_path, 'im_full.png' )
            util.saveImage(im_cube, save_path, 'im_cube.png' )
            util.saveImage(im_horiz, save_path, 'im_horiz.png' )
            util.saveImage(xpred, save_path, 'im_pred.png' )
            return

        # y_crop = model.constLoss(self.y, 0)
        total_time = 0.0
        for iter in range(iterations):
            t_begin = time.time()
            x, y, fov, pano = sess.run(self.getDataset)

            noise = np.zeros([self.batch_sz, 128,512,3])

            if net_type == 'horizontal':
                x_in, x_pred, g_out, y_gt, y_v = sess.run([self.x_proc, self.x_pred,
                                                           self.g, self.y, y_crop],
                                                      feed_dict={self.x_: x,
                                                                 self.y_: y,
                                                                 self.fov_: fov,
                                                                 self.noise_: noise,
                                                                 self.pano_: pano})
                print('step [%d/%d]' %(iter, iterations))

                # Save image
                util.saveImage(x_in, save_path, '/input' + str(iter) + '.png')
                util.saveImage(x_pred, save_path, 'xpred' + str(iter) + '.png')
                util.saveImage(g_out, save_path, 'output' + str(iter) + '.png')
                util.saveImage(y_gt, save_path, 'gt' + str(iter) + '.png')

            elif net_type == 'full':
                x_in, x_pred, g_h_out, g_full_out, g_cube_out, y_gt, y_v = \
                    sess.run([self.x_proc, self.x_pred, self.g_horizontal,
                              self.g_full, self.g_cube, self.y, self.yv],
                             feed_dict={self.x_: x,
                                        self.y_: y,
                                        self.fov_: fov,
                                        self.noise_: noise,
                                        self.pano_: pano})
                elapsed_time = time.time() - t_begin
                total_time = total_time + elapsed_time
                print('step [%d/%d]' %(iter, iterations))

                # Save image
                util.saveImage(x_in, save_path, 'input' + str(iter) + '.png')
                util.saveImage(x_pred, save_path, 'xpred' + str(iter) + '.png')
                util.saveImage(g_h_out, save_path, 'output_horizontal' + str(iter) + '.png')
                util.saveImage(g_full_out, save_path, 'output_full' + str(iter) + '.png')
                util.saveImage(g_cube_out, save_path, 'output' + str(iter) + '.png')
                util.saveImage(y_gt, save_path, 'gt' + str(iter) + '.png')
                util.saveImage(y_v, save_path, 'gt_vertical' + str(iter) + '.png')

                # save_path = '/home/juliussurya/workspace/360pano/incept_score/inception-score-pytorch/horizontal_fake/pano'
                # util.saveImage(g_h_out, save_path, '/im' + str(iter) + '.png')
                # util.saveImage(g_full_out, save_path, 'output_eval/vertical_fake/pano/im' + str(iter) + '.png')
                # util.saveImage(y_gt, save_path, 'output_eval/horizontal_real/pano/im' + str(iter) + '.png')
                # util.saveImage(y_v, save_path, 'output_eval/vertical_real/pano/im' + str(iter) + '.png')


    def setLearningRate(self, lr_s=[0.0006, 0.0005], lr_m=[0.0006,0.0002], lr_h=[0.0006,0.0002]):
        with tf.variable_scope('GlobalSteps'):
            # FOV
            self.step_fov = tf.Variable(0, trainable=False)
            self.lr_fov = tf.train.exponential_decay(0.0001, self.step_fov, 2000,
                                                     0.90, staircase=True)

            self.step_s = tf.Variable(0, trainable=False)
            self.lr_gs = tf.train.exponential_decay(lr_s[0], self.step_s, 200000,
                                                    0.90, staircase=True)
            self.lr_ds = tf.train.exponential_decay(lr_s[1], self.step_s, 200000,
                                                    0.90, staircase=True)

            self.step_m = tf.Variable(0, trainable=False)
            self.lr_gm = tf.train.exponential_decay(lr_m[0], self.step_m, 20000,
                                                    0.90, staircase=True)
            self.lr_dm = tf.train.exponential_decay(lr_m[1], self.step_m, 20000,
                                                    0.90, staircase=True)

            self.step = tf.Variable(0, trainable=False)
            self.lr_g = tf.train.exponential_decay(lr_h[0], self.step, 200000,
                                                   0.90, staircase=True)
            self.lr_d = tf.train.exponential_decay(lr_h[1], self.step, 200000,
                                                   0.90, staircase=True)

    def mergeSummarySmall(self):
        with tf.name_scope('Small'):
            with tf.name_scope('Total'):
                tf.summary.scalar('Generator', self.loss_gs)
                tf.summary.scalar('Discriminator', self.loss_ds)
            with tf.name_scope('GLoss'):
                tf.summary.scalar('Adv', self.loss_gs_adv)
                # tf.summary.scalar('Hole', self.loss_gs_hole)
                # tf.summary.scalar('Valid', self.loss_gs_valid)
            with tf.name_scope('DLoss'):
                tf.summary.scalar('Real', self.loss_ds_real)
                tf.summary.scalar('Fake', self.loss_ds_fake)

        tf.summary.image('Input', self.x)
        tf.summary.image('OutputS', self.gs)
        tf.summary.image('Cropped', self.gs_crop)
        tf.summary.image('OutputM', self.gm)
        tf.summary.image('Output', self.g)
        tf.summary.image('Gt', self.y)
        tf.summary.image('GtM', self.ym)
        tf.summary.image('GtS', self.ys)

        self.summary = tf.summary.merge_all()

        return 0

    def mergeSummaryVertS(self):
        # im_out = tf.concat((self.gv_s[:,:,:,0:3], self.gv_s[:,:,:,3:6],
        #                     self.gv_s[:,:,:,6:9], self.gv_s[:,:,:,9:12]), axis=2)
        im_out = self.gv_s
        tf.summary.image('OutputVSmall', im_out)
        # im_gt = tf.concat((self.yv_s[:,:,:,0:3], self.yv_s[:,:,:,3:6],
        #                     self.yv_s[:,:,:,6:9], self.yv_s[:,:,:,9:12]), axis=2)
        im_gt = self.yv_s
        tf.summary.image('GtVSmall', im_gt)
        # im_in = tf.concat((self.gv_in_s[:,:,:,0:3], self.gv_in_s[:,:,:,3:6],
        #                     self.gv_in_s[:,:,:,6:9], self.gv_in_s[:,:,:,9:12]), axis=2)
        im_in = self.gv_in_s
        tf.summary.image('InVSmall', im_in)

        self.summary = tf.summary.merge_all()

    def mergeSummaryVertM(self):
        # im_out = tf.concat((self.gv_m[:,:,:,0:3], self.gv_m[:,:,:,3:6],
        #                     self.gv_m[:,:,:,6:9], self.gv_m[:,:,:,9:12]), axis=2)
        im_out = self.gv_m
        tf.summary.image('OutputVMed', im_out)
        # im_gt = tf.concat((self.yv_m[:,:,:,0:3], self.yv_m[:,:,:,3:6],
        #                     self.yv_m[:,:,:,6:9], self.yv_m[:,:,:,9:12]), axis=2)
        im_gt = self.yv_m
        tf.summary.image('GtMedium', im_gt)
        # im_in = tf.concat((self.gv_in_m[:,:,:,0:3], self.gv_in_m[:,:,:,3:6],
        #                     self.gv_in_m[:,:,:,6:9], self.gv_in_m[:,:,:,9:12]), axis=2)
        im_in = self.gv_in_m
        tf.summary.image('InVMed', im_in)

        self.summary = tf.summary.merge_all()

    def mergeSummaryVertH(self):
        # im_out = tf.concat((self.gv_m[:,:,:,0:3], self.gv_m[:,:,:,3:6],
        #                     self.gv_m[:,:,:,6:9], self.gv_m[:,:,:,9:12]), axis=2)
        im_out = self.gv_h
        tf.summary.image('OutputVHigh', im_out)
        # im_gt = tf.concat((self.yv_m[:,:,:,0:3], self.yv_m[:,:,:,3:6],
        #                     self.yv_m[:,:,:,6:9], self.yv_m[:,:,:,9:12]), axis=2)
        im_gt = self.yv
        tf.summary.image('GtVHigh', im_gt)
        # im_in = tf.concat((self.gv_in_m[:,:,:,0:3], self.gv_in_m[:,:,:,3:6],
        #                     self.gv_in_m[:,:,:,6:9], self.gv_in_m[:,:,:,9:12]), axis=2)
        im_in = self.gv_in
        tf.summary.image('InVHigh', im_in)

        self.summary = tf.summary.merge_all()

    def mergeSummaryMed(self):
        with tf.name_scope('Medium'):
            with tf.name_scope('Total'):
                tf.summary.scalar('Generator', self.loss_gm)
                tf.summary.scalar('Discriminator', self.loss_dm)
            with tf.name_scope('GLoss'):
                tf.summary.scalar('Adv', self.loss_gm_adv)
                tf.summary.scalar('Hole', self.loss_gm_hole)
                tf.summary.scalar('Valid', self.loss_gm_valid)
            with tf.name_scope('DLoss'):
                tf.summary.scalar('Real', self.loss_dm_real)
                tf.summary.scalar('Fake', self.loss_dm_fake)
        self.summary = tf.summary.merge_all()

    def mergeSummaryHigh(self):
        with tf.name_scope('High'):
            with tf.name_scope('Total'):
                tf.summary.scalar('Generator', self.loss_g)
                tf.summary.scalar('Discriminator', self.loss_d)
            with tf.name_scope('GLoss'):
                tf.summary.scalar('Adv', self.loss_g_adv)
                tf.summary.scalar('Hole', self.loss_g_hole)
                tf.summary.scalar('Valid', self.loss_g_valid)
            with tf.name_scope('DLoss'):
                tf.summary.scalar('Real', self.loss_d_real)
                tf.summary.scalar('Fake', self.loss_d_fake)
        self.summary = tf.summary.merge_all()

    def mergeSummaryRef(self):
        with tf.name_scope('Refine'):
            with tf.name_scope('Total'):
                tf.summary.scalar('Generator', self.loss_g)
                tf.summary.scalar('Discriminator', self.loss_d)
            with tf.name_scope('GLoss'):
                tf.summary.scalar('Adv', self.loss_g_adv)
                tf.summary.scalar('Valid', self.loss_g_valid)
            with tf.name_scope('DLoss'):
                tf.summary.scalar('Real', self.loss_d_real)
                tf.summary.scalar('Fake', self.loss_d_fake)
            tf.summary.image('Refined', self.g)
            tf.summary.image('Refined_cropped', self.g_crop)
        self.summary = tf.summary.merge_all()

    def mergeSummaryNoise(self):
        with tf.name_scope('Small'):
            with tf.name_scope('Total'):
                tf.summary.scalar('Generator', self.loss_gs)
                tf.summary.scalar('Discriminator', self.loss_ds)
            with tf.name_scope('DLoss'):
                tf.summary.scalar('Real', self.loss_ds_real)
                tf.summary.scalar('Fake', self.loss_ds_fake)

        #tf.summary.image('Input', self.x)
        tf.summary.image('OutputS', self.gs)
        tf.summary.image('GtS', self.ys)

        self.summary = tf.summary.merge_all()

        return 0