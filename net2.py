import tensorflow as tf
import matplotlib.pyplot as plt
import util
import model 
import random

class Network2():
    def __init__(self, dataset_path, batch_sz=2, shuffle_buff=50):
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
    
    def initIterator(self):
        self.iterator = util.initDataset(self.dataset_path, self.batch_sz, self.shuffle_buff) 
        self.getDataset = self.iterator.get_next()

    def readDataset(self):
        # Initialize iterator
        self.initIterator()

        # placeholder
        self.x_ = tf.placeholder(tf.float32,[None,256,256,18], name='x_in')
        self.y_ = tf.placeholder(tf.float32,[None,256,256,18], name='y_in')
        self.fov_ = tf.placeholder(tf.float32,[None,128], name='fov_in')

        # Process input
        self.fov_gt = self.fov_
        self.x = util.catResize(self.x_) # concat and resize
        self.y = util.catResize(self.y_) # concat and resize
        self.ym = tf.image.resize_bilinear(self.y,[64,256])
        self.ys = tf.image.resize_bilinear(self.y,[32,128])
        return self.x, self.y, self.fov_gt

    def forwardFOV(self):
        self.fov_pred, fov_scalar = model.fovNet(self.x)
        self.x = util.procInputFov(self.x,fov_scalar)
        self.xm = tf.image.resize_bilinear(self.x, [64,256])
        self.xs = tf.image.resize_bilinear(self.x, [32,128])
        self.mask = util.createMask(fov_scalar)
        self.maskm = tf.image.resize_bilinear(self.mask,[64,256])
        self.masks = tf.image.resize_bilinear(self.mask,[32,128])
    
    def forwardGAN(self):
        # G
        self.gs, self.gm, self.g = model.Gs3(self.x)
        # D small
        self.ds_real, self.dsr = model.Ds(model.dstack(self.xs,self.ys))
        self.ds_fake, self.dsf = model.Ds(model.dstack(self.xs,self.gs),reuse=True)
        # D med
        self.dm_real, self.dmr  = model.Dm(model.dstack(self.xm, self.ym))
        self.dm_fake, self.dmf = model.Dm(model.dstack(self.xm, self.gm),reuse=True)
        # D high
        self.d_real, self.dr = model.D(model.dstack(self.x,self.y))
        self.d_fake, self.df = model.D(model.dstack(self.x,self.g),reuse=True)

        self.lbl_real = tf.ones_like(self.ds_real) * random.uniform(0.8,1.1)
        self.lbl_fake = tf.zeros_like(self.ds_fake) + random.uniform(0.0,0.4)

    def forward(self):
        # Run FOVNET
        self.forwardFOV()
        # Run GAN
        self.forwardGAN()

    def lossFOV(self, a=1):   
        sigmoidEnt = tf.nn.sigmoid_cross_entropy_with_logits
        self.loss_fov = tf.reduce_mean(sigmoidEnt(logits=self.fov_pred, labels=self.fov_gt))
    
    def lossGAN(self):
        # Loss Small
        self.loss_ds_real, self.loss_ds_fake, self.loss_ds = model.dLossGan(
            self.ds_real, self.ds_fake, self.lbl_real, self.lbl_fake)
        self.loss_gs_adv, self.loss_gs_hole, self.loss_gs_valid, self.loss_gs = model.gLossGanMask(
            self.gs,self.ds_fake,self.ys,self.masks,self.lbl_real,10,10)

        # Loss Med
        self.loss_dm_real, self.loss_dm_fake, self.loss_dm = model.dLossGan(
            self.dm_real,self.dm_fake, self.lbl_real, self.lbl_fake)
        self.loss_gm_adv, self.loss_gm_hole, self.loss_gm_valid, self.loss_gm = model.gLossGanMask(
            self.gm,self.dm_fake,self.ym,self.maskm,self.lbl_real,10,10)

        # Loss High
        self.loss_d_real, self.loss_d_fake, self.loss_d = model.dLossGan(
            self.d_real, self.d_fake, self.lbl_real, self.lbl_fake)
        self.loss_g_adv, self.loss_g_hole, self.loss_g_valid, self.loss_g = model.gLossGanMask(
            self.g, self.d_fake, self.y, self.mask, self.lbl_real,10,10)

    def loss(self):
        self.lossGAN()
        self.lossFOV()

    def setLearingRate(self, lr_s=[0.0006, 0.0002], lr_m=[0.0006,0.0002], lr_h=[0.0006,0.0002]):
        with tf.variable_scope('GlobalSteps'):
            # FOV
            self.step_fov = tf.Variable(0, trainable=False) 
            self.lr_fov = tf.train.exponential_decay(0.0001, self.step_fov, 2000, 0.90, staircase=True)
        
            self.step_s = tf.Variable(0, trainable=False)
            self.lr_gs = tf.train.exponential_decay(lr_s[0], self.step_s, 20000, 0.90, staircase=True)
            self.lr_ds = tf.train.exponential_decay(lr_s[1], self.step_s, 20000, 0.90, staircase=True)
        
            self.step_m = tf.Variable(0, trainable=False)
            self.lr_gm = tf.train.exponential_decay(lr_m[0], self.step_m, 20000, 0.90, staircase=True)
            self.lr_dm = tf.train.exponential_decay(lr_m[1], self.step_m, 20000, 0.90, staircase=True)

            self.step = tf.Variable(0, trainable=False)
            self.lr_g = tf.train.exponential_decay(lr_h[0], self.step, 20000, 0.90, staircase=True)
            self.lr_d = tf.train.exponential_decay(lr_h[1], self.step, 20000, 0.90, staircase=True)

    def optimize(self):
        self.setLearingRate()

        # FOV
        self.opt_fov = tf.train.AdamOptimizer(self.lr_fov).minimize(self.loss_fov, global_step=self.step_fov)
        # Small
        d_var_s = [v for v in tf.trainable_variables() if v.name.startswith('DiscriminatorS')]
        g_var_s = [v for v in tf.trainable_variables() if v.name.startswith('Generator/g_') or v.name.startswith('Generator/im_s')]
        self.opt_gs = tf.train.AdamOptimizer(self.lr_gs).minimize(self.loss_gs, global_step=self.step_s, var_list=g_var_s)
        self.opt_ds = tf.train.AdamOptimizer(self.lr_ds).minimize(self.loss_ds, global_step=self.step_s, var_list=d_var_s)

        # Med
        d_var_m = [v for v in tf.trainable_variables() if v.name.startswith('DiscriminatorM')]
        g_var_m = [v for v in tf.trainable_variables() if v.name.startswith('Generator/g_') or v.name.startswith('Generator/im_m')]
        self.opt_gm = tf.train.AdamOptimizer(self.lr_gm).minimize(self.loss_gm, global_step=self.step_m, var_list=g_var_m)
        self.opt_dm = tf.train.AdamOptimizer(self.lr_ds).minimize(self.loss_dm, global_step=self.step_m, var_list=d_var_m)

        d_var = [v for v in tf.trainable_variables() if v.name.startswith('DiscriminatorH')]
        g_var = [v for v in tf.trainable_variables() if v.name.startswith('Generator/g_') or v.name.startswith('Generator/im_h')]
        self.opt_g = tf.train.AdamOptimizer(self.lr_g).minimize(self.loss_g, global_step=self.step, var_list=g_var)
        self.opt_d = tf.train.AdamOptimizer(self.lr_d).minimize(self.loss_d, global_step=self.step, var_list=d_var)

    def build(self,sess, graph_path, model_path, model_name):
        self.sess = sess
        self.model_name = model_name
        self.model_path = model_path
        self.graph_path = graph_path

        # Build network
        self.readDataset()
        self.forward()
        self.loss()
        self.setLearingRate()
        self.optimize()

        self.mergeSummary()

    def gatherVars(self):
        self.vars_g = model.getModelVars('Generator')
        self.vars_d = model.getModelVars('DiscriminatorS')
        # Append all the vars_dictionary
        vars_dict = {}
        vars_dict.update(self.vars_g)
        vars_dict.update(self.vars_d)
        return vars_dict

    def train(self, iter_s, iter_m, iter, restore=False):
        sess = self.sess
        self.vars_dict = self.gatherVars() 
        saver = tf.train.Saver(self.vars_dict)
        writer = tf.summary.FileWriter(self.graph_path, sess.graph)

        # Small
        self.iter_s = iter_s
        sess.run([tf.global_variables_initializer(), self.iterator.initializer])
        if restore == True:
            saverRestore = tf.train.Saver(self.vars_dict)
            saverRestore.restore(sess, self.model_path+'v3.0_test_small_400')
        for itr in range(self.iter_s + 1):
            x, y, fov = sess.run(self.getDataset)
            sess.run([self.opt_fov], feed_dict={self.x_:x, self.y_:y, self.fov_:fov})
            if itr % 2 == 0 :
                sess.run(self.opt_ds, feed_dict={self.x_:x, self.y_:y, self.fov_:fov})
            if itr % 2 != 0:
                sess.run(self.opt_gs, feed_dict={self.x_:x, self.y_:y, self.fov_:fov})

            if itr % 200 == 0:
                summary, loss_ds, loss_gs, dr, df = sess.run([self.summary, self.loss_ds, self.loss_gs,
                                                      self.dsr, self.dsf],
                                                    feed_dict={self.x_:x,self.y_:y,self.fov_:fov})
                writer.add_summary(summary,itr)
                print('Step[%d/%d]: %g -- %g || %g (0.5) -- %g (1)'
                      %(itr,self.iter_s,loss_gs,loss_ds, df[0], dr[0]))

            if itr % 400 == 0 and itr > 0:
                print('Saving checkpoint..')
                saver.save(sess, self.model_path + self.model_name + '_small_' + str(itr))
        
        # # Med
        # self.iter_m = iter_m

        # sess.run([tf.global_variables_initializer(), self.iterator.initializer])
        # for itr in range(self.iter_m + 1):
        #     x, y, fov = sess.run(self.getDataset)
        #     sess.run([self.opt_dm], feed_dict={self.x_:x, self.y_:y, self.fov_:fov})
        #     sess.run(self.opt_gm, feed_dict={self.x_:x, self.y_:y, self.fov_:fov})

        #     if itr % 200 == 0:
        #         summary, loss_dm, loss_gm, dmf, dmr = sess.run([self.summary, self.loss_dm, self.loss_gm,
        #                                               self.dmf, self.dmr],
        #                                             feed_dict={self.x_:x,self.y_:y,self.fov_:fov})
        #         writer.add_summary(summary,itr)
        #         print('Step[%d/%d]: %g -- %g || %g (0.5) -- %g (1)' 
        #               %(itr,self.iter_m,loss_gm,loss_dm, dmf[0], dmr[0]))

        #     if itr % 10000 == 0 and itr > 0:
        #         print('Saving checkpoint..')
        #         saver.save(sess, self.model_path + self.model_name + '_medium_' + str(itr))
        
        # # High
        # self.iter = iter
        # sess.run([tf.global_variables_initializer(), self.iterator.initializer])
        # for itr in range(self.iter + 1):
        #     x, y, fov = sess.run(self.getDataset)
        #     sess.run([self.opt_d], feed_dict={self.x_:x, self.y_:y, self.fov_:fov})
        #     sess.run(self.opt_g, feed_dict={self.x_:x, self.y_:y, self.fov_:fov})

        #     if itr % 200 == 0:
        #         summary, loss_d, loss_g = sess.run([self.summary, self.loss_d, self.loss_g],
        #                                             feed_dict={self.x_:x,self.y_:y,self.fov_:fov})
        #         writer.add_summary(summary,itr)
        #         print('Step[%d/%d]: %g -- %g' %(itr,self.iter,loss_g,loss_d))

        #     if itr % 10000 == 0 and itr > 0:
        #         print('Saving checkpoint..')
        #         saver.save(sess, self.model_path + self.model_name + '_high_' + str(itr))
       

        writer.close()

    def loadMetaSmall(self ,import_scope=None):
        self.saver = tf.train.import_meta_graph(self.model_path+self.meta_name+'.meta',import_scope=import_scope)
        self.graph = tf.get_default_graph()

    def buildfromSmall(self,sess, graph_path, model_path, model_name, meta_name):
        self.sess = sess
        self.model_name = model_name
        self.model_path = model_path
        self.graph_path = graph_path
        self.meta_name = meta_name

        # Load pretrained structured
        self.loadMetaSmall()

        print([v.name for v in self.graph.get_operations() if v.name.startswith('Generator')])
        # Build network
        self.readDataset()
        self.input_x = self.graph.get_tensor_by_name('x_in:0')
        output = self.graph.get_tensor_by_name('Generator/im_s/conv2d_transpose:0')
        self.output = tf.nn.tanh(output)
        
    def buildHighfromMed(self, sess, graph_path, model_path, model_name, meta_name):
        self.sess = sess
        self.model_name = model_name
        self.model_path = model_path
        self.graph_path = graph_path
        self.meta_name = meta_name
        self.loadMetaSmall()

        self.iterator = util.initDataset(self.dataset_path, self.batch_sz, self.shuffle_buff)
        self.getDataset = self.iterator.get_next()

        # NEtwork
        self.x_ = self.graph.get_tensor_by_name('Pretrained/x_in:0')
        self.y_ = self.graph.get_tensor_by_name('Pretrained/y_in:0')
        self.fov_ = self.graph.get_tensor_by_name('Pretrained/fov_in:0')
        self.y = util.catResize(self.y_)
        output = self.graph.get_tensor_by_name('Pretrained/Generator/im_h/conv2d_transpose:0')
        self.g = tf.nn.tanh(output)

        # D high
        self.d_real, self.dr = model.D(self.y)
        self.d_fake, self.df = model.D(self.g,reuse=True)

        self.lbl_real = tf.ones_like(self.d_real) * random.uniform(0.8,1.1)
        self.lbl_fake = tf.zeros_like(self.d_fake) + random.uniform(0.0,0.4)

        # Loss High
        self.loss_d_real, self.loss_d_fake, self.loss_d = model.dLossGan(
            self.d_real, self.d_fake, self.lbl_real, self.lbl_fake)
        self.loss_g_adv, self.loss_g_hole, self.loss_g_valid, self.loss_g = model.gLossGan(
            self.g, self.d_fake, self.y, self.lbl_real,10,10)


        self.setLearingRate()
        d_var = [v for v in tf.trainable_variables() if v.name.startswith('DiscriminatorH')]
        g_var = [v for v in tf.trainable_variables() if v.name.startswith('Pretrained/Generator/g_') or v.name.startswith('Pretrained/Generator/im_h')]

        with tf.variable_scope('MinimizerH'):
            self.opt_g = tf.train.AdamOptimizer(self.lr_g).minimize(self.loss_g, global_step=self.step, var_list=g_var)
            self.opt_d = tf.train.AdamOptimizer(self.lr_d).minimize(self.loss_d, global_step=self.step, var_list=d_var)

    def buildMedfromSmall(self, sess, graph_path, model_path, model_name, meta_name):
        self.sess = sess
        self.model_name = model_name
        self.model_path = model_path
        self.graph_path = graph_path
        self.meta_name = meta_name
        self.loadMetaSmall()

        self.iterator = util.initDataset(self.dataset_path, self.batch_sz, self.shuffle_buff) 
        self.getDataset = self.iterator.get_next()
        # Generator
        self.x_ = self.graph.get_tensor_by_name('Pretrained/x_in:0')
        self.y_ = self.graph.get_tensor_by_name('Pretrained/y_in:0')
        self.fov_ = self.graph.get_tensor_by_name('Pretrained/fov_in:0')
        self.y = util.catResize(self.y_)
        self.ym = tf.image.resize_bilinear(self.y,[64,256])
        output = self.graph.get_tensor_by_name('Pretrained/Generator/im_m/conv2d_transpose:0')
        self.gm = tf.nn.tanh(output)
        print('Graph')
        print(output.graph == tf.get_default_graph())

        g_var_m = [v for v in tf.trainable_variables() if v.name.startswith('Generator/g_enc1/kernel')]
        print(g_var_m)
        g_var_m = [v for v in self.graph.get_operations() if v.name.startswith('Generator/g_enc1/kernel')]
        print(g_var_m)

        # D med
        self.dm_real, self.dmr  = model.Dm(self.ym)
        self.dm_fake, self.dmf = model.Dm(self.gm,reuse=True)

        self.lbl_real = tf.ones_like(self.dm_real) * random.uniform(0.8,1.1)
        self.lbl_fake = tf.zeros_like(self.dm_fake) + random.uniform(0.0,0.4)

        # LOSS
        # Loss Med
        self.loss_dm_real, self.loss_dm_fake, self.loss_dm = model.dLossGan(
            self.dm_real,self.dm_fake, self.lbl_real, self.lbl_fake)
        self.loss_gm_adv, self.loss_gm_hole, self.loss_gm_valid, self.loss_gm = model.gLossGan(
            self.gm,self.dm_fake,self.ym,self.lbl_real,10,10)

        # Optimize
        self.setLearingRate()
        # Med
        d_var_m = [v for v in tf.trainable_variables() if v.name.startswith('DiscriminatorM')]
        g_var_m = [v for v in tf.trainable_variables() if v.name.startswith('Pretrained/Generator/g_') or v.name.startswith('Pretrained/Generator/im_m')]
        print(g_var_m)
        input("...")
        with tf.name_scope('Minimizer'):
            self.opt_gm = tf.train.AdamOptimizer(self.lr_gm,name='AdamGM').minimize(self.loss_gm, global_step=self.step_m, var_list=g_var_m)
            self.opt_dm = tf.train.AdamOptimizer(self.lr_ds,name="AdamDM").minimize(self.loss_dm, global_step=self.step_m, var_list=d_var_m)

        self.mergeSummary()

    def trainMedfromSmall(self, iter_m):
        sess = self.sess
        saver = self.saver
        writer = tf.summary.FileWriter(self.graph_path, sess.graph)
        # Med
        self.iter_m = iter_m
        sess.run([tf.global_variables_initializer(), self.iterator.initializer])
        # Restore trained variables
        saver.restore(sess,self.model_path+self.meta_name)
        for itr in range(self.iter_m + 1):
            x, y, fov = sess.run(self.getDataset)
            sess.run([self.opt_dm], feed_dict={self.x_:x, self.y_:y, self.fov_:fov})
            sess.run(self.opt_gm, feed_dict={self.x_:x, self.y_:y, self.fov_:fov})

            if itr % 200 == 0:
                summary, loss_dm, loss_gm, dmf, dmr = sess.run([self.summary, self.loss_dm, self.loss_gm,
                                                      self.dmf, self.dmr],
                                                    feed_dict={self.x_:x,self.y_:y, self.fov_:fov})
                writer.add_summary(summary,itr)
                print('Step[%d/%d]: %g -- %g || %g (0.5) -- %g (1)' 
                      %(itr,self.iter_m,loss_gm,loss_dm, dmf[0], dmr[0]))

            if itr % 10000 == 0 and itr > 0:
                print('Saving checkpoint..')
                saver.save(sess, self.model_path + self.model_name + '_medium_' + str(itr))

    def testSmall(self, iter_s):
        sess = self.sess
        #self.saver.restore(sess, self.model_path+'v3.0_small_200000')

        sess.run(self.iterator.initializer)
        for itr in range(50):
            x, y, _ = sess.run(self.getDataset)

            ys,pred = sess.run([self.ys,self.output], feed_dict={self.input_x:x, self.y_:y})
            util.saveImage(ys,self.output_folder, '/lbl_'+str(itr)+'_.png')
            util.saveImage(pred,self.output_folder, '/pred_'+str(itr)+'_.png')

    def trainHighfromMed(self, iter):
        sess = self.sess
        saver = self.saver
        writer = tf.summary.FileWriter(self.graph_path, sess.graph)

        self.summary = tf.summary.merge_all()
        self.iter = iter
        sess.run([tf.global_variables_initializer(), self.iterator.initializer])
        # Restore trained variables
        saver.restore(sess,self.model_path+self.meta_name)
        for itr in range(self.iter + 1):
            x, y, fov = sess.run(self.getDataset)
            sess.run([self.opt_d], feed_dict={self.x_:x, self.y_:y, self.fov_:fov})
            sess.run(self.opt_g, feed_dict={self.x_:x, self.y_:y, self.fov_:fov})

            if itr % 200 == 0:
                summary,loss_d, loss_g, df, dr = sess.run([self.summary, self.loss_d, self.loss_g,
                                                      self.df, self.dr],
                                                    feed_dict={self.x_:x,self.y_:y, self.fov_:fov})
                writer.add_summary(summary,itr)
                print('Step[%d/%d]: %g -- %g || %g (0.5) -- %g (1)' 
                      %(itr,self.iter,loss_g,loss_d, df[0], dr[0]))

            if itr % 10000 == 0 and itr > 0:
                print('Saving checkpoint..')
                saver.save(sess, self.model_path + self.model_name + '_high_' + str(itr))

    def buildHigh(self, sess, graph_path, model_path, model_name, meta_name):
        self.sess = sess
        self.model_name = model_name
        self.model_path = model_path
        self.graph_path = graph_path
        self.meta_name = meta_name
        self.loadMetaSmall()

        self.iterator = util.initDataset(self.dataset_path, 2, self.shuffle_buff) 
        self.getDataset = self.iterator.get_next()
        # Generator
        self.x_ = self.graph.get_tensor_by_name('Pretrained/x_in:0')
        self.y_ = self.graph.get_tensor_by_name('Pretrained/y_in:0')
        self.fov_ = self.graph.get_tensor_by_name('Pretrained/fov_in:0')
        self.pano_ = tf.placeholder(tf.float32, shape=[512,1024], name='pano_in')
        self.x_im = self.graph.get_tensor_by_name('Pretrained/ResizeBilinear:0')
        self.x_pred = self.graph.get_tensor_by_name('Pretrained/concat_6:0')
        self.y_im = self.graph.get_tensor_by_name('Pretrained/ResizeBilinear_1:0')
        self.y = util.catResize(self.y_)
        output = self.graph.get_tensor_by_name('Pretrained/Generator/im_h/conv2d_transpose:0')
        self.g = tf.nn.tanh(output)

    def testHigh(self,iter, save_path):
        sess = self.sess
        saver = self.saver

        self.iter = iter
        sess.run([tf.global_variables_initializer(), self.iterator.initializer])
        # Restore trained variables
        saver.restore(sess, self.model_path + self.meta_name)
        for itr in range(self.iter + 1):
            x, y, fov, pano = sess.run(self.getDataset)

            g, x_im, y_im, x_pred = sess.run([self.g, self.x_im, self.y_im, self.x_pred], 
                                             feed_dict={self.x_:x, self.y_:y,
                                                        self.fov_:fov, self.pano_:pano})
            print('Step[%d/%d]'%(itr,self.iter))

            # Save input
            util.saveImage(x_im, save_path,'input_'+str(itr)+'.png')
            util.saveImage(y_im, save_path,'gt_'+str(itr)+'.png')
            util.saveImage(x_pred, save_path,'xpred_'+str(itr)+'.png')
            util.saveImage(g, save_path,'g_'+str(itr)+'.png')

    def buildFullPano(self, sess, graph_path, model_path, model_name, meta_name):
        self.sess = sess
        self.model_name = model_name
        self.model_path = model_path
        self.graph_path = graph_path
        self.meta_name = meta_name
        self.loadMetaSmall()

        self.iterator = util.initDataset(self.dataset_path, self.batch_sz, self.shuffle_buff)
        self.getDataset = self.iterator.get_next()
        # Generator
        self.x_ = self.graph.get_tensor_by_name('Pretrained/x_in:0')
        self.y_ = self.graph.get_tensor_by_name('Pretrained/y_in:0')
        self.fov_ = self.graph.get_tensor_by_name('Pretrained/fov_in:0')
        self.pano_ = tf.placeholder(tf.float32, shape=[None,512,1024,3])
        self.pano_gt = tf.image.resize_bilinear(self.pano_,[128,512])
        self.y = util.catResize(self.y_)
        output = self.graph.get_tensor_by_name('Pretrained/Generator/im_h/conv2d_transpose:0')
        self.g = tf.nn.tanh(output)

        # Cylindrical to 360
        self.g_pano = model.fullPano(self.g)
        # loss function
        self.loss_pano = model.panoLoss(self.g_pano, self.pano_gt)
        # Minimize loss
        pano_vars = [v for v in tf.trainable_variables() if v.name.startswith('360Pano')]
        self.opt_pano = tf.train.AdamOptimizer().minimize(self.loss_pano,var_list=pano_vars)

        tf.summary.image('pano',self.g_pano)
        tf.summary.image('pano_gt',self.pano_gt)
        tf.summary.scalar('loss_pano',self.loss_pano)

    def trainFullPano(self, iter):
        sess = self.sess
        saver = self.saver
        writer = tf.summary.FileWriter(self.graph_path, sess.graph)
        self.summary = tf.summary.merge_all()
        sess.run([tf.global_variables_initializer(), self.iterator.initializer])
        saver.restore(sess,self.model_path + self.meta_name) #restore trained variables

        for itr in range(iter+1):
            x, y, fov, pano = sess.run(self.getDataset)
            summary, _, loss_pano = sess.run([self.summary, self.opt_pano, self.loss_pano],
                                             feed_dict = {self.x_: x, self.y_: y,
                                                          self.fov_: fov, self.pano_: pano})

            if itr % 200 == 0:
                print('Step [%d/%d]: %g' %(itr, iter, loss_pano))
                writer.add_summary(summary,itr)

            if itr % 10000 == 0:
                saver.save(sess,self.model_path + self.model_name + str(itr))




    def mergeSummary(self):
        # self.addSummaryFOV()
        # self.addSummaryScalar('small', self.loss_gs_adv, self.loss_gs_hole, self.loss_gs_valid,
        #                       self.loss_gs, self.loss_ds_fake, self.loss_ds_real, self.loss_ds)
        self.addSummaryScalar('Medium', self.loss_gm_adv, self.loss_gm_hole, self.loss_gm_valid,
                              self.loss_gm, self.loss_dm_fake, self.loss_dm_real, self.loss_dm)
        # self.addSummaryScalar('high', self.loss_g_adv, self.loss_g_hole, self.loss_g_valid,
        #                       self.loss_g, self.loss_d_fake, self.loss_d_real, self.loss_d)
        self.addSummaryImage('NewImages')
        self.summary = tf.summary.merge_all()

    def addSummaryScalar(self, scope, g_adv, g_hole, g_valid, total_g, d_fake, d_real, total_d):
        with tf.name_scope(scope):
            with tf.name_scope('GenLoss'):
                tf.summary.scalar('Adv', g_adv)
                tf.summary.scalar('Hole', g_hole)
                tf.summary.scalar('Valid', g_valid)
            with tf.name_scope('DiscLoss'):
                tf.summary.scalar('Fake', d_fake)
                tf.summary.scalar('Real', d_real)
            with tf.name_scope('TotalLoss'):
                tf.summary.scalar('G', total_g)
                tf.summary.scalar('D', total_d)

    def addSummaryImage(self, scope):
        with tf.name_scope(scope):
            # tf.summary.image('Input', self.x, 10)
            # tf.summary.image('OutputS', self.gs, 10)
            tf.summary.image('OutputM', self.gm, 10)
            # tf.summary.image('Output', self.g, 10)
            tf.summary.image('GT', self.y, 10)

    def addSummaryFOV(self):
        with tf.name_scope('FOVLoss'):
            tf.summary.scalar('FOV', self.loss_fov)

    def verify(self):           
        print(' . . . . . . . . . . . . . . . ')
        print('Dataset :')
        print(self.dataset_path)
        print('Batch :',self.batch_sz)
        print('Model :')
        print(self.model_path,self.model_name)
        input(' . . . Press to continue . . .')
