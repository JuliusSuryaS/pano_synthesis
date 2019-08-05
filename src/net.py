import tensorflow as tf
from util import *
import model

class Network():
    def __init__(self):
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
        self.batch_sz = 2
        self.print_range = 200
        self.save_range = 10000
        self.sess = None
        self.soft_noise_real = 1
        self.soft_noise_fake = 0

    def readDataset(self, dataset_path, batch_sz=2, shuffle_buff=50):
        self.dataset_path = dataset_path
        self.iterator = initDataset(dataset_path, batch_sz, shuffle_buff) 
        x_, y_, fov = self.iterator.get_next()

        # Process input
        self.fov_gt = fov
        self.x = catResize(x_) # concat and resize
        self.y = catResize(y_) # concat and resize
        return self.x, self.y, self.fov_gt

    def forwardFOV(self): 
        self.fov_pred, fov_scalar = model.fovNet(self.x)
        self.x = procInputFov(self.x,fov_scalar)
        self.mask_pred = createMask(fov_scalar)
    
    def forwardSmall(self):
        # Resize all to 32x128
        self.mask_pred_s = self.resizeImage(self.mask_pred)
        #self.x = self.resizeImage(self.x) + ((self.mask_pred-1) * tf.random_normal([2,32,128,1],stddev=0.3))
        self.xs = self.resizeImage(self.x) 
        self.ys = self.resizeImage(self.y)

        # =============
        # Generator
        # =============
        self.gs = model.Gs2(self.xs)
        # =============
        # Discriminator Real
        # =============
        self.dys, self.dys_ = model.Ds_ml(model.dstack(self.xs,self.ys))
        self.lbl_real_s = tf.ones_like(self.dys) * self.soft_noise_real
        # =============
        # Discriminator Fake
        # =============
        self.dgs, self.dgs_ = model.Ds_ml(model.dstack(self.xs,self.gs), reuse=True) #Share weights for real-fake
        self.lbl_fake_s = tf.zeros_like(self.dgs) + self.soft_noise_fake 

    def forwardMed(self):
        self.xm = self.resizeImage(self.x,[64,256])
        self.ym = self.resizeImage(self.y,[64,256])
        self.gs_up = self.resizeImage(self.gs,[64,256])

        # Generator
        self.gm = model.Gm(self.xm)
        
        # Dreal
        self.dym = model.Dm(model.dstack(self.xm,self.ym))
        self.lbl_real = tf.ones_like(self.dym) * self.soft_noise_real

        # Dfake
        self.dgm = model.Dm(model.dstack(self.xm,self.gm), reuse=True)
        self.lbl_fake = tf.zeros_like(self.dgm) + self.soft_noise_fake

    def addSoftNoise(self, real_noise, fake_noise):
        self.lbl_fake = self.lbl_fake + fake_noise
        self.lbl_real = self.lbl_real * real_noise

    def lossFOV(self, a=1):   
        sigmoid_ent = tf.nn.sigmoid_cross_entropy_with_logits
        self.loss_fov = tf.reduce_mean(sigmoid_ent(logits=self.fov_pred, labels=self.fov_gt))
    
    def lossGANsmall(self, a_valid=10, a_hole=10):
        sigmoidEnt = tf.nn.sigmoid_cross_entropy_with_logits
        self.loss_ds_real = tf.reduce_mean(sigmoidEnt(logits=self.dys, labels=self.lbl_real))
        self.loss_ds_fake = tf.reduce_mean(sigmoidEnt(logits=self.dgs, labels=self.lbl_fake))
        self.loss_ds = self.loss_ds_real + self.loss_ds_fake
    
        self.loss_gs1 = tf.reduce_mean(sigmoidEnt(logits=self.dgs, labels=self.lbl_real))
        self.loss_hole_s = tf.reduce_mean(tf.abs((1 - self.mask_pred_s) * (self.ys-self.gs)))
        self.loss_valid_s = tf.reduce_mean(tf.abs(self.mask_pred_s * (self.ys-self.gs)))
        self.loss_gs = self.loss_gs1 + a_hole * self.loss_hole_s + a_valid * self.loss_valid_s 

    def lossGANmed(self, a_valid=10):
        sigmoidEnt = tf.nn.sigmoid_cross_entropy_with_logits
        self.loss_dm_real = tf.reduce_mean(sigmoidEnt(logits=self.dym, labels=self.lbl_real))
        self.loss_dm_fake = tf.reduce_mean(sigmoidEnt(logits=self.dgm, labels=self.lbl_fake))
        self.loss_dm = self.loss_dm_real + self.loss_dm_fake
    
        self.loss_gm1 = tf.reduce_mean(sigmoidEnt(logits=self.dgm, labels=self.lbl_real))
        self.loss_content_m = tf.reduce_mean(tf.abs(self.ym-self.gm))
        self.loss_gm = self.loss_gm1 + a_valid * self.loss_content_m

    def lossLSGAN(self, a_valid=10, a_hole=10):
        self.loss_d_real = tf.reduce_mean(tf.square(self.dy - self.lbl_real))
        self.loss_d_fake = tf.reduce_mean(tf.square(self.dg - self.lbl_fake))
        self.loss_d = 0.5 * (self.loss_d_real + self.loss_d_fake)
    
        self.loss_g1 = tf.reduce_mean(tf.square(self.dg - self.lbl_real))
        self.loss_hole = tf.reduce_mean(tf.abs((1 - self.mask_pred) * (self.y-self.g)))
        self.loss_valid = tf.reduce_mean(tf.abs(self.mask_pred * (self.y-self.g)))
        self.loss_g = 0.5 * (self.loss_g1) + a_hole * self.loss_hole + a_valid * self.loss_valid 

    def lossDecayFov(self, lr_fov):
        self.step_fov = tf.Variable(0, trainable=False)
        self.learn_rate_fov = tf.train.exponential_decay(lr_fov[0], self.step_fov, 
                                                         lr_fov[1], 0.90, staircase=True)
    def lossDecay(self, lr_d, lr_g):
        # Learning rate decay
        self.step_g = tf.Variable(0, trainable=False)
        self.step_d = tf.Variable(0, trainable=False)
        self.learn_rate_d = tf.train.exponential_decay(lr_d[0], self.step_d, 
                                                       lr_d[1], 0.90, staircase=True)
        self.learn_rate_g = tf.train.exponential_decay(lr_g[0], self.step_d,
                                                       lr_g[1], 0.90, staircase=True)

    def optimize(self, lr_fov, lr_d, lr_g):
        # LOSS DECAY
        self.lossDecayFov(lr_fov)
        #self.lossDecay(lr_d, lr_g)
        # Optimize
        # d_vars = [v for v in tf.trainable_variables() if v.name.startswith('Discriminator')]
        # g_vars = [v for v in tf.trainable_variables() if v.name.startswith('Generator')]
        # self.opt_gs = tf.train.AdamOptimizer(lr_g[0]).minimize(self.loss_gs,var_list=g_vars) # minimize only G
        # self.opt_ds = tf.train.AdamOptimizer(lr_d[0]).minimize(self.loss_ds, var_list=d_vars) # minimize only D
        self.opt_fov = tf.train.AdamOptimizer(self.learn_rate_fov).minimize(
            self.loss_fov, global_step=self.step_fov)

        d_vars_m = [v for v in tf.trainable_variables() if v.name.startswith('DiscriminatorMed')]
        g_vars_m = [v for v in tf.trainable_variables() if v.name.startswith('GeneratorMed')]
        self.opt_gm = tf.train.AdamOptimizer(lr_g[0]).minimize(self.loss_gm,var_list=g_vars_m)
        self.opt_dm = tf.train.AdamOptimizer(lr_d[0]).minimize(self.loss_dm,var_list=d_vars_m)

    def train(self, sess, graph_path, model_path, model_name, restore_model=None, restore=False):
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(graph_path, sess.graph)

        sess.run(self.iterator.initializer)

        if restore == True:
            saver.restore(sess, model_path + restore_model)
        if restore == False:
            sess.run(tf.global_variables_initializer())
        
        for epoch in range(self.epochs):
            sess.run(self.opt_fov)
            sess.run(self.opt_dm)
            sess.run(self.opt_gm)

            if epoch % self.print_range == 0:
                #summary, loss_d, loss_g = sess.run([self.summary, self.loss_d, self.loss_g])
                summary, loss_d, loss_g = sess.run([self.summary, self.loss_dm, self.loss_gm])
                writer.add_summary(summary, epoch)
                print('Step [%d/%d] : %g | %g *-*-* <= 0.5 |~ 1' 
                    %(epoch, self.epochs, loss_g, loss_d ))
                #print('Step [%d/%d] : %g | %g' %(epoch, self.epochs, loss_g, loss_d))
            
            if (epoch % self.save_range == 0) and (epoch > 0):
                print('Saving checkpoint........')
                saver.save(sess, model_path + model_name + str(epoch) + '.ckpt')
        
        writer.close()

    def test(self, sess, graph_path, model_path, model_name):
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(graph_path, sess.graph)

        sess.run(self.iterator.initializer)

        saver.restore(sess, (model_path + model_name))
        
        for epoch in range(self.epochs):

            if epoch % self.print_range == 0:
                summary, loss_d, loss_g = sess.run([self.summary, self.loss_d, self.loss_g])
                writer.add_summary(summary, epoch)
                print('Step [%d/%d] : %g | %g' %(epoch, self.epochs, loss_g, loss_d))
            
        
        writer.close()

    def test2(self, sess, graph_path, model_path, model_name, meta_name):
        
        saver = tf.train.import_meta_graph(model_path + meta_name)
        saver.restore(sess, model_path + model_name)



        writer = tf.summary.FileWriter(graph_path, sess.graph)



    def getPSNR(self):
        psnr = tf.reduce_mean(tf.image.psnr((self.g+1)/2, (self.y+1)/2, max_val = 1))
        return psnr

    def mergeSummary(self):
        self.addSummaryImage()
        self.addSummaryScalar()
        self.summary = tf.summary.merge_all()

    def addSummaryScalar(self):    
        with tf.name_scope('Generator'):
            tf.summary.scalar('AdvLoss', self.loss_gs1)
            tf.summary.scalar('AdvLoss', self.loss_gm1)
            tf.summary.scalar('HoleLoss', self.loss_hole_s)
            tf.summary.scalar('ValidLoss', self.loss_valid_s)
            tf.summary.scalar('Content', self.loss_content_m)
        with tf.name_scope('Discriminator'):
            tf.summary.scalar('FakeLoss', self.loss_ds_fake)
            tf.summary.scalar('FakeLoss', self.loss_dm_fake)
            tf.summary.scalar('RealLoss', self.loss_ds_real)
            tf.summary.scalar('RealLoss', self.loss_dm_real)
        with tf.name_scope('TotalLoss'):
            tf.summary.scalar('Discriminator', self.loss_ds)
            tf.summary.scalar('Discriminator', self.loss_dm)
            tf.summary.scalar('Generator', self.loss_gs)
            tf.summary.scalar('Generator', self.loss_gm)
            tf.summary.scalar('FOV', self.loss_fov)
           # tf.summary.scalar('PSNR',self.getPSNR())
        with tf.name_scope('LearningRate'):
            #tf.summary.scalar('D', self.learn_rate_d)
            #tf.summary.scalar('G', self.learn_rate_g)
            tf.summary.scalar('FOV', self.learn_rate_fov)

    def addSummaryImage(self):
        with tf.name_scope('Images'):
            tf.summary.image('Input', self.x, 10)
            tf.summary.image('Output', self.gs, 10)
            tf.summary.image('Output', self.gm, 10)
            tf.summary.image('GT', self.y,10)

    def checkRecords(self, x_, y_, fov):
        sess = self.sess

        sess.run(self.iterator.initializer)
        input_im, label_im, label_fov = sess.run([x_,y_,fov])
        return input_im, label_im, label_fov

    def verifyParameters(self, train_type):
        if train_type == '' or train_type == 'train':
            train = '============Training=============='
        elif train_type == 'test':
            train = '============Testing=============='
        print(train)
        print('Epochs :', self.epochs)
        print('Dataset :', self.dataset_path)
        input('Press enter to continue...')    

    def configTrain(self, epochs, save_range, print_step):
        self.epochs = epochs
        self.save_range = save_range
        self.print_range = print_step
        
    def resizeImage(self,x,size=[32,128]):
        return tf.image.resize_bilinear(x,size)