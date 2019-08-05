import os, sys, math, random
import tensorflow as tf
from net2 import Network2

tf.reset_default_graph()

train_data_200='/home/juliussurya/workspace/360pano/tfrecords/pano_FOVrand_200_outdoor'
train_data_full = '/home/juliussurya/workspace/360pano/tfrecords/pano_FOVrand_train_outdoor' 
val_data = '/home/juliussurya/workspace/360pano/tfrecords/pano_FOVrand_val_outdoor' 
test_data = '/home/juliussurya/workspace/360pano/tfrecords/pano_FOVrand_test_outdoor' 
graph_path = '/home/juliussurya/workspace/360pano/graph/fov/enc_dec_flow_gan'
model_path = '/home/juliussurya/workspace/360pano/checkpoint/'
output_path = '/home/juliussurya/workspace/360pano/output/'

iter_small = 300000
iter_med = 300000
iter_high = 300000

if sys.argv[1] == 'train':
    dataset = train_data_full
elif sys.argv[1] == 'test':
    dataset = val_data

net = Network2(dataset)

# TF Configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    net.buildFullPano(sess, graph_path, model_path, 'v3.1_pano_', 'v3.0_fromM_high_300000')
    net.trainFullPano(100000)
    
    #net.buildMedfromSmall(sess,graph_path, model_path, 'v3.0_fromS_', 'v3.0_small_300000')
    #net.trainMedfromSmall(iter_med)
    # net.buildHighfromMed(sess,graph_path, model_path, 'v3.0_fromM', 'v3.0_fromS__medium_300000')
    # net.trainHighfromMed(iter=iter_high)

    #net.buildHigh(sess, graph_path, model_path, 'v3.0_test_eval', 'v3.0_fromM_high_300000')
    #net.testHigh(200,output_path+'test_full/')

    # if sys.argv[1] == 'train':
    #     net.build(sess, graph_path, model_path, 'v3.0_test')
    #     net.verify()
    #     net.train(400, iter_med, iter_high, True)
    # if sys.argv[1] == 'test':
    #     net.buildfromSmall(sess, graph_path, model_path, 'v3.0_pre_s', 'v3.0_small_200000.meta')
    #     net.verify()
    #     net.testSmall(10)


