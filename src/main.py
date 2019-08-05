import os, sys, math, random
from net import Network
import tensorflow as tf
import matplotlib.pyplot as plt

tf.reset_default_graph()

train_data_200='/home/juliussurya/workspace/360pano/tfrecords/pano_FOVrand_200_outdoor'
train_data_full = '/home/juliussurya/workspace/360pano/tfrecords/pano_FOVrand_train_outdoor' 
val_data = '/home/juliussurya/workspace/360pano/tfrecords/pano_FOVrand_val_outdoor' 
test_data = '/home/juliussurya/workspace/360pano/tfrecords/pano_FOVrand_test_outdoor' 
graph_path = '/home/juliussurya/workspace/360pano/graph/fov/enc_dec_flow_gan'
model_path = '/home/juliussurya/workspace/360pano/checkpoint/'

if sys.argv[1] == 'train':
    dataset = train_data_200
elif sys.argv[1] == 'test':
    dataset = val_data

with tf.name_scope('ReadDataset'):
    net = Network()
    x, y, fov = net.readDataset(dataset,2,50) #return output for checking

with tf.name_scope('FoVNet'):
    net.forwardFOV() # Run FOV network estimation
    net.lossFOV() # Compute loss FOV

net.forwardSmall() # Run single model network
net.forwardMed()
net.addSoftNoise(random.uniform(0.8,1.1),random.uniform(0.0,0.4)) #Add label noise
net.lossGANsmall()
net.lossGANmed()

with tf.name_scope('Minimizer'):
    # Learning rate , decay step
    lr_fov = [0.0001, 5000]
    lr_g = [0.0002, 20000]
    lr_d = [0.0003, 20000]
    net.optimize(lr_fov, lr_d, lr_g) # Minimize loss fov and synthesis

net.mergeSummary() # Gather all summary

# TF Configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Training configuration
net.configTrain(epochs=300000, print_step=200, save_range=10000)

# Check before training
net.verifyParameters(sys.argv[1])

# Train the network
with tf.Session(config=config) as sess:
    # TRAIN
    if sys.argv[1] == 'train' or sys.argv[1] == '':
        net.train(sess, graph_path, model_path, 'stable_full_2_', 
                  restore_model='stable_full_1_290000.ckpt',restore=False)

    # TEST
    if sys.argv[1] == 'test':
        net.epochs = 10000
        net.test(sess, graph_path, model_path, 'stable_full_1_290000.ckpt')
