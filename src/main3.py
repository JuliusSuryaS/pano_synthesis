import os, sys, math, random
import tensorflow as tf
from net3 import Network

tf.reset_default_graph()

train_data_200='/home/juliussurya/workspace/360pano/tfrecords/pano_FOVrand_200_outdoor'
train_data_full = '/home/juliussurya/workspace/360pano/tfrecords/pano_FOVrand_train_outdoor'
val_data = '/home/juliussurya/workspace/360pano/tfrecords/pano_FOVrand_val_outdoor'
test_data = '/home/juliussurya/workspace/360pano/tfrecords/pano_FOVrand_test_outdoor'
graph_path = '/home/juliussurya/workspace/360pano/graph/fov/enc_dec_flow_gan'
model_path = '/home/juliussurya/workspace/360pano/checkpoint/'
output_path = '/home/juliussurya/workspace/360pano/output/'

dataset = None
if sys.argv[1] == 'train':
    dataset = train_data_full
elif sys.argv[1] == 'train200':
    dataset = train_data_200
elif sys.argv[1] == 'test' or sys.argv[1] == 'testfull':
    dataset = val_data

net = Network(dataset, batch_sz=2, shuffle_buff=50)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

iterations = 300000 + 1  # type: int
with tf.Session(config=config) as sess:
    if sys.argv[1] == 'train' and sys.argv[2] == 'fov':
        net.buildNetworkFoV(sess, graph_path, model_path, 'fov_trained_')
        net.trainNetworkFov(iterations)
    # Small
    if sys.argv[1] == 'train' and sys.argv[2] == 'small':
        net.buildNetworkSmall(sess, graph_path, model_path, 'v3.4_small_')
        net.trainNetworkSmall(iterations,False)

    if sys.argv[1] == 'train' and sys.argv[2] == 'vsmall':
        net.buildNetworkVert(sess, graph_path, model_path, 'v3.5_v_small_', 'v3.3noise_high_300000', 'small')
        net.trainNetworkVert(iterations)

    # Medium
    if sys.argv[1] == 'train' and sys.argv[2] == 'medium':
        net.buildNetworkMed(sess, graph_path, model_path,
                            'v3.4_medium_', 'v3.4_small_210000') # typo saving name
        net.trainNetworkMed(iterations)

    if sys.argv[1] == 'train' and sys.argv[2] == 'vmedium':
        net.buildNetworkVert(sess, graph_path, model_path, 'v3.5_v_medium_', 'v3.5_v_small_130000', 'medium')
        net.trainNetworkVert(iterations)

    # High
    if sys.argv[1] == 'train' and sys.argv[2] == 'high':
        net.buildNetworkHigh(sess, graph_path, model_path,
                             'v3.4_high_', 'v3.4_medium_250000')
        net.trainNetworkHigh(iterations)

    if sys.argv[1] == 'train' and sys.argv[2] == 'vhigh':
        net.buildNetworkVert(sess, graph_path, model_path, 'v3.5_v_high3_', 'v3.5_v_high2_300000', 'high')
        # net.buildNetworkVert(sess, graph_path, model_path, 'v3.5_v_high2_', 'v3.5_v_high_300000', 'high')
        net.trainNetworkVert(iterations)

    # Refine
    if sys.argv[1] == 'train' and sys.argv[2] == 'refine':
        net.buildNetworkRef(sess, graph_path, model_path, 'v3.4ref_', 'v3.3noise_high_300000')
        net.trainNetworkRef(iterations)

    if sys.argv[1] == 'train' and sys.argv[2] == 'noise':
        net.buildNetworkNoise(sess, graph_path, model_path, 'vnoise_')
        net.trainNetworkNoise(iterations)

    # Test
    if sys.argv[1] == 'test':
        net.buildNetworkTest(sess, graph_path, model_path, 'v3.4_small_300000')
        net.runTestNetwork(300, output_path + 'v3.4/')

    if sys.argv[1] == 'testfull':
        net.buildNetworkTestFull(sess, graph_path, model_path, 'v3.5_v_high2_300000')
        net.runTestNetwork(500, output_path + 'v3.5/', net_type='inference')
