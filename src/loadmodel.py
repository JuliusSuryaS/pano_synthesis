import tensorflow as tf
import util 
import modlayers as ml

# train_data_200='/home/juliussurya/workspace/360pano/tfrecords/pano_FOVrand_200_outdoor'
# graph_path = '/home/juliussurya/workspace/360pano/graph/fov/enc_dec_flow_gan'
# model_path = '/home/juliussurya/workspace/360pano/checkpoint/'

train_data_200='/home/juliussurya/workspace/360pano/tfrecords/pano_FOVrand_200_outdoor'
train_data_full = '/home/juliussurya/workspace/360pano/tfrecords/pano_FOVrand_train_outdoor' 
val_data = '/home/juliussurya/workspace/360pano/tfrecords/pano_FOVrand_val_outdoor' 
test_data = '/home/juliussurya/workspace/360pano/tfrecords/pano_FOVrand_test_outdoor' 
graph_path = '/home/juliussurya/workspace/360pano/graph/fov/enc_dec_flow_gan'
model_path = '/home/juliussurya/workspace/360pano/checkpoint/'



# Model

tf.reset_default_graph()

saver = tf.train.import_meta_graph(model_path+'v3.0_fromS__medium_300000.meta')
graph = tf.get_default_graph()
ops = [v.name for v in graph.get_operations() if 'x_in' in v.name]
print(ops)

# x = tf.placeholder(tf.float32,shape=[None,32,32,3])
# with tf.variable_scope('Generator'):
#     out = ml.leaky(ml.convLyr(x,1,name='conv1'))
#     out = ml.leaky(ml.convLyr(out,1,name='conv2'))


# vars_dict = {}
# vars_to_save = [v for v in tf.global_variables() if 'Generator' in v.name]
# for v in vars_to_save:
#     vars_dict[v.name[:-2]] = v

# print(vars_dict)
# saver = tf.train.Saver(vars_dict)



# tf.reset_default_graph()

# iterator = util.initDataset(train_data_200,1,10)
# getDataset = iterator.get_next()

# x_in = tf.placeholder(tf.float32,shape=[None,256,256,18],name='input_x')
# y_in = tf.placeholder(tf.float32,shape=[None,256,256,18],name='input_y')
# fov_in = tf.placeholder(tf.int64, shape=[None,128],name='im_fov')

# x = util.catResize(x_in)
# y = util.catResize(y_in)
# fov_gt = fov_in

# def model(x):
#     with tf.variable_scope('model1'):
#         conv1 = op.sigmoid(op.convLyr(x,32,name='conv1'))
#         conv1 = op.sigmoid(op.convLyr(conv1,64,name='conv2'))
#     return conv1



# saver = tf.train.import_meta_graph(model_path+'test.meta')
# graph = tf.get_default_graph()
# opsname = [v.name for v in graph.get_operations()]
# print(opsname)

# load_x = graph.get_tensor_by_name('im_x:0')
# out_conv = graph.get_tensor_by_name('model1/conv1/Conv2D:0')

# with tf.Session(graph=graph) as sess:
#     saver.restore(sess,model_path+'test')
#     sess.run(iterator.initializer)
#     tr_x, tr_y, tr_fov = sess.run(getDataset)

#     output = sess.run(out_conv, feed_dict={load_x:tr_x})
#     print(output.shape)

# out_lyr = model(x)

# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(iterator.initializer)
#     tr_x, tr_y, tr_fov = sess.run(getDataset)
#     sess.run(out_lyr, feed_dict={x_in: tr_x, y_in: tr_y, fov_in: tr_fov})
#     saver.save(sess, model_path + 'test')




# model_path = '/home/juliussurya/workspace/360pano/checkpoint/'
# model_name = tf.train.import_meta_graph(model_path + 'stable_full_1_290000.ckpt.meta')


# graph = tf.get_default_graph()
# opsname = [v.name for v in graph.get_operations() if v.name.startswith('FoVNet/logistic')]
# print(opsname)

# # Import then run these
# inputs = graph.get_tensor_by_name('ReadDataset/ResizeBilinear')
# gs = graph.get_tensor_by_name('Generator/convT_9/conv2d_transpose:0')

# # sess.run(gs, feed_dict={inputs:<MYIMAGE>})


# graph = tf.get_default_graph()
# filt = graph.get_tensor_by_name('Generator/convT/Conv2D:0')
# print(filt)
# filt = graph.get_tensor_by_name('Generator/convT_1/Conv2D:0')
# print(filt)
# print(filt)
