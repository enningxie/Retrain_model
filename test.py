import tensorflow as tf

model_path = '/home/enningxie/Documents/DataSets/trained_model/vgg_16_2016_08_28/vgg_16.ckpt'
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, model_path)
    tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    for tensor_name in tensor_name_list:
        print(tensor_name, '\n')
