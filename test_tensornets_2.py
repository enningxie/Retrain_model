import tensornets as nets
import tensorflow as tf


inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
outputs = tf.placeholder(tf.float32, [None, 50])
model = nets.DenseNet169(inputs, is_training=True, classes=50)

loss = tf.losses.softmax_cross_entropy(outputs, model)
train = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

with tf.Session() as sess:
    nets.pretrained(model)
    print('oops.')