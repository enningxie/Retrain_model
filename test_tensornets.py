import tensornets as nets
import tensorflow as tf
import argparse
import os

FLAGS = None


def parser():
    args = argparse.ArgumentParser()
    args.add_argument('--data_path',
                      default='/home/enningxie/Documents/DataSets/data_augmentation_13')
    args.add_argument('--label_list_path',
                      default='/home/enningxie/Documents/DataSets/trained_model/butterfly_14/output_labels.txt')
    return args.parse_args()


def create_train_data_labels(data_path, label_list_path):
    train_data = []
    train_labels = []
    label_list = load_labels(label_list_path)
    for dir_name in os.listdir(data_path):
        dir_path = os.path.join(data_path, dir_name)
        if os.path.isdir(dir_path):
            for pic_name in os.listdir(dir_path):
                pic_path = os.path.join(dir_path, pic_name)
                train_data.append(pic_path)
                train_labels.append(label_list.index(dir_name))
    return train_data, train_labels


def load_labels(filename):
    """Read in labels, one label per line."""
    return [line.rstrip() for line in tf.gfile.GFile(filename)]


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [224, 224])
    return image_resized, label


def create_dataSet(data, labels):
    data = tf.constant(data)
    labels = tf.constant(labels)
    dataset = tf.contrib.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(32)
    return dataset


def create_iterator(dataset):
    iterator = tf.contrib.data.Iterator.from_structure(dataset.output_types,
                                                       dataset.output_shapes)
    next_element = iterator.get_next()
    training_init_op = iterator.make_initializer(dataset)
    return next_element, training_init_op


if __name__ == '__main__':
    FLAGS = parser()
    data_path = None
    label_list_path = None
    if FLAGS.data_path:
        data_path = FLAGS.data_path
    if FLAGS.label_list_path:
        label_list_path = FLAGS.label_list_path

    train_data, train_labels = create_train_data_labels(data_path, label_list_path)
    dataset = create_dataSet(train_data, train_labels)
    (img, label), training_init_op = create_iterator(dataset)
    img = tf.reshape(img, shape=[-1, 224, 224, 3])
    model = nets.DenseNet169(img, is_training=True, classes=94)
    loss = tf.losses.sparse_softmax_cross_entropy(label, model)
    train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        nets.pretrained(model)
        for i in range(10):
            sess.run(training_init_op)
            while True:
                try:
                    _, loss_ = sess.run([train, loss])
                    print('loss: {0}.'.format(loss_))
                except tf.errors.OutOfRangeError:
                    print('epoch: {0} over.'.format(i))


