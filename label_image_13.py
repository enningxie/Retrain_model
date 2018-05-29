# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simple image classification with Inception.
Run image classification with your model.
This script is usually used with retrain.py found in this same
directory.
This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. You are required
to pass in the graph file and the txt file.
It outputs human readable strings of the top 5 predictions along with
their probabilities.
Change the --image_file argument to any jpg image to compute a
classification of that image.
Example usage:
python label_image.py --graph=retrained_graph.pb
  --labels=retrained_labels.txt
  --image=flower_photos/daisy/54377391_15648e8d18.jpg
NOTE: To learn to use this file and retrain.py, please see:
https://codelabs.developers.google.com/codelabs/tensorflow-for-poets
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument(
    '--image', default='/home/enningxie/Documents/DataSets/flower_photos_testset/tulip.jpg', type=str, help='Absolute path to image file.')
parser.add_argument(
    '--num_top_predictions',
    type=int,
    default=5,
    help='Display this many predictions.')
parser.add_argument(
    '--graph',
    default='/home/enningxie/Documents/DataSets/trained_model/butterfly/output_graph.pb',
    type=str,
    help='Absolute path to graph file (.pb)')
parser.add_argument(
    '--labels',
    default='/home/enningxie/Documents/DataSets/trained_model/butterfly/output_labels.txt',
    type=str,
    help='Absolute path to labels file (.txt)')
parser.add_argument(
    '--output_layer',
    type=str,
    default='final_result:0',
    help='Name of the result operation')
parser.add_argument(
    '--input_layer',
    type=str,
    default='Placeholder:0',
    help='Name of the input operation')


def load_image(filename):
  """Read in the image_data to be classified."""
  return tf.gfile.FastGFile(filename, 'rb').read()

def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)
  return result


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def run_graph(graph_cur, image_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  with tf.Session(graph=graph_cur) as sess:
    # Feed the image_data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))

    return 0


def fix_graph_def(graph_def):
    # fix nodes
    for node in graph_def.node:
        if node.op == 'RefSwitch':
            print('+++++')
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            print('++')
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                print('++++++++++++++++++++++++++++++++++++')
                del node.attr['use_locking']
        if "dilations" in node.attr:
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            del node.attr["dilations"]
        if "index_type" in node.attr:
            print('++++++++++++++++++++++++++++++++++++++++++++')
            del node.attr["index_type"]


def main(_):
  """Runs inference on an image."""

  if not tf.gfile.Exists(FLAGS.image):
    tf.logging.fatal('image file does not exist %s', FLAGS.image)

  if not tf.gfile.Exists(FLAGS.labels):
    tf.logging.fatal('labels file does not exist %s', FLAGS.labels)

  if not tf.gfile.Exists(FLAGS.graph):
    tf.logging.fatal('graph file does not exist %s', FLAGS.graph)

  # load image
  image_data = load_image(FLAGS.image)
  t = read_tensor_from_image_file(
      FLAGS.image,
      input_height=299,
      input_width=299,
      input_mean=0,
      input_std=255)

  # load labels
  labels = load_labels(FLAGS.labels)

  # load graph, which is stored in the default session
  load_graph(FLAGS.graph)



  # fix_graph_def(graph_cur.as_graph_def())

  for node in tf.get_default_graph().as_graph_def().node:
      if "dilations" in node.attr:
          print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
          del node.attr["dilations"]
  for node in tf.get_default_graph().as_graph_def().node:
      print(node.attr)
      if "dilations" in node.attr:
          print('--------------------------------')
          del node.attr["dilations"]

  with tf.Session() as sess:
    # Feed the image_data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(FLAGS.output_layer)
    predictions, = sess.run(softmax_tensor, {FLAGS.input_layer: t})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))

  # run_graph(graph_cur, t, labels, FLAGS.input_layer, FLAGS.output_layer,
  #           FLAGS.num_top_predictions)

  # tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
  # for tensor_name in tensor_name_list:
  #     print(tensor_name, '\n')
  #  and

  # with tf.Session() as sess:
  #     print(t.shape)



if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)