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
import os
import tensorflow as tf
import numpy as np
from read_box import get_tiny_image

parser = argparse.ArgumentParser()
parser.add_argument(
    '--image', default='/var/Data/xz/butterfly/crop_img1', type=str, help='Absolute path to image file.')
parser.add_argument(
    '--num_top_predictions',
    type=int,
    default=1,
    help='Display this many predictions.')
parser.add_argument(
    '--graph',
    default='/var/Data/xz/butterfly/trained_models/last/output_graph.pb',
    type=str,
    help='Absolute path to graph file (.pb)')
parser.add_argument(
    '--labels',
    default='/var/Data/xz/butterfly/trained_models/last/output_labels.txt',
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
    default='DecodeJpeg/contents:0',
    help='Name of the input operation')


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def load_image(filename):
  """Read in the image_data to be classified."""
  return tf.gfile.FastGFile(filename, 'rb').read()


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]

def load_labels_(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def run_graph(image_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  with tf.Session() as sess:
    # Feed the image_data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    # sess.run(tf.global_variables_initializer())
    preds = []
    logits = []
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    for image in image_data:
        predictions, = sess.run(softmax_tensor, {input_layer_name: image})
        preds.append(predictions)

    # Sort to show labels in order of confidence
    for predictions in preds:
        top_k = predictions.argsort()[-num_top_predictions:][::-1]
        logits.append(top_k[0])

    return logits, preds


def main(_):
  """Runs inference on an image."""

  if not tf.gfile.Exists(FLAGS.image):
    tf.logging.fatal('image file does not exist %s', FLAGS.image)

  if not tf.gfile.Exists(FLAGS.labels):
    tf.logging.fatal('labels file does not exist %s', FLAGS.labels)

  if not tf.gfile.Exists(FLAGS.graph):
    tf.logging.fatal('graph file does not exist %s', FLAGS.graph)


  # load labels
  labels = load_labels_(FLAGS.labels)
  # load image
  image_data = []
  file_names = []
  # true_y = []
  # for image in os.listdir(FLAGS.image):
  #   true_y.append(labels.index(image[:11].lower()))
  #   image_data.append(load_image(os.path.join(FLAGS.image, image)))
  file_names, image_data = get_tiny_image()
  # for image in os.listdir(FLAGS.image):
  #   file_names.append(image)
  #   # true_y.append(labels.index(image[:11].lower()))
  #   image_data.append(load_image(os.path.join(FLAGS.image, image)))



  # load graph, which is stored in the default session
  load_graph(FLAGS.graph)

  logits, preds = run_graph(image_data, labels, FLAGS.input_layer, FLAGS.output_layer,
            FLAGS.num_top_predictions)

  # print(len(true_y))
  # count = 0
  # for i in range(len(true_y)):
  #     if true_y[i] != logits[i]:
  #         count += 1
  # print(count)
  with open('./last_result.txt', 'a') as f:
    for file_name, label_index in zip(file_names, logits):
      str_ = file_name + ' ' + labels[label_index] + '\n'
      f.write(str_)
  #mAP = mapk([true_y], [logits], k=10)
  #print(mAP)


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
