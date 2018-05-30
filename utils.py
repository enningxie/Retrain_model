import os
import argparse
from collections import defaultdict
from sklearn.model_selection import train_test_split
import shutil
import numpy as np


FLAGS = None


def parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--test_path', default='/home/enningxie/Documents/DataSets/test')
    argparser.add_argument('--test_set_path', default='/home/enningxie/Documents/DataSets/data_augmentation_13/amxx0001003')
    argparser.add_argument('--data_test_path', default='/var/Data/xz/butterfly/data_augmentation_14_test')
    argparser.add_argument('--data_path', default='/var/Data/xz/butterfly/data_augmentation_14', type=str)
    return argparser.parse_args()


def mkdir_op(source_name):
    if not os.path.exists(source_name):
        os.mkdir(source_name)
        # print('done.')


def copy_op(source_path):
    data_list = os.listdir(source_path)
    for data in data_list:
        new_name = '1' + data
        shutil.copy(os.path.join(source_path, data), os.path.join(source_path, new_name))




def train_test_split_(data_path, test_set_path):
    data_list = np.asarray(os.listdir(data_path))
    label_list = []
    for data in data_list:
        label_list.append(data[:11])
    label_list_ = np.asarray(label_list)
    X_train, X_test, y_train, y_test = train_test_split(data_list, label_list_, test_size=0.2, random_state=42)
    # print(X_train, X_test, y_train, y_test)
    for x_test in X_test:
        shutil.move(os.path.join(data_path, x_test), os.path.join(test_set_path, x_test))


def construct_folders(data_path):
    for data in os.listdir(data_path):
        mkdir_op(os.path.join(data_path, data[:11].lower()))
    for dir_name in os.listdir(data_path):
        if not os.path.isdir(os.path.join(data_path, dir_name)):
            shutil.move(os.path.join(data_path, dir_name), os.path.join(os.path.join(data_path, dir_name[:11].lower()), dir_name))

if __name__ == '__main__':
    FLAGS = parser()
    train_test_split_(FLAGS.data_path, FLAGS.data_test_path)
    construct_folders(FLAGS.data_path)


