import tensorflow as tf
import os
import numpy as np
import cv2


def input_data():
    filename = '/var/Data/xz/butterfly/last.txt'
    name = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    with open(filename, 'r') as file_to_read:

        while True:
            lines = file_to_read.readline()  # 整行读取数据
            if not lines:
                break
            name_1, _, xmin_1, ymin_1, xmax_1, ymax_1 = [i for i in lines.split()]
            name.append(name_1)
            xmin.append(float(xmin_1))
            ymin.append(float(ymin_1))
            xmax.append(float(xmax_1))
            ymax.append(float(ymax_1))
    name = np.array(name)
    xmin = np.array(xmin)
    ymin = np.array(ymin)
    xmax = np.array(xmax)
    ymax = np.array(ymax)
    # print(name)
    return name, xmin, ymin, xmax, ymax


def box_image(name, xmin, ymin, xmax, ymax):
    image_dir = '/var/Data/xz/butterfly/last_test'
    # to_path = '/var/Data/xz/butterfly/saveImages'
    name_2 = ''
    image_data = []
    for name_1, xmin_1, ymin_1,xmax_1,ymax_1 in zip(name, xmin, ymin, xmax, ymax):
        if name_1 != name_2:
            name_2 = name_1
            image_path = os.path.join(image_dir, name_1 + '.jpg')
            image = cv2.imread(image_path)
            image_cp = image[int(ymin_1):int(ymax_1), int(xmin_1):int(xmax_1)]
            img_str = cv2.imencode('.jpg', image_cp)[1].tostring()
            # print(type(img_str))
            # cv2.imwrite(os.path.join(to_path, name_1 + '.jpg'), image_cp)
            image_data.append(img_str)
    return image_data


def get_tiny_image():
    n, x1, y1, x2, y2 = input_data()
    return n, box_image(n, x1, y1, x2, y2)


if __name__ == '__main__':
    names, image_data = get_tiny_image()


