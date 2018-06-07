import tensorflow as tf

path = '/home/enningxie/Documents/DataSets/butter_data/crop_img1/AAaa0001002_1_1.jpg'

if __name__ == '__main__':
    test_image = tf.read_file(path, 'test_name')
    print(type(test_image))