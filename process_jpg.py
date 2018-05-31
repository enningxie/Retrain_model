import os
import cv2
import argparse
import shutil

FLAGS = None


def parser():
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', default='/home/enningxie/Documents/DataSets/butterfly_/butt_train/train')
    args.add_argument('--data_jpg_path', default='/home/enningxie/Documents/DataSets/butterfly_/butt_train/train_jpg')
    args.add_argument('--data_rename_path', default='/home/enningxie/Documents/DataSets/butterfly_/butt_train/train_rename')
    args.add_argument('--data_png_path', default='/home/enningxie/Documents/DataSets/butterfly_/butt_train/train_png')
    return args.parse_args()



def convert_png_to_jpg(path, to_path):
    for filename in os.listdir(path):
        if os.path.splitext(filename)[-1] == '.png':
            # print(filename)
            img = cv2.imread(os.path.join(path, filename))
            # img = Image.open(path + '\\' + filename)
            # # print(filename.replace(".jpg", ".png"))
            newfilename = filename.replace(".png", ".jpg")
            # img.save(to_path + '\\' + newfilename)
            # # cv2.imshow("Image",img)
            # # cv2.waitKey(0)
            cv2.imwrite(os.path.join(to_path, newfilename), img)
    print('convert done.')


def move_png_to(source_path, to_path):
    for dir_name in os.listdir(source_path):
        if dir_name[-3:] == 'png':
            shutil.move(os.path.join(source_path, dir_name), os.path.join(to_path, dir_name))


def rename_files(path, to_path):
    for dir_name in os.listdir(path):
        src_name = os.path.join(path, dir_name)
        dst_name = os.path.join(to_path, dir_name[:11] + '_p_' + dir_name[11:])

if __name__ == '__main__':
    FLAGS = parser()
    # move_png_to(FLAGS.data_path, FLAGS.data_png_path)
    # convert_png_to_jpg(FLAGS.data_png_path, FLAGS.data_jpg_path)
