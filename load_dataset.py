import os
import cv2
import numpy as np
import sys

IMAGE_SIZE = 100
images = []
labels = []



def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    h, w, _ = image.shape
    longest_edge = max(h, w)

    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        top = dw // 2
        right = dw - left
    else:
        pass

    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    # cv2.imshow('1', image)
    # cv2.imshow('2', constant)
    # cv2.imshow('3', cv2.resize(constant, (100, 100)))
    #
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return cv2.resize(constant, (height, width))


def read_path(path_name):
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        if os.path.isdir(full_path):
            read_path(full_path)
            print(full_path)
        else:
            if dir_item.endswith('.jpg'):
                image = resize_image(cv2.imread(full_path), IMAGE_SIZE, IMAGE_SIZE)
                images.append(image)
                labels.append(os.path.split(os.path.dirname(full_path))[1])

    return images, labels


def load_dataset(path_name):
    images, labels = read_path(path_name)
    images = np.array(images)
    print(images.shape)

    for i in range(len(labels)):
        if labels[i] == 'zhuang':
            labels[i] = 0
        elif labels[i] == 'deng':
            labels[i] = 1
        elif labels[i] == 'others':
            labels[i] = 2
        else:
            labels[i] = 3

    labels = np.array(labels)

    print(labels)

    return images, labels


if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("you don't need input anything")
    else:
        images, labels = load_dataset("./data")
