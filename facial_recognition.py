'''
Facial Recognition
'''

import glob
import cv2
import numpy as np
from PIL import Image


def im2double(im):
    return im.astype('float') / (np.iinfo(im.dtype)).max


def avg_images():
    """
    Gets all jpg images from the imgs and converts them to greyscale, resizes
    them to be m x n, and calculates their averages. Returns a list that
    contains the sum, the vector of averages, and the number of images.
    """
    m = 168
    n = 192

    img_dir = './data/imgs/*.jpg'
    imgs = []
    for filename in glob.glob(img_dir):
        im = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_RGB2GRAY)
        imgs.append(im)

    sum = np.zeros((n, m))
    for i in range(len(imgs)):
        im = cv2.resize(im2double(imgs[i]), [m, n])
        sum = np.add(sum, im)

    avg = sum / len(imgs)
    x = np.reshape(avg, (n * m, 1))
    count = len(imgs)

    return [sum, x, count]


def main():
    res = avg_images()
    print(res[0])
    print(res[1])
    print(res[2])


if __name__ == '__main__':
    main()
