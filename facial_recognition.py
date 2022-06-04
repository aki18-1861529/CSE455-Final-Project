'''
Facial Recognition
'''

import cv2
import numpy as np
from PIL import Image
import glob

def im2double(im):
    return im.astype(np.float) / (np.iinfo(im.dtype)).max

'''
Gets all jpg images from the imgs and converts them to greyscale, resizes
them to be m x n, and calculates their averages. Returns a list that contains
the sum, the vector of averages, and the number of images.
'''
def avg_images():
    m = 168
    n = 192

    img_dir = '../data/imgs/*.jpg'
    imgs = []
    for filename in glob.glob(img_dir):
        im = Image.open(filename).convert('L')
        imgs.append(im)

    sum = np.zeros(n, m)
    for i in imgs:
        im = cv2.resize(im2double(cv2.imread(i)), [n, m])
        sum = sum + im

    avg = sum / len(imgs)
    x = np.reshape(avg, (n * m, 1))
    count = len(imgs)

    return [sum, x, count]

def main():
    print('Facial Recognition')

if __name__ =='__main__':
    main()
