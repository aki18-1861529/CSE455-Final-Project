import argparse
import os
import numpy as np

from PCA.pca import PCA
from PCA.pca import person


def main(file:str, train:bool):
    parser.add_argument('pos_arg', type=int,
    help='A required integer positional argument')
    if file is not None:
        assert file.endswith('.jpg')

    training_data = os.path.abspath('./data/PCA/allFaces')
    if train:
        pca = PCA(training_data)
        pca.train()
    else:
        UrT = np.load('UrT.npy')
        U = np.load('U.npy')
        avg = np.load('avg_face_vector.npy')
        pca = PCA(training_data, U, UrT, avg)
    
    """DETR:
    Take image
    run through detr
    get bounding box (points)
    crop image
    """

    """ Load:
    loop of people: 
        take faces,
        find avg,
        find alpha of avg,
        add person
    """
    pca.add_person(0, './data/imgs/')


    """ Prediction

    take face
    find alpha
    compare
    """
    pca.predict('./data/d13.jpg')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('-f', type=str, required=True, help="Required location of input image")
    parser.add_argument('-f', type=str, help="Required location of input image")
    parser.add_argument('-t', '--train', action='store_true', help="If present, will train PCA")

    arg = parser.parse_args()
    main(arg.f, arg.train)