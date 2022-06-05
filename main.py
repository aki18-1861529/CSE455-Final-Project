import argparse
import os

from PCA.pca import PCA
from PCA.pca import person


def main(file:str, train:bool):
    parser.add_argument('pos_arg', type=int,
    help='A required integer positional argument')
    assert file.endswith('.jpg')

    if train:
        training_data = os.path.abspath('./data/PCA/pca')
        pca = PCA(training_data)
        pca.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, required=True, help="Required location of input image")
    parser.add_argument('-t', '--train', action='store_true', help="If present, will train PCA")

    arg = parser.parse_args()

    main(arg.f, arg.t)