import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from PCA.pca import PCA
from PCA.pca import person


def main( train:bool):
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
    # Cropped Image directories:
    cropped_data = dict()
    cropped_data['ryan'] = './data/detr_output/cropped_rr'
    cropped_data['dwayne'] = './data/detr_output/cropped_dj'
    cropped_data['tom'] = './data/detr_output/cropped_ts'
    cropped_data['zendaya'] = './data/detr_output/cropped_zc'
    """ Load:
    loop of people: 
        take faces,
        find avg,
        find alpha of avg,
        add person
    """
    for i, (name, dir) in enumerate(cropped_data.items()):
        train_dir = dir + "/train/"
        pca.add_person(i, train_dir, name)


    """ Prediction
    take face
    find alpha
    compare
    """
    num_right = 0
    num_wrong = 0
    for i, (name, dir) in enumerate(cropped_data.items()):
        test_dir = dir + "/test/"
        for filename in os.listdir(test_dir):
            if not filename.endswith(".jpg"):
                continue
            img_path = os.path.join(test_dir, filename)
            id, dist, img = pca.predict(img_path)
            if id == i:
                print('Matched!')
                num_right += 1
            else:
                print("Incorrect match :(")
                print("   Euc Dist:", dist)
                print("   Guessed", pca.persons[id].name, "should have been", name)
                pca.vector_show(img)
                num_wrong += 1
    print("Number right:", num_right)
    print("Num_wrong:", num_wrong)

    '''Show points in 3d:
    fig = plt.figure()
    
    # syntax for 3-D projection
    ax = plt.axes(projection ='3d')
    
    # defining all 3 axes
    z = np.linspace(0, 1, 100)
    x = z * np.sin(25 * z)
    y = z * np.cos(25 * z)
    
    # plotting
    ax.plot3D(x, y, z, 'green')
    ax.set_title('3D line plot geeks for geeks')
    plt.show()'''

    """2d plot"""
    '''for i, (name, dir) in enumerate(cropped_data.items()):
        test_dir = dir + "/test/"
        for filename in os.listdir(test_dir):
            if not filename.endswith(".jpg"):
                continue'''






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('-f', type=str, required=True, help="Required location of input image")
    parser.add_argument('-t', '--train', action='store_true', help="If present, will train PCA")

    arg = parser.parse_args()
    main(arg.train)