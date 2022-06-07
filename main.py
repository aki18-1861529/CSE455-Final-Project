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
    test_alphas = dict()
    for i in cropped_data:
        test_alphas[i] = []

    for i, (name, dir) in enumerate(cropped_data.items()):
        test_dir = dir + "/test/"
        for filename in os.listdir(test_dir):
            if not filename.endswith(".jpg"):
                continue
            img_path = os.path.join(test_dir, filename)
            id, dist, img = pca.predict(img_path)
            test_alphas[name].append(pca.compute_alpha(img))
            if id == i:
                print('Matched!')
                num_right += 1
            else:
                print("Incorrect match :(")
                print("   Euc Dist:", dist)
                print("   Guessed", pca.persons[id].name, "should have been", name)
                #pca.vector_show(img)
                num_wrong += 1
    print("Number right:", num_right)
    print("Num_wrong:", num_wrong)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("Principal Components of Images")
    ax.set_xlabel("9th PC")
    ax.set_ylabel("10th PC")
    ax.set_zlabel("11th PC")

    colors = ['green', 'red', 'blue', 'orange']
    e1=9
    e2=10
    e3=8
    for i, person in enumerate(pca.persons.values()):
        a = person.alpha
        ax.scatter3D([a[e1,0]], [a[e2,0]],[a[e3,0]], marker="o",color=colors[i],
            label=person.name+' avg')

    for i, (name, alphas) in enumerate(test_alphas.items()):
        for k, a in enumerate(alphas):
            if k == 0:
                ax.scatter3D([a[e1,0]], [a[e2, 0]], [a[e3,0]], marker="^", color=colors[i], label=name+ ' test')
            else:
                ax.scatter3D([a[e1,0]], [a[e2, 0]], [a[e3,0]], marker="^", color=colors[i])
    '''for i, person in enumerate(pca.persons.values()):
        a = person.alpha
        plt.plot([a[e1,0]], [a[e2,0]], marker="o",color=colors[i],
            label=person.name+' Avg')

    for i, (name, alphas) in enumerate(test_alphas.items()):
        for a in alphas:
            plt.plot([a[e1,0]], [a[e2, 0]], marker="^", color=colors[i], label=name+' test')'''
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.show()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('-f', type=str, required=True, help="Required location of input image")
    parser.add_argument('-t', '--train', action='store_true', help="If present, will train PCA")

    arg = parser.parse_args()
    main(arg.train)