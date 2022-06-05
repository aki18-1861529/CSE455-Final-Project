from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import cv2

class person:
    def __init__(self, id, alpha, sum, n, name = "") -> None:
        self.name = name
        self.id = id
        self.alpha = alpha
        self.sum = sum
        self.n = n

class PCA:
    
    def __init__(self, data_path: str, m, n) -> None:
        self.data_path = data_path
        # Connects id to person
        self.persons = dict()
        self.m = m
        self.n = n
        self.U = None
        self.UrT = None
        self.avg_face_vector = None

    def im2double(self, im):
        assert not np.issubdtype(im.dtype, np.float64)
        return im.astype('float') / (np.iinfo(im.dtype)).max

    def train(self):
        """
        Named train as it it prepares a model and therefore the equivalent
        of a neural network's train function.
        PCA, however, is able to train in a single pass through
        """
        mat_contents = scipy.io.loadmat(os.path.join('.','data','allFaces.mat'))

        faces = mat_contents['faces']
        self.m = int(mat_contents['m'])
        self.n = int(mat_contents['n'])

        avg_face_vector = np.mean(faces, axis=1, dtype='float64')
        avg_face_vector = np.reshape(avg_face_vector, (self.n*self.m, 1))

        X = faces - np.tile(avg_face_vector, (faces.shape[1], 1)).T
        U, S, VT = np.linalg.svd(X, full_matrices=0)

        r = 800

        d = plt.imshow(np.reshape(avg_face_vector, (self.m, self.n)).T)
        d.set_cmap('gray')
        plt.axis('off')
        plt.show()

        Ur = U[:, :r]
        print(np.shape(avg_face_vector))
        self.U = U
        self.UrT = Ur.T
        self.avg_face_vector = avg_face_vector
        np.save("U.npy", U)
        np.save("UrT.npy", Ur.T)
        np.save("avg_face_vector.npy", avg_face_vector)


    def compute_alpha(self, face_vector):
        return self.UrT @ (face_vector - self.avg_face_vector)

    def add_person(self, id, path, name=""):
        Sum_Img, avg_vector, n =self.avg_images(path)
        sum_vector = np.reshape(Sum_Img, (self.n * self.m, 1), 'F')

        alpha = self.compute_alpha(avg_vector)
        p = person(id, alpha, sum_vector, n, name)
        self.persons[id] = p

    def update_person(self, id:int, Sum_Face:np.ndarray, face_vector:np.ndarray):
        self.persons[id].Sum_Face += face_vector
        self.persons[id].n += 1
        avg = self.persons[id].Sum_Face / self.persons[id].n
        alpha = self.compute_alpha(avg)
        self.persons[id].alpha = alpha

    def euc_dist(self, u, v):
        return np.sqrt(np.sum(np.square(u, v)))
    
    def predict(self, img_path:str):
        """
        Predict who this is.
        On failure, returns None.
        Returns id of best match and euclidean dist, otherwise.
        Eucludean dist can be used to determine uncertainty.
        Eucludean distance above a cetain point should be understood
        as no-match.
        """
        match = -1
        if not os.path.isfile(img_path):
            return match
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.m, self.n))
        face_vector = np.reshape(img, (self.m * self.n, 1), 'F')

        alpha = self.compute_alpha(self, face_vector)
    
        #  Find person with most similar alpha
        #  Ignore first 3 principle components as these will have most
        #  to do lighting
        min_dist = float('inf')
        for id in self.persons:
            dist = self.euc_dist(self.persons[id].alpha[4:], alpha[4:])
            if dist < min_dist:
                min_dist = dist
                match = id
        return match, min_dist
    
    def avg_images(self, img_dir):
        """
        Gets all jpg images from the imgs and converts them to greyscale, resizes
        them to be m x n, and calculates their averages. Returns a list that
        contains the sum, the vector of averages, and the number of images.
        """

        imgs = []
        for filename in Path(img_dir).glob('*.jpg'):
            im = cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_RGB2GRAY)
            imgs.append(im)

        sum = np.zeros((self.n, self.m))
        for i in range(len(imgs)):
            im = cv2.resize(self.im2double(imgs[i]), (self.m, self.n))
            sum = np.add(sum, im)

        avg = sum / len(imgs)
        x = np.reshape(avg, (self.n * self.m, 1), 'F')
        count = len(imgs)

        return (sum, x, count)
