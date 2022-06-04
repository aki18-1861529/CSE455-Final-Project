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
    
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        # Connects id to person
        self.persons = dict()
        self.m
        self.n
        self.U
        self.UrT
        self.Avg_Face

    #  Named train as it it prepares a model and therefore the equivalent
    #  of a neural network's train function.
    #  PCA, however, is able to train in a single pass through
    def train(self):
        mat_contents = scipy.io.loadmat(os.path.join('..','data','allFaces.mat'))

        faces = mat_contents['faces']
        self.m = int(mat_contents['m'])
        self.n = int(mat_contents['n'])

        Avg_Face = np.mean(faces, axis=1)

        X = faces - np.tile(Avg_Face, (faces.shape[1],1)).T
        U, S, VT = np.linalg.svd(X, full_matrices=0)

        r = 800

        d = plt.imshow(np.reshape(Avg_Face, (self.m, self.n)).T)
        d.set_cmap('gray')
        plt.axis('off')
        plt.show()

        Ur = U[:, :r]
        print(np.shape(Avg_Face))
        self.U = U
        self.UrT = Ur.T
        self.Avg_Face = Avg_Face
        np.save("U.npy", U)
        np.save("UrT.npy", Ur.T)
        np.save("Avg_Face.npy", Avg_Face)


    def compute_alpha(self, face_vector):
        return self.UrT @ (face_vector - self.Avg_Face)

    def add_person(self, id, path, name=""):
        """"""
        
        Sum_Face = 0
        n = 0
        Avg_Person = 0
        
        """"""
        alpha = self.compute_alpha(Avg_Person)
        p = person(id, alpha, Avg_Person, name)
        self.persons[id] = p

    def update_person(self, id:int, Sum_Face:np.ndarray, face_vector:np.ndarray):
        self.persons[id].Sum_Face += face_vector
        self.persons[id].n += 1
        avg = self.persons[id].Sum_Face / self.persons[id].n
        alpha = self.compute_alpha(avg)

    def euc_dist(u, v):
        return np.sqrt(np.sum(np.square(u, v)))

    # Predict who this is.
    # On failure, returns None.
    # Returns id of best match and euclidean dist, otherwise.
    # Eucludean dist can be used to determine uncertainty.
    # Eucludean distance above a cetain point should be understood
    # as no-match.
    def predict(self, img_path:str):
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