# CSE455 Final Project: Facial Recognition
## Introduction:
This is the final project for CSE 455, done by Duke Parks, Amanda Ki, Priti Ahya, Abas Hersi, Michael Wen, and Edison Chau.

## Description:
This project will use Principal Component Analysis (PCA) and Eigenfaces to identify who a face belongs to, or if the face is
unidentified.

## Walkthrough:
PCA/pca.py contains functions used to train our function, add a person to be indentified, update a person, predict who a person
from an image is (or return None if the person is unidentified),
and also to average all the images of a person after converting
them to grayscale, and resizing them to be a specifed size. This
file also has two classes. The first class is a person class, which is defined by a name, id, alpha (the projection of a given face onto the data), sum and n. The PCA class is defined by m, n (the dimentions the images should be resized to).
U (the unitary matrix from the SVD of the data), UrT (the transpose of the rank approximation of U) and Avg_Face (the average of the data).