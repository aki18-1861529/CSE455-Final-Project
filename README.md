# CSE455 Final Project: Facial Recognition
## Introduction:
This is the final project for CSE 455, done by Duke Parks, Amanda Ki, Priti Ahya, Abas Hersi, Michael Wen, and Edison Chau.

## Project Website:
The webpage describing our project can be found in the following link:
https://aki18-1861529.github.io/CSE455-Final-Project/

## Description:
This project will use Principal Component Analysis (PCA) and Eigenfaces to identify who a face belongs to, as well as output an euclidean distance which can be used as a metric of uncertainty.

## Walkthrough:
PCA/pca.py contains functions used to train our function, add a person to be indentified, update a person, predict who a person
from an image is (or return None if the file isn't found),
and also to average all the images of a person after converting
them to grayscale, and resizing them to be a specifed size. This
file also has two classes. The first class is a person class, which is defined by a name, id, alpha (the projection of a given face onto the data), sum and n. The PCA class is defined by m, n (the dimentions the images should be resized to).
U (the unitary matrix from the SVD of the data), UrT (the transpose of the rank approximation of U) and avg_face_vector (the average of the data).

## References:
WiderFace data:

Shuo Yang, Ping Luo, Chen Change Loy, and Xiaoou Tang. Wider face: A face
detection benchmark. In IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2016.

	
allFaces.mat:

Brunton, S. L., Kutz, J. N. (2022). Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control. Cambridge University Press. 

  
