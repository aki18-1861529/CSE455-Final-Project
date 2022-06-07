## Face Detection and Recognition

CSE 455 Spring 2022

Duke Parks, Amanda Ki, Priti Ahya, Abas Hersi, Michael Wen, Edison Chau

## Introduction
Facial recognition is a technology that has been around for decades. During this time, the technology has matured, and saw many different applications in various fields, ranging from live camera filters to security. Despite the range of applications it has, facial recognition boils down to software that can detect a person's face in visual media, such as photos and videos.

However, like with many other technologies, the way facial recognition is implemented varies between developers, and therefore can have some variation in terms of accuracy and performance.

Based on this idea, we decided to implement a facial recognition software that builds on two different facial recognition techniques in order to compare their performance:
- The first technique uses a Facebook technology called DETR (DEtection TRansformer), which is used to detect and recognize different subject models, which we modified forrecognizing faces.
- The second technique uses Principal Component Analysis (PCA) and data from the Yale Face Database to detect faces based on a generated "average" face.

## DETR
As previously mentioned, DETR is a Facebook-developed technology that is used to recognize different objects in pictures. For the purposes of our project, we introduced data from WiderFaces to train it to recognize faces.

## PCA
For this approach, we generated a PCA model using the Yale Face Database to analyze multiple samples of human faces, and used greyscaled images of celebrities to generate an "average" face which was used for classification.

## Conclusions
Based on the data we collected, we found that DETR has better performance than PCA does. However, this performance comes with a tradeoff. DETR relies on a very large amount of data that takes a very long time and a lot of resources to properly train and run, and requires data to be tailored beforehand. On the other hand, PCA is much less demanding in terms of data and resources, and is more flexible with the data it is given, at the cost of less accuracy.

## References
[DETR](https://github.com/facebookresearch/detr)