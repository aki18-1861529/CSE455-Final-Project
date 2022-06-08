## Facial Detection and Recognition

CSE 455 Spring 2022

Duke Parks, Amanda Ki, Priti Ahya, Abas Hersi, Michael Wen, Edison Chau

## Introduction
Facial recognition is a technology that has been around for decades. During this time, the technology has matured, and saw many different applications in various fields, ranging from live camera filters to security. Despite the range of applications it has, facial recognition boils down to software that can detect a person's face in visual media, such as photos and videos.

However, like with many other technologies, the way facial recognition is implemented varies between developers, and therefore can have some variation in terms of accuracy and performance.

## Project Setup
For this project, our group addressed the issue of facial detection and recognition.  Facial recognition has a multitude of different applications within various fields such as security, education, healthcare, etc.  For our project, we wanted to create a complete program that could be given an image and subsequently find and identify faces in that image. In addition to this, we wanted to do a project that utilized both old and new techniques for two reasons.  The first is for the sake of creating a qualitative comparison.  The second was in hopes of finding a middle ground between the accurate, but expensive modern neural networks, and the efficient, but somewhat erroneous older techniques.  This led us to creating a project that uses the novel DETR in conjunction with the well-researched PCA.

## Pre-existing Work
Some pre-existing work and resources that we used in this project include:
- [DETR Implementation](https://github.com/facebookresearch/detr)
- Code to convert WiderFaces to COCO format
- Eigenfaces (implemented from scratch for this project, but by no means a new technique)
- Matrix of cropped and vectorized images from Yale Face Database B

## Our Implementation
As a brief summary of the parts of our project that we implemented ourselves:
- Collected celebrity photos
- Trained DETR on WiderFaces data for 4 epochs
- Ran collected photos through detr
- Extracted bounding boxes from output of DETR and cropped out detected faces
- PCA face recognition

## Techniques
![Mismatches](/imgs/mismatches.jpg)

DETR is able to detect multiple faces in an image. We used the existing DETR network pre-trained on the COCO dataset with a ResNet50 backbone and retrained it on the WiderFaces dataset so that it would detect faces rather than objects. Then, we gave it images of celebrities collected from the internet. Since DETR outputs bounding boxes for each face detected in an image, we used the bounding boxes to crop and save each face found. The cropped celebrity faces output from DETR were then used as input images for PCA.

For facial recognition, we chose to use PCA/Eigenfaces. In implementing this project, we created a PCA model from scratch to analyze the human face using faces from the Yale Face Database B. The input images of celebrities from DETR are first grayscaled and resized to the same width and length. The resized images are then used to find their “average” face for classification. PCA is performed on each input image. The result of each input image is compared with the results of the average faces to classify each image. The intuition of this can be seen in the graph below that shows the 8th, 9th, and 10th principal components of test images and average faces.

![Plot](/imgs/plotgit.gif)

Even though this is a lower dimensional image, you can still see some clustering occurring.  We can expect a greater degree of clustering to be present in the full 800-dimensional space.

## Datasets
- [WiderFaces Dataset](http://shuoyang1213.me/WIDERFACE/)
- [Yale Face Database B](http://www.databookuw.com/page-17/)
- About 80 self-collected images of celebrities

## Results
While we had a small test set, the results were very positive. Of 27 test images, 23 were correctly identified. The below images show the four mismatched faces:

![DETR PCA Diagram](/imgs/detr-pca_diagram.jpg)

It’s important to note that the four misidentified images all returned higher than the average Euclidean distance to its match, indicating that the match wasn’t confident. The Euclidean distance can be used as a measurement of uncertainty. In our implementation, we set it so that a guess would always be made. If we were scaling this program up for a different purpose, such as finding a specific person in a crowd, we would tweak it so that faces with a high minimum Euclidean distance are instead identified as “Unknown.”

A similarity between all of the mismatched images is that the bounding boxes on their faces weren’t very tight and thus included a lot of background information which is likely the reason for the misidentification. This highlights an important weakness of PCA: noise and variability. Had we been able to train DETR for longer than four epochs, the bounding boxes would likely have been tighter and less likely to have resulted in these mismatches.

## Alternative Approaches
As one could imagine, there are a number of different approaches to both face detection and face recognition. Just a few examples are listed below:

<ins>Face Detection:</ins>
- Haar Cascade Classifier
- R-CNNs
- YOLO

<ins>Face Recognition:</ins>
- 3D Face Reconstruction + Classification
- ViT
- CNNs
- K-mean clustering

All of these techniques have both advantages and disadvantages. For instance, 3D face reconstruction techniques are very accurate and very tamper-proof but a lot more expensive than, for example, K-mean clustering.

We decided to go with our implementation for a number of reasons. We chose to use DETR over the many possible networks/algorithms that can do face detection for its novelty. DETR is among the first detection/classification networks to incorporate transformers into its architecture. The use of transformers for such tasks is an exciting new frontier in computer vision that has thus far shown great results. As an example of this, DINO, a refinement of DETR, is ranked #1 on Object Detection on COCO test-dev and COCO minival on PapersWithCode.com. We chose to use PCA for face recognition because it requires far less data and computes resources to produce fairly accurate results. Because PCA is so lightweight, we were able to collect our own data that was tailored precisely to our needs, thereby making implementation faster and easier than if we had to find an existing dataset for more general purposes.

## Challenges and Potential Improvements
Our initial project goal was to design software that allowed users to superimpose clothing designs onto images of models. This would enable users to simulate what the clothing would look like without the need for a physical model. However, due to constraints such as a lack of time and unfamiliarity with certain resources, such as how we would have needed Blender for the project, a program none of our group members have experience with, we decided to shift our project topic to the current facial recognition focus.

Most of the challenges we faced over the course of the project involved figuring out how to integrate the pre-existing technologies we chose into the project. DETR was especially difficult to incorporate. We had trouble figuring out how it was structured, making it challenging to give it a proper input and interpret the output correctly. Additionally, DETR is very resource-demanding and requires a GPU that no one in our group had. This resulted in our group splitting our project: half of it on Colab, half of it on Github.

If given more time on the project, we would like to explore and incorporate other publicly-available facial recognition technologies and see how they compare with each other and see if we can determine a “sweet spot” of sorts that balances resource usage and accuracy. We also would like to create a more quantitative comparison of different face detection and face recognition techniques. Such a comparison could be the differences in accuracy and compute time of DETR versus Haar Cascade Classifiers for face detection.

## References
Shuo Yang, Ping Luo, Chen Change Loy, and Xiaoou Tang. Wider face: A face
detection benchmark. In IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2016.

Brunton, S. L., Kutz, J. N. (2022). Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control. Cambridge University Press.
