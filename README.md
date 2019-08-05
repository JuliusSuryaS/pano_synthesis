# pano_synthesis
Partial panorama synthesis based on hierarchical model

### Introduction
FOV panorama prediction and partial panorama synthesis (128 x 512) implemented in Tensorflow. \
Implemented with hierarchical GAN based network.
![Framework](./img/framework.png)

### Results
Input is 4 panorama images concatenated together. The FOV is predicted and the input processed into panorama with missing pixel based on the FOV. 
###### Output 1.
![Framework](./img/Picture1.png)
###### Output 2.
![Framework](./img/Picture2.png)
###### Output 3.
![Framework](./img/Picture3.png)

### Prerequisites
* Python 2.7
* OpenCV
* Tensorflow
