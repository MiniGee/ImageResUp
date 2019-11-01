# Image Resolution Up

Note: This project was made for the purpose of my learning experience and is not intended to actually be used

## Summary

This network is designed to increase the resolution of images. The network is very simple and only consists of 4 transposed convolutional layers, each followed by batch normalization and LeakyReLU as the activation. There is only 1 upsampling layer in the very middle of the layers.
For the training, a dataset of anime faces was used. Each image lower than a certain resolution was discarded, and every image above the resolution was down scaled to the target size. These were the labels. These images were downscaled again to half the size, and the lower resolution images became the features.
The final layer uses a sigmoid activation, binary_crossentropy was used, and the adam optimizer was used.

The dataset can be found here: https://drive.google.com/file/d/1HG7YnakUkjaxtNMclbl2t5sJwGLcHYsI/view?usp=sharing
