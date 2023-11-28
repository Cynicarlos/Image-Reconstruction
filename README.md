# Image-Reconstruction  
This project is for IAT and it's just an attempt to do two-dimensional image reconstruction and denoising with different methods, including UNet, NAFNet, with my own dataset.(to be improved)  
The cotton silk images with noisy were collected by microscope as the dataset and it is provided. Just **961** images in total so far. It is divided into training set, verification set and test set according to **8:1:1**. You can divide it by your own. The Ground Truth is got by Hilbert transform with evey three continuous frame images. But the results I got so far with simpel network is not good enough, and the improvement is to be continued.

# Quick start
pip install -r requirements.txt  
python train.py


