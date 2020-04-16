## MaskRCNN

### This project is adapted and show case how to handle MaskRCNN for 2 classes
### It demonstrate the use of the following main tools:
- library imgaug, opencv
- tensorflow 1.14, keras 2.2.4
- skimage
- matplotlib, numpy, scipy
- logging, shutil, warnings, re, json

### Prepare the data images using the online tool: 
- #### Tool Name:  VGG Image Annotator 
- #### URL:        http://www.robots.ox.ac.uk/~vgg/software/via/via-1.0.6.html


### Tool use to generate augmented images from a fixed set of base images
- Tool Name:        library imgaug
- Jupyter Notebook: image_augmentation.ipynb

### The model weights required are:
- File:  	mask_rcnn_coco.h5:   
Link:	https://drive.google.com/file/d/1bjypFhACIiwSacWv_LN-DPEwrnEYMaH_/view?usp=sharing

- File: 	mask_rcnn_worktool.h5   
Link:	https://drive.google.com/file/d/1xsmboTiwzKr5gpoD6QrOqpEyMZJYP4WH/view?usp=sharing

### Train the Mask-RCNN model with Google's Colab tool:
#### Steps:
- Upload the folder structure into your google drive, make sure the root folder is VSE/CA1
- Download the coco model weights file "mask_rcnn_coco.h5" and upload it to VSE/CA1
- Open the jupyter notebook file "train_worktool_model_colab.ipynb" in Google Colab
- Run the scripts to create the model

#### To check the dataset you have prepared, open the jupyter notebook:
- inspect_worktool_data.ipynb

#### To test your trained model open the jupyter notebook:
- inspect_worktool_model.ipynb

![](image/result.jpg)
