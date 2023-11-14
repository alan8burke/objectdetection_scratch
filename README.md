# Custom Object Detection Model

## Project Description

The goal of this project is two-fold:
- Build and train a custom lightweight object detection model (= no pre-trained model) using CNNs
- Use classical CV methods to segment parts of the predictions

The model is built using **Keras**. You will need a mid-range GPU to train the model.

The dataset is composed of images of people with and without a face masks. It is divided into a train, validation and test set. The train set contains only 105 images. 

## Installation

All the files were tested on Windows 10 with Python 3.8 and a NVIDIA RTX 2060.

Clone the repositery:   
 
`git clone https://github.com/alan8burke/objectdetection_scratch.git`

To execute the notebooks, I recommend creating a conda environment: 

`conda create --name <env> --file requirements.txt`

Download the dataset at [Roboflow](https://public.roboflow.com/object-detection/mask-wearing).

## Usage

Use Jupyter Notebook or Google Colab to open the notebooks.
1. Explore the dataset in the **data_visualisation** notebook. 
2. Build & train an object detection from scratch in **object_det**.
3. Create a segmentation pipeline using classical computer vision techniques in **segmentation**.

## File structure

```
├───data
│   ├───test
│   ├───train
│   └───valid
├───data_visualisation.ipynb
├───segmentation.ipynb
├───object_det.ipynb
└───utils_maskdataset.py
```
