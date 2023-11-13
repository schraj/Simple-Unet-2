# Medical Image Segmentation

## 1. Overview
This repository is an exercise in medical image segmentation using lung images as the dataset. 

Medical Image Segmentation is the process of automatic detection of boundaries within medical images. In this exercise, I train a neural network with [U-Net](https://arxiv.org/abs/1505.04597) architecture.

Uses: 
- Identifying Regions of Interest: Medical image segmentation is used to distinguish and isolate specific regions or structures within medical images, such as organs, tissues, or tumors.
- Facilitating Diagnosis: It assists in diagnosing diseases, planning treatments, and evaluating medical conditions by providing detailed anatomical information.
- Enhancing Image Analysis: Segmentation improves the visibility and clarity of different structures, making it easier for medical professionals to interpret complex images.
- Supporting Surgical Planning: By clearly demarcating areas of interest, it aids surgeons in planning surgical procedures, especially in delicate or complex operations.
- Quantitative Analysis: Enables the quantification of volumes, areas, and other measurements of specific anatomical structures, crucial for monitoring disease progression or response to therapy.

Two step analysis for pathologies such as TB:
- Initial Lung Segmentation: The first step typically involves segmenting the lung region from the rest of the chest X-ray or CT scan. This focuses the analysis on the lungs, eliminating irrelevant areas.
- Detailed Analysis of Lung Region: After segmentation, the identified lung area undergoes detailed analysis specifically for signs of tuberculosis, like nodules, consolidation, or patterns indicative of TB.
- Improves Accuracy: Segmenting the lung first reduces background noise and distractions, improving the accuracy of TB identification.
- Enhances Computational Efficiency: It narrows down the area of interest, making subsequent processing faster and more efficient.
- Multi-stage Approach: This two-step approach (segmentation followed by disease identification) is common in medical image analysis for various diseases.

![unet](images/unet.png)

Inspiration for lung segmentation project was taken from this (notebook)[https://www.kaggle.com/code/eduardomineo/u-net-lung-segmentation-montgomery-shenzhen]

U-net model was adapted from this these two projects

- (blog post)[https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-model-building-6ab09d6a0862]

## 1. Installation

1. Set up a new environment with an environment manager (recommended):
   1. [conda](https://docs.conda.io/en/latest/miniconda.html):
      1. `conda create --name unet-lung -y`
      2. `conda activate unet-lung`
      3. `conda install python=3.10 -y`
   2. [venv](https://docs.python.org/3/library/venv.html):
      1. `python3 -m venv unet-lung`
      2. `source unet-lung/bin/activate`
2. Install the libraries:
  1. conda
```
TODO: get list of libraries
```

  2. venv
    `pip install -r requirements.txt`

3. Start a jupyter server or run in VS code:
`jupyter-notebook` OR `jupyter-lab`

## 2. Prepare Kaggle Environment to Sing :)
1. We don't want to write a bunch of python code in notebook, much prefer python modules.  To make this work well, we need to do a number of things:
- Clone your github repo and it will then be available in your `/kaggle/working` directory
```
!git clone https://github.com/schraj/simple-unet-2.git
```
- add the following code so that every time you update your python code in the repo and pull from it, your local kernel will update:
```
%load_ext autoreload
%autoreload 2
```
- Pull latest from your repo every time you have made a change:
```
# Change directory to the local repository's directory
%cd /kaggle/working/simple-unet-2

# Pull the latest changes
!git pull origin main
```
- Add your `src` directory from your cloned repo to your local notebook kernel's path:
```
import sys
sys.path.append('/kaggle/working/simple-unet-2/src')
```
- To get your notebook to run correctly when you run it offline with "Save and Run All" with a large imported dataset(like the images datasets used in this projects), you need to wait for it to load before access it:

```
import time
time.sleep(200)
```

- Finally, things will run very differently on your local machine versus on the Kaggle kernel, with GPUs etc.  Therefore, you'll want to be able to change config settings on the fly in your notebook.  You can have a config file in your python project and then overwrite this values in your notebook with the settings you want for running in Kaggle:

```
import src.config as h
h.LOCAL = False
h.NUM_EPOCHS = 50
h.BATCH_SIZE = 16
h.LEARNING_RATE = 1e-4
```

Here is a (link)[https://www.kaggle.com/jeremyschrader1/kaggle-with-github-repo] to an example of this setup

## 3. Data Preparation

1. Combine left and right lung segmentation masks of Montgomery chest x-rays
1. Resize images to 512x512 pixels
1. Split images into training and test datasets
1. Write images to /segmentation 
1. Training dataset will have an 80:20 training:validation split
1. Test dataset can be either 50 or 100 images that are put aside for final testing of the model

## 4. Data Pipeline
1. Define a custom dataset
- Holds the list of file names and how to load one
2. Define data loaders
- Defines the transformation pipeline that an image will undergo when asked for by the training loop.

## 5. Training
1. Trainer class implements the training loop and the testing phase
1. Loss Function: 
 - PyTorch's (BCEWithLogitsLoss)[https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html]

 - Torchmetrics (dice coefficient)[https://towardsdatascience.com/biomedical-image-segmentation-u-net-a787741837fa#:~:text=Dice%20coefficient,-A%20common%20metric&text=The%20calculation%20is%202%20*%20the,denotes%20perfect%20and%20complete%20overlap]

## 5. Test Phase
1. Test dataset is 50-100 images that are put aside for final testing of the model
1. Retrieve saved model, send these images through the pretrained model and compare the result with the target
1. Determine the average Accuracy, Dice Score, and BCE Loss.

## 6. Deployment
1. Save model at the end of training on kaggle
1. Download model
1. Create a new dataset in kaggle and upload model to it

# Finally. Learning
1. The data/image manipulation had a higher learning curve than expected.  Mixing image libraries led to unexpected formatting.
