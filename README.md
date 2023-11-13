# 1. Overview
This repository is an exercise in image segmentation using lung images as the dataset. 

Medical Image Segmentation is the process of automatic detection of boundaries within images. In this exercise, I train a neural network with [U-Net](https://arxiv.org/abs/1505.04597) architecture.

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

# 1. Installation

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


# 3. Data Preparation

1. Combine left and right lung segmentation masks of Montgomery chest x-rays
1. Resize images to 512x512 pixels
1. Split images into training and test datasets
1. Write images to /segmentation 
1. Training dataset will have an 80:20 training:validation split
1. Test dataset is 50 images that are put aside for final testing of the model

# 4. Training
1. Trainer class implements the training loop and the testing loops
1. Loss Function: 
 - I'm currently using PyTorch's (BCEWithLogitsLoss)[https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html]

 - (dice coefficient)[https://towardsdatascience.com/biomedical-image-segmentation-u-net-a787741837fa#:~:text=Dice%20coefficient,-A%20common%20metric&text=The%20calculation%20is%202%20*%20the,denotes%20perfect%20and%20complete%20overlap]

1. Tuning
 - Adding augmentations had a positive effect on the final performance as measured by the Dice Score
  - Used the `albumentations` library which was much clearer than doing by hand
 - Training model longer(50 epochs) 
 - combined loss function
  - High weight for Dice(*10) gave poorer Dice output scores to start and then once the BCE score started to outweigh the Dice score then learning proceeeded as if BCE only.

1. Next steps

# 5. Test
1. Test dataset is 50 images that are put aside for final testing of the model
2. Retrieve saved model, send these images through the pretrained model and compare the result with the target

# 6. Deployment
1. Save model at the end of training on kaggle
1. Download model
1. Create a new dataset in kaggle and upload model to it

## Kaggle

1. Link github repo to kaggle and it appears in `/kaggle/working`.
  - This was key if you don't want to only use a notebook, but want to write python modules
  - See notebook for keys to making this work
1. Add it to the path and then can use it as a custom library from your notebook

# Finally. Learning
1. The data/image manipulation had a higher learning curve than expected.  Mixing image libraries led to unexpected formatting.
1. Interactions between Kaggle and Github
 - Key is that you can clone a github repo into your kaggle working directory and then add the path to your python modules into the notebook environment's path. 
1. Interactions between jupyter notebook and code modules
 - Auto reload of changes within your modules can be enabled with this

[Reference](https://bobbyhadz.com/blog/jupyter-notebook-reload-module#:~:text=Use%20the%20%25load_ext%20autoreload%20magic,before%20executing%20the%20Python%20code.)
```
%load_ext autoreload
%autoreload 2

```
1. With this interaction between kaggle and github setup, it is good to instrument your code to easily toggle between your local dev environment and the kaggle environment.  The process is:
 - Do development locally
 - upload to github
 - pull your updates into your kaggle notebook