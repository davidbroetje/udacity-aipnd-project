#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: David Brötje
# DATE CREATED: 08/07/2018                                  
# REVISED DATE:
# PURPOSE: Provide utility function for data loading with PyTorch and image processing.

from torchvision import datasets, transforms
from PIL import Image
import numpy as np

# Define transforms for the training and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model.
    
    Parameters:
     image - PIL image, image to be processed
    Returns:
     The processed image as Numpy array.
    '''

    # Resize
    width, heigth = image.size
    
    if heigth > width and heigth > 256:
        image.thumbnail([256,heigth], Image.ANTIALIAS)
    elif width > heigth and width > 256:
        image.thumbnail([width,256], Image.ANTIALIAS)
    else:
        image.thumbnail([256,256], Image.ANTIALIAS)

    
    # Center Crop
    width, heigth = image.size
    
    left = (width - 224)/2
    top = (heigth - 224)/2
    right = (width + 224)/2
    bottom = (heigth + 224)/2
    cropped_image = image.crop((left, top, right, bottom))
    
    # Convert color channels 0-255 to float 0-1
    np_image = np.array(cropped_image)/255
    
    # Normalize color channels
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # PyTorch expects the color channel to be the first dimension 
    #but it's the third dimension in the PIL image and Numpy array
    return np_image.transpose(((2, 0, 1)))


def show_processed_image(image, ax=None, title=None):
    ''' Displays an image that was processed for a PyTorch model.
    
    Parameters:
     image - Numpy array/PyTorch tensor, the processed image
     ax - matplotlib.pyplot.axes, axes the image should be plotted on
     title - string, title of the subplot
    Returns:
     ax - matplotlib.pyplot.axes, axes the image is plotted on
    '''
   
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    if title is not None:
        ax.set_title(title)
        
    return ax


def show_prediction(probs, classes, cat_to_name=None, ax=None, title=None):
    ''' Display class names and probabilities of the prediction as horizontal bar chart.
    
    Parameters:
     probs - numpy.ndarray, probabilities of the top k most likely classes
     classes - numpy.ndarray, top K most likely classes
     cat_to_name - dict, a mapping of categories to real names
     ax - matplotlib.pyplot.axes, axes the prediction should be plotted on
     title - string, title of the subplot
    Returns:
     ax - matplotlib.pyplot.axes, axes the prediction is plotted on
    '''
    if ax is None:
        fig, ax = plt.subplots()
        
    ax.barh(-np.arange(len(classes)), probs)
    ax.set_aspect(0.1)
    ax.set_yticks(-np.arange(len(classes)))
    ax.set_xlim(0, 1.1)
    
    if cat_to_name:
        class_names = [cat_to_name[str(cat)] for cat in classes]
        ax.set_yticklabels(class_names)
    
    if title:
        ax.set_title(title)
    
    return ax