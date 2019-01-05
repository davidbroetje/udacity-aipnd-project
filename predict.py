#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: David Brötje
# DATE CREATED: 08/07/2018                                  
# REVISED DATE: 
# PURPOSE: Use a trained network to predict the class for an input image.
#          Prints the most likely classes.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python predict.py </path/to/image> <checkpoint>
#                      --top_k <return top K most likely classes> 
#                      --category_names <path to a JSON file that maps the class values to other category names>
#                      --gpu 
#   Example call:
#    python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --category_names cat_to_name.json --top_k 5 --gpu
##

# Imports python modules
import matplotlib.pyplot as plt
import argparse
from time import time, sleep
import json
import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from PIL import Image
import torch.nn.functional as F

from model import load_checkpoint
from utils import process_image
from utils import show_prediction

# Main program function defined below
def main():
    # Measures total program runtime by collecting start time
    start_time = time()
    
    # Creates & retrieves Command Line Arugments
    in_arg = get_input_args()
        
    # Set device to cuda if gpu flag is set
    device = 'cuda' if in_arg.gpu==True else 'cpu'
    
    # If given, read the mapping of categories to class names
    cat_to_name = {}
    if in_arg.category_names:
        with open(in_arg.category_names, 'r') as f:
            cat_to_name = json.load(f)
    
    # Load checkpoint
    model, _, _ = load_checkpoint(in_arg.checkpoint)

    # Predict classes
    probs, classes = predict(in_arg.img_path, model, device, in_arg.top_k)
    
    # Convert categories to real names if a mapping was given
    if cat_to_name:
        classes = [cat_to_name[str(cat)] for cat in classes]

    # Print results
    print('\nThe top {} most likely classes are:'.format(in_arg.top_k))
    max_name_len = len(max(classes, key=len))
    row_format ="{:<" + str(max_name_len + 2) + "}{:<.4f}"
    for prob, name in zip(probs, classes):
        print(row_format.format(name, prob))
    
    # Measure total program runtime by collecting end time
    end_time = time()
    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    


# Functions defined below
def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create parser 
    parser = argparse.ArgumentParser()

    # Add command line arguments
    parser.add_argument('img_path', type=str,
                        help='path to input image')
    parser.add_argument('checkpoint', type=str, 
                        help='path to a saved checkpoint')
    parser.add_argument('--top_k', type=int, default=3, 
                        help='return top K most likely classes')
    parser.add_argument('--category_names', type=str,
                        help='path to a JSON file that maps the class values to other category names')
    parser.add_argument('--gpu', action='store_true', 
                        help='boolean switch: use GPU for inference')

    # return parsed argument collection
    return parser.parse_args()


#Define function to predict top K classes
def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
    Parameters:
     image_path - string, path to input image
     model - torch.nn.Module, model to use for prediction
     device - string, the device on which the prediction should be done
     topk - integer, return top K most likely classes
    Returns:
     probs - numpy.ndarray, probabilities of the top k most likely classes
     classes - numpy.ndarray, top K most likely classes
    '''
    
    # DONE: Implement the code to predict the class from an image file
    image = Image.open(image_path).convert('RGB')
    image = process_image(image)
    image = torch.from_numpy(image).unsqueeze_(0).float()
    
    model = model.to(device)
    image = image.to(device)
    model.eval()
    
    # Calculate class probabilities 
    with torch.no_grad():
        outputs = model.forward(image)
    
    # Get topk probabilities and classes
    probs, class_idxs = outputs.topk(topk)
    
    probs, class_idxs = probs.to('cpu'), class_idxs.to('cpu')
    probs = probs.exp().data.numpy()[0]
    class_idxs = class_idxs.data.numpy()[0]
    
    # Convert from indices to the actual class labels
    classes = np.array([model.idx_to_class[idx] for idx in class_idxs])
    
    return probs, classes


# Call to main function to run the program
if __name__ == "__main__":
    main()