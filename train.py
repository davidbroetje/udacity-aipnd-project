#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: David Brötje
# DATE CREATED: 08/07/2018                                  
# REVISED DATE: 
# PURPOSE: Train a new network on a dataset and save the model as a checkpoint.
#          Prints out training loss, validation loss, and validation accuracy 
#          as the network trains. Prints out the accuracy on the test set and 
#          the total elapsed runtime after training.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py <data directory> 
#                      --save_path <file path to save the checkpoint> 
#                      --arch <model architecture>
#                      --learning_rate <learning rate>
#                      --hidden_units <number of hidden units>
#                      --epochs <number of training epochs>
#                      --gpu
#   Example call:
#    python train.py flowers 
##

# Imports python modules
import argparse
from time import time, sleep

import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F

from model import build_model
from model import train_model
from model import validate_model
from model import save_checkpoint
from utils import train_transforms
from utils import test_transforms

# Main program function defined below
def main():
    # Measure total program runtime by collecting start time
    start_time = time()
    
    # Create & retrieve Command Line Arugments
    in_arg = get_input_args()
    
    # Set device to cuda if gpu flag is set
    device = 'cuda' if in_arg.gpu==True else 'cpu'
    
    # Load the data
    train_dir = in_arg.data_dir + '/train'
    valid_dir = in_arg.data_dir + '/valid'
    test_dir = in_arg.data_dir + '/test'

    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the transforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32)
    
    # Get the number of classes as the size of the output layer
    output_size = len(train_datasets.classes)

    # Build the model
    model = build_model(in_arg.arch, output_size, in_arg.hidden_units)

    # Insert mapping from class to index and index to class
    model.class_to_idx = train_datasets.class_to_idx
    model.idx_to_class = {i: c for c, i in model.class_to_idx.items()}
    
    # Move model to cuda before constructing the optimizer
    model = model.to(device)

    # Define criterion
    criterion = nn.NLLLoss()

    # Define optimizer
    # Only train the classifier parameters, feature parameters are frozen (p.requires_grad == false)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=in_arg.learning_rate)

    # Train the model
    train_model(model, trainloader, validloader, in_arg.epochs, criterion, optimizer, device)

    # Test the model
    print('Start testing')
    _, test_accuracy = validate_model(model, testloader, criterion, device)
    print('Finished testing')
    print("Accuracy of the model on the test set: {:.3f}".format(test_accuracy))
    
    # Save checkpoint
    save_checkpoint(in_arg.save_path, model, in_arg.arch, output_size, in_arg.epochs, optimizer)
    
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
    parser.add_argument('data_dir', type=str,
                        help='directory with training, validation and testing data')
#    parser.add_argument('--save_dir', type=str, default='./', 
#                        help='directory to save checkpoints')
# better use file_path for checkpoints
    parser.add_argument('--save_path', type=str, default='./checkpoint.pth', 
                        help='file path to save the checkpoint')
    parser.add_argument('--arch', type=str, default='densenet121', 
                        choices=['alexnet', 'densenet121', 'vgg16'],
                        help='model architecture')
    # the architecure is handled in the build_model function from model.py
    # if the given architecture is not supported, it will raise an error.
    # per specification the program should allow to choose, so three choices are given
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='applied learning rate')
    parser.add_argument('--hidden_units', nargs='+', type=int, default=[512],
                        help='number of units in the hidden layers. provide one value for each hidden layer.' +
                        ' e.g. "--hidden_units 512 256" will create two hidden layers. the first one will have 512 and the second one 256 units')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs')
    parser.add_argument('--gpu', action='store_true', 
                        help='boolean switch: use GPU for training')

    # return parsed argument collection
    return parser.parse_args()


# Call to main function to run the program
if __name__ == "__main__":
    main()
