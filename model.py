#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: David Brötje
# DATE CREATED: 08/07/2018                                  
# REVISED DATE:
# PURPOSE: Provide functions and classes for transfer learning with PyTorch models.

# Imports here
import torch
from torchvision import models
from torch import nn, optim
import torch.nn.functional as F

# Define class for a feedforward network with arbitrary hidden_layers
class FeedforwardNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
        Parameters:
         input_size - integer, size of the input
         output_size - integer, size of the output layer
         hidden_layers - list of integers, the sizes of the hidden layers
         drop_p - float between 0 and 1, dropout probability
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)


# Define function for building the model
def build_model(arch, output_size, hidden_sizes):
    """ Builds and returns a model for transfer learning.
    
    Loads a pretrained torchvision model, freezes it's parameters and replaces the classifier
    with a new feedforwad network.
    Parameters:
     arch - string, architecture of the pretrained model e.g. alexnet, densenet121...
     output_size - integer, size of the output layer of the classifier
     hidden_sizes - list of integers, the sizes of the hidden layers of the classifier
    Returns:
     model - torch.nn.Module, the built model (parameters of the feature network are frozen)
    """
    # Load pre-trained network
    model = models.__dict__[arch](pretrained=True)
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Define new, untrained feed-forward network based on arch
    if arch.startswith('densenet'):
        model.classifier = FeedforwardNetwork(model.classifier.in_features, output_size, hidden_sizes)
    
    #elif arch.startswith('resnet'):
    #    model.fc = FeedforwardNetwork(model.fc.in_features, output_size, hidden_sizes)
    
    elif arch.startswith('vgg'):
        model.classifier = FeedforwardNetwork(model.classifier[0].in_features, output_size, hidden_sizes)
    
    elif arch == 'alexnet':
        model.classifier = FeedforwardNetwork(model.classifier[1].in_features, output_size, hidden_sizes)
        
    else:
        raise ValueError("Architecture '{}' not supported".format(arch)) 
    
    return model


# Define validation function
def validate_model(model, dataloader, criterion, device):
    ''' Validates the model with a validation set.
    
    Parameters:
     model - torch.nn.Module, model to be validated
     dataloader - torch.utils.data.DataLoader, data the model will be validated with
     criterion - nn.criterion, criterion/loss function to use
     device - string, the device on which the validation should be done 
    Returns:
     loss - float, validation loss
     accuracy - float, validation accuracy
    '''
    
    valid_loss = 0
    accuracy = 0
    
    # Set model to eval mode 
    model.eval()
    
    model = model.to(device)
    # Turn of gradients
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            valid_loss += criterion(output, labels).item() * inputs.size(0)
            
            #the output is log-softmax -> inverse via exponential to get probabilities
            ps = torch.exp(output)
            #ps.max(dim=1)[1] gives the class with the highest probability for each input
            #equality will result in a tensor where 1 is correct prediction and 0 is false prediction
            equality = (labels.data == ps.max(dim=1)[1])
            #equality is a ByteTensor which has no mean method -> convert to FloatTensor
            accuracy += equality.type(torch.FloatTensor).mean() * inputs.size(0)
    
    return valid_loss/len(dataloader.dataset), accuracy/len(dataloader.dataset)


# Define training function
def train_model(model, trainloader, validloader, epochs, criterion, optimizer, device):
    """ Trains the model with a training set.
    
    Parameters:
     model - torch.nn.Module, model to be trained
     trainloader - torch.utils.data.DataLoader, data the model will be trained on
     validloader - torch.utils.data.DataLoader, data the model will be validated with
     epochs - integer, number of training epochs
     criterion - nn.criterion, criterion/loss function to use
     optimizer - torch.optim, optimizer to use
     device - string, the device on which the training should be done 
    Returns:
     Nothing
    """
    print('Start training')
    model = model.to(device)
    
    # Set model to train mode
    model.train()
    
    for e in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
                  
            # Clear the gradients (gradients are accumulated)
            optimizer.zero_grad()
        
            # Feed forward
            outputs = model.forward(inputs)
            # Calculate loss
            loss = criterion(outputs, labels)
            # Pass backward to calculate the gradients
            loss.backward()
            # Take a step with the optimizer to update the weights
            optimizer.step()
        
            running_loss += loss.item() * inputs.size(0)
            
        # end of epoch, validate the model
        valid_loss, valid_accuracy = validate_model(model, validloader, criterion, device)
            
        print("Epoch: {:2}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader.dataset)),
              "Validation Loss: {:.3f}.. ".format(valid_loss),
              "Validation Accuracy: {:.3f}".format(valid_accuracy))
        
        # Set model back to train mode
        model.train()
            
    print('Finished training')
    

# Define function to save checkoints
def save_checkpoint(file_path, model, arch, output_size, epochs, optimizer):
    """ Saves the model as a torch.utils.checkpoint.
    
    Parameters:
     file_path - string, path to the file the checkpoint should be saved in
     model - torch.nn.Module, model to be saved
     arch - string, architecture of the pretrained model e.g. alexnet, densenet121...
     output_size - integer, size of the output layer
     epochs - integer, number of training epochs
     optimizer - torch.optim, optimizer to be saved
    Returns:
     Nothing
    """
    checkpoint = {'arch': arch,
                  'output_size': output_size,
                  'hidden_sizes': [each.out_features for each in model.classifier.hidden_layers],
                  'state_dict': model.state_dict(),
                  'epochs': epochs,
                  'optimizer_state': optimizer.state_dict,
                  'class_to_idx': model.class_to_idx,
                  'idx_to_class': model.idx_to_class
                 }
    torch.save(checkpoint, file_path)

# Define function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    ''' Loads a checkpoint and rebuilds the model.
    
    Parameters:
     file_path - string, path to the file holding the checkpoint
    Returns:
     model - torch.nn.Module, loaded/rebuild model
     epochs - integer, number of epochs the model was trained
     optimizer_state - torch.optim, optimizer to be saved
    '''
    checkpoint = torch.load(filepath)
    
    # Rebuild the model
    model = build_model(checkpoint['arch'], checkpoint['output_size'], checkpoint['hidden_sizes'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    
    return model, checkpoint['epochs'], checkpoint['optimizer_state']
