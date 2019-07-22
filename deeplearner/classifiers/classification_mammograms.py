import torch
import torch.nn as nn
from torchvision import models

class ClassificationMammograms(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ClassificationMammograms, self).__init__()

        self.model = models.alexnet(pretrained)
        self.upsample = nn.Upsample(size=(227, 227))
        self.model.fc = nn.Linear(4096, num_classes)

        
        
        # bring input to size 227x227 like in alexnet necessary
        # modify last layer to output layer of size num_classes
    def forward(self, x):

        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        # my code
        x = self.upsample(x)
        x = self.model(x)
        #

        return x

    def num_flat_features(self, x):
        """
        Computes the number of features if the spatial input x is transformed
        to a 1D flat input.
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
