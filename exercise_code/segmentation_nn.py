"""SegmentationNN"""
import torch
from torch import nn
from torchvision import models


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, pretrained=True, caffe=False):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################

        self.features = models.alexnet(pretrained).features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(256, 4096, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, num_classes, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=40),
            nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1)
        )

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################

        x = self.features(x)
        x = self.classifier(x)

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
