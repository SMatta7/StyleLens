from torch import nn
import torchvision
import torch.nn.functional as F
import config as c

class Resnet50(nn.Module):

    def __init__(self):
        
        super().__init__()

        # Load pre-trained ResNet50
        resnet50 = torchvision.models.resnet50(pretrained=True)

        # Freeze all layers except the 5th layer (fc layer)
        for param in resnet50.parameters():
            param.requires_grad = False
        for param in resnet50.layer4.parameters():
            param.requires_grad = True

        # Modify the fully connected layer to output embeddings
        num_ftrs = resnet50.fc.in_features
        resnet50.fc = nn.Identity()

        # Define a new fully connected layer to output embeddings of desired size
        fc_layer = nn.Linear(num_ftrs, c.embedding_size)

        # Define the model as a sequential module with ResNet50 feature extractor and FC layer
        self.model = nn.Sequential(resnet50, fc_layer)
      
    def forward(self, x):
        return self.model(x)