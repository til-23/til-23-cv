from torchvision.models import resnet50, ResNet50_Weights
from torch import cat
import torch.nn as nn

class SiameseNetwork(nn.Module):
    """
        https://github.com/pytorch/examples/tree/main/siamese_network

        BCE Loss
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = resnet50(ResNet50_Weights.DEFAULT)

        for ct, child in enumerate(self.resnet.children()):
            if ct < 6:
                for param in child.parameters():
                    param.requires_grad = False


        self.fc_in_features = self.resnet.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )


    def get_embeddings(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        output1 = self.get_embeddings(input1)
        output2 = self.get_embeddings(input2)
        output = cat((output1, output2), 1)
        output = self.fc(output)
        
        return output