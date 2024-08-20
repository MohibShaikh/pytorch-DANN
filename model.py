import torch.nn as nn
import torch.nn.functional as F
from utils import ReverseLayerF

class Extractor3D(nn.Module):
    def __init__(self):
        super(Extractor3D, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the classifier
        return x

class Classifier3D(nn.Module):
    def __init__(self, input_dim):
        super(Classifier3D, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=10)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

class Discriminator3D(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator3D, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2)
        )

    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        x = self.discriminator(reversed_input)
        return x
