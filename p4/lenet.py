import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

class LeNetEncoder_special(nn.Module):
    def __init__(self):
        super(LeNetEncoder_special, self).__init__()

        self.encoder = nn.Sequential(
            # Conv 1
            nn.Conv2d(1, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(True),
            # Conv 2
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(True)
        )

        self.fc = nn.Linear(50*4*4, 500)
    
    def forward(self, data):
        out1 = self.encoder(data)
        out1 = out1.view((-1, 50*4*4))
        feature = self.fc(out1)

        return feature

class LeNetClassifier_special(nn.Module):
    def __init__(self):
        super(LeNetClassifier_special, self).__init__()

        self.class_classifier = nn.Sequential(
            nn.BatchNorm1d(500),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(500, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=-1)
        )
    
    def forward(self, data):
        out = self.class_classifier(data)

    

        return out

class LeNetEncoder(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetEncoder, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv layer
            # input [1 x 28 x 28]
            # output [20 x 12 x 12]
            nn.Conv2d(3, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # 2nd conv layer
            # input [20 x 12 x 12]
            # output [50 x 4 x 4]
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(50 * 4 * 4, 500)

    def forward(self, input):
        """Forward the LeNet."""
        conv_out = self.encoder(input)
        feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))
        return feat


class LeNetClassifier(nn.Module):
    """LeNet classifier model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetClassifier, self).__init__()
        self.fc2 = nn.Linear(500, 10)

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        return out


def main():
    import torch
    img_size = 28
    encoder = LeNetEncoder()
    classifier = LeNetClassifier()
    img = torch.ones((2, 1, 28, 28))

    feature = encoder(img)
    out = classifier(feature)
    print(out.size())
if __name__ == '__main__':
    main()