import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
"""
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.domain = nn.Sequential(
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(True),
            #nn.Linear(500, 2),
            nn.Linear(500, 1),
            #nn.LogSoftmax(dim=-1)
            nn.Sigmoid()
        )
    
    def forward(self, feature):
        domain = self.domain(feature)

        return domain
"""
class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims=500, hidden_dims=500, output_dims=2):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out

def main():
    import torch
    domain_class = Discriminator()
    feat = torch.ones((2, 500))

    out = domain_class(feat)
    print(out.size())
if __name__ == '__main__':
    main()

