import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, num_classes=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)