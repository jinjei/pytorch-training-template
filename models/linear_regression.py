import torch.nn as nn

class LinearRegressionModel(nn.Module):
    # nn.Module : PyTorch 中“模型”的标准模板
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)