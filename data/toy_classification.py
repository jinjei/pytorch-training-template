import torch
from torch.utils.data import Dataset

class ToyClassificationDataset(Dataset):
    def __init__(self, num_samples=300):
        super().__init__()

        self.x = torch.randn(num_samples, 2)

        # 规则：如果 x1 + x2 > 0，则标签为 1，否则为 0
        # CrossEntropyLoss要求标签是整型类别索引，所以要转long
        self.y = (self.x[:, 0] + self.x[:, 1] > 0).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]