# synthetic: 人造数据 y = 3x + 2 + noise，需要模型学习线性回归拟合直线
import torch
from torch.utils.data import Dataset

# Dataset: PyTorch 提供的一个“数据集基类”, 只要继承它，并实现两个方法：
#  __len__ 和 __getitem__ , PyTorch 的 DataLoader就知道怎么批量读取数据了
class LinearRegressionDataset(Dataset):
    def __init__(self, num_samples=500):
        super().__init__()

        # 生成 x: 形状是 [num_samples, 1]
        self.x = torch.randn(num_samples, 1)
        # 生成噪声
        noise = 0.1 * torch.randn(num_samples, 1)
        # 构造 y = 3x + 2 + noise
        self.y = 3 * self.x + 2 + noise

    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]