import torch
from torch.utils.data import DataLoader, random_split

from data.synthetic import LinearRegressionDataset
from models.linear_regression import LinearRegressionModel

def main():
    # 1. 固定随机种子（方便复现）
    torch.manual_seed(42)

    # 2. 创建数据集
    dataset = LinearRegressionDataset()

    # 3. 划分训练集 / 验证集
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # 4. 创建 DataLoader
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # 5.创建模型
    model = LinearRegressionModel()

    # 6.损失函数
    criterion = torch.nn.MSELoss()

    # 7.优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # 8.开始训练
    num_epochs = 10
    for epoch in range(num_epochs):
        # === 训练阶段 ===
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            # 清空上一轮梯度
            optimizer.zero_grad()

            # 前向传播
            preds = model(batch_x)

            # 计算loss
            loss = criterion(preds, batch_y)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # === 验证阶段 ===
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # 9. 打印学到的参数
        weight = model.linear.weight.item()
        bias = model.linear.bias.item()

        print("\n训练结束")
        print(f"学到的 weight: {weight:.4f}")
        print(f"学到的 bias:   {bias:.4f}")


if __name__ == '__main__':
    main()