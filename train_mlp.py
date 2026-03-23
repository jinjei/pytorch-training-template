import torch
from torch.utils.data import DataLoader, random_split

from data.toy_classification import ToyClassificationDataset
from models.mlp import MLPClassifier

def calculate_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total

def main():
    torch.manual_seed(42)

    # 1. 数据集
    dataset = ToyClassificationDataset()

    # 2. 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # 3. DataLoader
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    # 4. model
    model = MLPClassifier()

    # 5. Loss and Optim
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    num_epochs = 30

    for epoch in range(num_epochs):
        # === train ===
        model.train()
        train_loss = 0
        train_acc = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += calculate_accuracy(logits, batch_y)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)

        # === val ===
        model.eval()
        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss += loss.item()
                val_acc += calculate_accuracy(logits, batch_y)

            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = val_acc / len(val_loader)

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train Acc: {avg_train_acc:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {avg_val_acc:.4f}"
            )

if __name__ == "__main__":
    main()