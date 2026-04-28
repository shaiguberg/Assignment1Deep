import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


# =========================
# Seed
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Data
# =========================
def get_loaders(config):
    if config["use_input_norm"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])
    else:
        transform = transforms.ToTensor()

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_size = int((1 - config["val_ratio"]) * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(config["seed"])

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    return train_loader, val_loader


# =========================
# Model
# =========================
class AdvancedMLP(nn.Module):
    def _init_(self, config):
        super()._init_()

        layers = []
        prev_dim = 3 * 32 * 32

        for hidden_dim in config["hidden_layers"]:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if config["use_batch_norm"]:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())

            if config["dropout_rate"] > 0:
                layers.append(nn.Dropout(config["dropout_rate"]))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 10))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


# =========================
# Train
# =========================
def train(model, train_loader, val_loader, config, device):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    best_val_acc = -1
    best_state = None

    for epoch in range(config["epochs"]):
        model.train()

        total = 0
        correct = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total

        # validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)

                preds = torch.argmax(out, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    model.load_state_dict(best_state)

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

    return model


# =========================
# Main
# =========================
def main():
    config = {
        "seed": 42,
        "batch_size": 24,
        "learning_rate": 0.01,
        "hidden_layers": [512, 512, 256, 128],
        "epochs": 20,
        "val_ratio": 0.2,

        "dropout_rate": 0.1,
        "use_batch_norm": True,
        "use_input_norm": True,
        "weight_decay": 0.0
    }

    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader = get_loaders(config)

    model = AdvancedMLP(config).to(device)

    model = train(model, train_loader, val_loader, config, device)

    # =========================
    # SAVE MODEL (important!)
    # =========================
    torch.save(model.state_dict(), "best_model.pth")
    print("Model saved to best_model.pth")


if __name__ == "__main__":
    main()