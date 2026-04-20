import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


class SeedManager:
    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CIFAR10DataModule:
    def __init__(self, batch_size=64, seed=42, val_ratio=0.2):
        self.batch_size = batch_size
        self.seed = seed
        self.val_ratio = val_ratio
        self.transform = transforms.ToTensor()

    def get_loaders(self):
        full_train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=self.transform
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=self.transform
        )

        train_size = int((1 - self.val_ratio) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size

        generator = torch.Generator().manual_seed(self.seed)
        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=generator
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_loader, val_loader, test_loader


class VanillaMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_layers):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


class Trainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=1e-3, epochs=10, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

    def train_one_epoch(self):
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        avg_acc = correct / total

        return avg_loss, avg_acc

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        avg_acc = correct / total

        return avg_loss, avg_acc

    @torch.no_grad()
    def test(self, test_loader):
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        avg_acc = correct / total

        return avg_loss, avg_acc

    def fit(self):
        best_val_acc = -1.0
        best_state_dict = None

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if best_state_dict is None or val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

                print(f"New BEST model at epoch {epoch} | val_acc = {val_acc * 100:.2f}%")

            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
            )

        self.model.load_state_dict(best_state_dict)

        print(f"\nBest Validation Accuracy: {best_val_acc * 100:.2f}%")

        return self.history


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [acc * 100 for acc in history["train_acc"]], label="Train Accuracy")
    plt.plot(epochs, [acc * 100 for acc in history["val_acc"]], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()


class Experiment:
    def __init__(self, config):
        self.config = config

    def run(self):
        SeedManager.set_seed(self.config["seed"])

        data_module = CIFAR10DataModule(
            batch_size=self.config["batch_size"],
            seed=self.config["seed"],
            val_ratio=self.config["val_ratio"]
        )

        train_loader, val_loader, test_loader = data_module.get_loaders()

        model = VanillaMLP(
            input_dim=3 * 32 * 32,
            num_classes=10,
            hidden_layers=self.config["hidden_layers"]
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=self.config["learning_rate"],
            epochs=self.config["epochs"]
        )

        history = trainer.fit()

        plot_history(history)

        test_loss, test_acc = trainer.test(test_loader)

        print("\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc * 100:.2f}%")

        return history, test_loss, test_acc


if __name__ == "__main__":
    config = {
        "seed": 42,
        "batch_size": 24,
        "learning_rate": 0.01,
        "hidden_layers": [512, 512, 256, 128],
        "epochs": 20,
        "val_ratio": 0.2
    }

    experiment = Experiment(config)
    history, test_loss, test_acc = experiment.run()