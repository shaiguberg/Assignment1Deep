import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import itertools


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
    def __init__(self, config):
        self.batch_size = config.get("batch_size", 64)
        self.seed = config.get("seed", 42)
        self.val_ratio = config.get("val_ratio", 0.2)

        if config.get("use_input_norm", False):
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)
                )])   # Input normalization
        else:
            self.transform = transforms.ToTensor()

    def get_loaders(self):
        full_train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
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

        return train_loader, val_loader


class AdvancedMLP(nn.Module):
    def __init__(self, input_dim, num_classes, config):
        super().__init__()
        layers = []
        prev_dim = input_dim

        hidden_layers = config["hidden_layers"]
        dropout_rate = config["dropout_rate"]
        use_bn = config.get("use_batch_norm", False)

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))  # ---Batch normalization ---
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))  # --- DropOut ---
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = config["epochs"]
        self.criterion = nn.CrossEntropyLoss()

        lr = config["learning_rate"]
        wd = config.get("weight_decay", 0)

        # Choose optimizer
        if config.get("AdamW"):
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif config.get("Adam"):
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd)

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

    def fit(self):
        best_val_acc = 0.0
        best_state_dict = None

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

                print(f" New BEST model at epoch {epoch} | val_acc = {val_acc:.4f}")

            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

        self.model.load_state_dict(best_state_dict)

        print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

        return self.history


class Experiment:
    def __init__(self, config):
        self.config = config

    def run(self):
        SeedManager.set_seed(self.config["seed"])

        data_module = CIFAR10DataModule(
            config=self.config
        )

        train_loader, val_loader = data_module.get_loaders()

        model = AdvancedMLP(
            input_dim=3 * 32 * 32,
            num_classes=10,
            config=self.config
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config
        )

        history = trainer.fit()
        return history


if __name__ == "__main__":
    # Base configuration
    base_config = {
        "seed": 42,
        "batch_size": 64,
        "learning_rate": 0.01,
        "hidden_layers": [512, 512, 256, 128],
        "epochs": 10,  # Recommended: lower epochs for grid search
        "val_ratio": 0.2
    }

    # Define the variations you want to test
    options = {
        "use_input_norm": [True, False],
        "use_batch_norm": [True, False],
        "dropout_rate": [0, 0.2, 0.5],
        "weight_decay": [0, 1e-4, 1e-3],
        "optimizer": ["SGD", "Adam", "AdamW"]  # Testing two main types
    }

    # Generate combinations
    keys = list(options.keys())
    combinations = list(itertools.product(*options.values()))

    results_summary = []

    # --- experiments loop ---
    for idx, combo in enumerate(combinations):
        current_setup = dict(zip(keys, combo))
        exp_config = base_config.copy()
        exp_config.update(current_setup)

        opt_name = exp_config.pop("optimizer")
        exp_config["AdamW"] = (opt_name == "AdamW")
        exp_config["Adam"] = (opt_name == "Adam")

        exp_name = (f"Opt: {opt_name:5} | "
                    f"BN: {str(current_setup['use_batch_norm']):5} | "
                    f"Drop: {current_setup['dropout_rate']:3} | "
                    f"WD: {current_setup['weight_decay']:6} | "
                    f"INorm: {str(current_setup['use_input_norm']):5}")

        print(f"\n🚀 [Exp {idx + 1}/{len(combinations)}] Running: {exp_name}")

        try:
            experiment = Experiment(exp_config)
            history = experiment.run()
            best_acc = max(history["val_acc"])

            print(f"✅ Finished {exp_name} -> Best Val Acc: {best_acc:.4f}")

            results_summary.append((exp_name, best_acc))
        except Exception as e:
            print(f"❌ Experiment failed: {exp_name}. Error: {e}")

    # --- all results ---
    print("\n" + "=" * 80)
    print(f"{'FULL EXPERIMENT LOG':^80}")
    print("=" * 80)
    print(f"{'Experiment Configuration':<65} | {'Best Val Acc':<12}")
    print("-" * 80)
    for name, acc in results_summary:
        print(f"{name:<65} | {acc:<12.4f}")

    # --- printing top 10 results ---
    results_summary.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 80)
    print(f"{'🏆 TOP 10 RESULTS 🏆':^80}")
    print("=" * 80)
    for i, (name, acc) in enumerate(results_summary[:10], 1):
        print(f"{i:2}. {name:<65} | {acc:.4f}")
    print("=" * 80)



