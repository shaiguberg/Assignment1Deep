import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


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
                )
            ])
        else:
            self.transform = transforms.ToTensor()

    def get_loaders(self):
        full_train_dataset = torchvision.datasets.CIFAR10(
            root="./data",
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
        dropout_rate = config.get("dropout_rate", 0.0)
        use_batch_norm = config.get("use_batch_norm", False)

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

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

        learning_rate = config["learning_rate"]
        weight_decay = config.get("weight_decay", 0.0)

        if config.get("AdamW", False):
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif config.get("Adam", False):
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )

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
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }

                print(f"New BEST model at epoch {epoch} | val_acc = {val_acc * 100:.2f}%")

            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
            )

        self.model.load_state_dict(best_state_dict)

        print(f"\nBest Validation Accuracy: {best_val_acc * 100:.2f}%")

        return self.history


class Experiment:
    def __init__(self, config):
        self.config = config

    def run(self):
        SeedManager.set_seed(self.config["seed"])

        data_module = CIFAR10DataModule(self.config)
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


def build_optimizer_flags(optimizer_name):
    return {
        "Adam": optimizer_name == "Adam",
        "AdamW": optimizer_name == "AdamW"
    }


def run_single_change_experiments(base_config, dropout_values, weight_decay_values):
    results_summary = []

    single_change_experiments = []

    single_change_experiments.append({
        "name": "Baseline"
    })

    single_change_experiments.append({
        "name": "Only Input Norm",
        "use_input_norm": True
    })

    single_change_experiments.append({
        "name": "Only BatchNorm",
        "use_batch_norm": True
    })

    for d in dropout_values:
        if d > 0:
            single_change_experiments.append({
                "name": f"Only Dropout={d}",
                "dropout_rate": d
            })

    for wd in weight_decay_values:
        if wd > 0:
            single_change_experiments.append({
                "name": f"Only WeightDecay={wd}",
                "weight_decay": wd
            })

    for opt in ["Adam", "AdamW"]:
        exp = {
            "name": f"Only Optimizer={opt}"
        }
        exp.update(build_optimizer_flags(opt))
        single_change_experiments.append(exp)

    print("\n" + "=" * 50)
    print("RUNNING: SINGLE-CHANGE EXPERIMENTS")
    print("=" * 50)

    for exp_setup in single_change_experiments:
        current_config = base_config.copy()

        current_config.update({
            "use_input_norm": False,
            "use_batch_norm": False,
            "dropout_rate": 0.0,
            "weight_decay": 0.0,
            "Adam": False,
            "AdamW": False
        })

        current_config.update(exp_setup)

        print(f"\nRunning Experiment: {current_config['name']}")
        experiment = Experiment(current_config)
        history = experiment.run()

        best_acc = max(history["val_acc"])

        results_summary.append({
            "group": "single_change",
            "name": current_config["name"],
            "best_val_acc": best_acc,
            "config": current_config.copy()
        })

    return results_summary


def run_all_combinations(base_config, dropout_values, weight_decay_values, optimizers):
    results_summary = []

    all_combinations = list(itertools.product(
        [False, True],          # use_input_norm
        [False, True],          # use_batch_norm
        dropout_values,         # dropout_rate
        weight_decay_values,    # weight_decay
        optimizers              # optimizer
    ))

    print("\n" + "=" * 50)
    print("RUNNING: ALL COMBINATIONS")
    print("=" * 50)
    print(f"Total combinations: {len(all_combinations)}")

    for i, (use_input_norm, use_batch_norm, dropout_rate, weight_decay, optimizer_name) in enumerate(all_combinations, start=1):
        current_config = base_config.copy()

        current_config.update({
            "name": (
                f"Combo {i}: "
                f"InputNorm={use_input_norm}, "
                f"BatchNorm={use_batch_norm}, "
                f"Dropout={dropout_rate}, "
                f"WeightDecay={weight_decay}, "
                f"Optimizer={optimizer_name}"
            ),
            "use_input_norm": use_input_norm,
            "use_batch_norm": use_batch_norm,
            "dropout_rate": dropout_rate,
            "weight_decay": weight_decay,
            "Adam": False,
            "AdamW": False
        })

        current_config.update(build_optimizer_flags(optimizer_name))

        print(f"\nRunning Experiment {i}/{len(all_combinations)}")
        print(current_config["name"])

        experiment = Experiment(current_config)
        history = experiment.run()

        best_acc = max(history["val_acc"])

        results_summary.append({
            "group": "all_combinations",
            "name": current_config["name"],
            "best_val_acc": best_acc,
            "config": current_config.copy()
        })

    return results_summary


if __name__ == "__main__":
    base_config = {
        "seed": 42,
        "batch_size": 24,
        "learning_rate": 0.01,
        "hidden_layers": [512, 512, 256, 128],
        "epochs": 20,
        "val_ratio": 0.2,
        "dropout_rate": 0.0,
        "use_batch_norm": False,
        "use_input_norm": False,
        "weight_decay": 0.0,
        "Adam": False,
        "AdamW": False
    }

    dropout_values = [0.0, 0.1, 0.3, 0.5]
    weight_decay_values = [0.0, 1e-5, 1e-4, 1e-3]
    optimizers = ["SGD", "Adam", "AdamW"]

    all_results = []

    single_change_results = run_single_change_experiments(
        base_config=base_config,
        dropout_values=dropout_values,
        weight_decay_values=weight_decay_values
    )
    all_results.extend(single_change_results)

    combination_results = run_all_combinations(
        base_config=base_config,
        dropout_values=dropout_values,
        weight_decay_values=weight_decay_values,
        optimizers=optimizers
    )
    all_results.extend(combination_results)

    all_results.sort(key=lambda x: x["best_val_acc"], reverse=True)

    print("\n" + "=" * 70)
    print("FINAL EXPERIMENT SUMMARY")
    print("=" * 70)

    for result in all_results:
        print(f"{result['group']:18} | {result['best_val_acc'] * 100:.2f}% | {result['name']}")

    best_result = all_results[0]

    print("\n" + "=" * 70)
    print("BEST OVERALL EXPERIMENT")
    print("=" * 70)
    print(f"Name: {best_result['name']}")
    print(f"Best Validation Accuracy: {best_result['best_val_acc'] * 100:.2f}%")
    print("Config:")
    for key, value in best_result["config"].items():
        print(f"  {key}: {value}")