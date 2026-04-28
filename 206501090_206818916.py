import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import itertools
from collections import Counter
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# Define a basic transformation: convert images to tensors
transform = transforms.ToTensor()

# Load the CIFAR-10 training dataset
full_train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Define split sizes (e.g., 80% train, 20% validation)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

# Split the dataset into train and validation sets
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)   # Shuffle training data
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)      # Do not shuffle validation data
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



print(len(train_dataset))  # number of training images
print(len(test_dataset))   # number of test images

image, label = train_dataset[5]

print(image.shape)  # shape of the image
print(label)        # class label

# Extract all labels from the dataset
labels = [label for _, label in train_dataset]

# Count occurrences of each label
label_counts = Counter(labels)

# Print results
for label, count in label_counts.items():
    print(f"Class {label}: {count}")


# Get class names
class_names = full_train_dataset.classes

# Define border width in pixels
border = 4

# Store mean border RGB for each class
border_rgb_by_class = defaultdict(list)

# Loop over all images in the full training dataset
for image, label in full_train_dataset:
    # Extract border regions
    top = image[:, :border, :]
    bottom = image[:, -border:, :]
    left = image[:, border:-border, :border]
    right = image[:, border:-border, -border:]

    # Flatten each border region to [3, number_of_pixels]
    top = top.reshape(3, -1)
    bottom = bottom.reshape(3, -1)
    left = left.reshape(3, -1)
    right = right.reshape(3, -1)

    # Concatenate all border pixels
    border_pixels = torch.cat([top, bottom, left, right], dim=1)

    # Compute mean RGB of the border
    mean_rgb = border_pixels.mean(dim=1)

    # Save result for this class
    border_rgb_by_class[label].append(mean_rgb)

# Print mean and std RGB for each class
for label in range(len(class_names)):
    rgb_values = torch.stack(border_rgb_by_class[label])

    class_mean = rgb_values.mean(dim=0)
    class_std = rgb_values.std(dim=0)

    print(f"Class: {class_names[label]}")
    print(f"Mean border RGB: {class_mean}")
    print(f"Std border RGB: {class_std}")
    print()

# Print standard deviation in a clear format
for label in range(len(class_names)):
    rgb_values = torch.stack(border_rgb_by_class[label])

    class_std = rgb_values.std(dim=0)

    print(f"{class_names[label]}:")
    print(f"  Std Red:   {class_std[0].item():.3f}")
    print(f"  Std Green: {class_std[1].item():.3f}")
    print(f"  Std Blue:  {class_std[2].item():.3f}")
    print()

# Store per-class mean and std values
mean_r, mean_g, mean_b = [], [], []
std_r, std_g, std_b = [], [], []

for label in range(len(class_names)):
    rgb_values = torch.stack(border_rgb_by_class[label])

    class_mean = rgb_values.mean(dim=0)
    class_std = rgb_values.std(dim=0)

    mean_r.append(class_mean[0].item())
    mean_g.append(class_mean[1].item())
    mean_b.append(class_mean[2].item())

    std_r.append(class_std[0].item())
    std_g.append(class_std[1].item())
    std_b.append(class_std[2].item())

# Define x positions
x = np.arange(len(class_names))
width = 0.25

# Create grouped bar chart
plt.figure(figsize=(12, 6))
plt.bar(x - width, mean_r, width, yerr=std_r, capsize=4, color='red', label='Red channel')
plt.bar(x, mean_g, width, yerr=std_g, capsize=4, color='green', label='Green channel')
plt.bar(x + width, mean_b, width, yerr=std_b, capsize=4, color='blue', label='Blue channel')

# Add labels and title
plt.xticks(x, class_names, rotation=45)
plt.ylabel("Mean border intensity")
plt.xlabel("Class")
plt.title("Mean Border RGB Values per Class with Standard Deviation")
plt.legend()
plt.tight_layout()
plt.show()





def analyze_cifar_colors_with_averages(dataset):
    # Dictionary to store metrics for each of the 10 classes
    class_ranges = {i: [] for i in range(10)}
    class_midpoints = {i: [] for i in range(10)}

    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



    for img, label in dataset:
        # img shape is [3, 32, 32] (C, H, W)

        # Get Max and Min RGB values per pixel
        max_rgb, _ = torch.max(img, dim=0)
        min_rgb, _ = torch.min(img, dim=0)

        # Metric A: Range (Max - Min) -> Related to Color Saturation
        pixel_range = max_rgb - min_rgb

        # Metric B: Midpoint (Max + Min) / 2 -> Related to Lightness/Brightness
        pixel_midpoint = (max_rgb + min_rgb) / 2.0

        # Store the average value for this specific image
        class_ranges[label].append(torch.mean(pixel_range).item())
        class_midpoints[label].append(torch.mean(pixel_midpoint).item())

    # Aggregate Results per Class
    class_stats = {}

    for i in range(10):
        avg_r = np.mean(class_ranges[i])
        avg_m = np.mean(class_midpoints[i])
        name = label_names[i]
        class_stats[name] = {'range': avg_r, 'midpoint': avg_m}

    # 3. Calculate Global Averages across all classes
    all_ranges_values = [stats['range'] for stats in class_stats.values()]
    all_midpoints_values = [stats['midpoint'] for stats in class_stats.values()]

    global_mean_range = np.mean(all_ranges_values)
    global_mean_midpoint = np.mean(all_midpoints_values)

    # Print Results Table
    print(f"\n{'Class':<12} | {'Avg Range (Saturation)':<22} | {'Avg Midpoint (Brightness)':<22}")
    print("-" * 65)
    for name, stats in class_stats.items():
        print(f"{name:<12} | {stats['range']:.4f} {' ':<16} | {stats['midpoint']:.4f}")

    print("-" * 65)
    print(f"{'Global Mean':<12} | {global_mean_range:.4f} {' ':<16} | {global_mean_midpoint:.4f}")

    return class_stats, global_mean_range, global_mean_midpoint


# Run the analysis
class_stats, g_range, g_midpoint = analyze_cifar_colors_with_averages(full_train_dataset)

# Visualization
labels = list(class_stats.keys())
ranges = [class_stats[l]['range'] for l in labels]
midpoints = [class_stats[l]['midpoint'] for l in labels]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 7))
rects1 = ax.bar(x - width / 2, ranges, width, label='Avg Range per Class (Saturation)', color='#1f77b4', alpha=0.7)
rects2 = ax.bar(x + width / 2, midpoints, width, label='Avg Midpoint per Class (Brightness)', color='#ff7f0e',
                alpha=0.7)

# --- Add Dashed Lines for Global Averages ---
ax.axhline(y=g_range, color='#1f77b4', linestyle='--', linewidth=2, label=f'Global Avg Range ({g_range:.3f})')
ax.axhline(y=g_midpoint, color='#ff7f0e', linestyle='--', linewidth=2, label=f'Global Avg Midpoint ({g_midpoint:.3f})')


# Add values on top of bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)


autolabel(rects1)
autolabel(rects2)

ax.set_ylabel('Values (0.0 to 1.0)')
ax.set_title('Color Metrics by CIFAR-10 Class with Global Averages')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.set_ylim(0, 0.7)  # Added space for labels on top
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

#----------------- part 2 - vanilla -----------------

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
        "hidden_layers": [512,256, 256, 128],
        "epochs": 20,
        "val_ratio": 0.2
    }

    experiment = Experiment(config)
    history, test_loss, test_acc = experiment.run()


#---------------- part 2 - improved -------------
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



#-------------------part 3---------------------

class AdvancedMLP(nn.Module):
    def __init__(self, input_dim, num_classes, config):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in config["hidden_layers"]:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if config["use_batch_norm"]:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())

            if config["dropout_rate"] > 0:
                layers.append(nn.Dropout(config["dropout_rate"]))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


def main():
    config = {
        "hidden_layers": [512, 512, 256, 128],
        "dropout_rate": 0.1,
        "use_batch_norm": True,
        "batch_size": 24
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )

    true_labels = np.array(test_dataset.targets)

    human_counts = np.load(
        r"C:\Users\User\Desktop\דברים\לימודים\שנה ג\מבוא ללמידה עמוקה\Assignment_1\cifar-10h-master\data\cifar10h-counts.npy"
    )

    human_labels = np.argmax(human_counts, axis=1)
    mismatch_indices = np.where(human_labels != true_labels)[0]

    print("Number of mismatches:", len(mismatch_indices))
    print("Mismatch indices:")
    print(mismatch_indices)

    model = AdvancedMLP(
        input_dim=3 * 32 * 32,
        num_classes=10,
        config=config
    ).to(device)

    model.load_state_dict(
        torch.load("best_cifar10_model.pth", map_location=device)
    )

    model.eval()

    # =====================================================
    # Part 3.1 - Run model only on the 79 mismatch images
    # =====================================================
    images_79 = []
    true_labels_79 = []
    human_labels_79 = []

    for idx in mismatch_indices:
        image, true_label = test_dataset[idx]

        images_79.append(image)
        true_labels_79.append(true_label)
        human_labels_79.append(human_labels[idx])

    images_79 = torch.stack(images_79).to(device)
    true_labels_79 = np.array(true_labels_79)
    human_labels_79 = np.array(human_labels_79)

    with torch.no_grad():
        outputs_79 = model(images_79)
        probabilities_79 = torch.softmax(outputs_79, dim=1)
        model_predictions_79 = torch.argmax(probabilities_79, dim=1).cpu().numpy()

    correct_vs_true = np.sum(model_predictions_79 == true_labels_79)
    agree_with_humans = np.sum(model_predictions_79 == human_labels_79)

    print("\n==============================")
    print("Results on the 79 mismatch images")
    print("==============================")
    print(f"Model correct vs TRUE labels: {correct_vs_true} / {len(mismatch_indices)}")
    print(f"Model agrees with HUMANS: {agree_with_humans} / {len(mismatch_indices)}")

    true_percent = correct_vs_true / len(mismatch_indices) * 100
    human_percent = agree_with_humans / len(mismatch_indices) * 100

    plt.figure()
    labels = ["True Labels", "Human Labels"]
    values = [true_percent, human_percent]

    plt.bar(labels, values)
    plt.ylabel("Agreement Percentage (%)")
    plt.title("Model Agreement: True Labels vs Human Labels")
    plt.ylim(0, 100)

    for i, v in enumerate(values):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center")

    plt.show()

    print("\nDetailed results:")
    class_names = test_dataset.classes

    for i, idx in enumerate(mismatch_indices):
        true_label = true_labels_79[i]
        human_label = human_labels_79[i]
        model_label = model_predictions_79[i]

        print(
            f"Image index: {idx} | "
            f"True: {true_label} ({class_names[true_label]}) | "
            f"Human: {human_label} ({class_names[human_label]}) | "
            f"Model: {model_label} ({class_names[model_label]})"
        )

    # =====================================================
    # Part 3.2a - Model probability for the TRUE label
    # for all 10,000 test images
    # =====================================================
    all_model_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_model_probs.append(probs.cpu())

    all_model_probs = torch.cat(all_model_probs, dim=0).numpy()

    indices = np.arange(len(true_labels))

    model_true_label_probs = all_model_probs[indices, true_labels]

    print("\n==============================")
    print("Part 3.2a")
    print("==============================")
    print("Number of model true-label probabilities:", len(model_true_label_probs))
    print("First 10 values:")
    print(model_true_label_probs[:10])
    print("Min probability:", model_true_label_probs.min())
    print("Max probability:", model_true_label_probs.max())

    np.save("model_true_label_probs.npy", model_true_label_probs)
    print("Saved to: model_true_label_probs.npy")

    # =====================================================
    # Part 3.2b - Human probability for the TRUE label
    # for all 10,000 test images
    # =====================================================
    human_probs = np.load(
        r"C:\Users\User\Desktop\דברים\לימודים\שנה ג\מבוא ללמידה עמוקה\Assignment_1\cifar-10h-master\data\cifar10h-probs.npy"
    )

    human_true_label_probs = human_probs[indices, true_labels]

    print("\n==============================")
    print("Part 3.2b")
    print("==============================")
    print("Number of human true-label probabilities:", len(human_true_label_probs))
    print("First 10 values:")
    print(human_true_label_probs[:10])
    print("Min probability:", human_true_label_probs.min())
    print("Max probability:", human_true_label_probs.max())

    np.save("human_true_label_probs.npy", human_true_label_probs)
    print("Saved to: human_true_label_probs.npy")

    # =====================================================
    # Part 3.2c - Scatter Plot: Model vs Human probabilities
    # =====================================================

    plt.figure()

    plt.scatter(
        model_true_label_probs,
        human_true_label_probs,
        alpha=0.4,
        s = 5
    )

    plt.xlabel("Model Probability for True Label")
    plt.ylabel("Human Probability for True Label")
    plt.title("Model vs Human Confidence (True Label)")

    plt.grid(True)

    plt.show()

    # =====================================================
    # Extra plot - Model true-label probability
    # vs Human maximum probability
    # =====================================================

    human_probs = np.load(
        r"C:\Users\User\Desktop\דברים\לימודים\שנה ג\מבוא ללמידה עמוקה\Assignment_1\cifar-10h-master\data\cifar10h-probs.npy"
    )

    # Highest human probability for each image
    human_max_probs = np.max(human_probs, axis=1)

    print("\n==============================")
    print("Human max probabilities")
    print("==============================")
    print("Number of human max probabilities:", len(human_max_probs))
    print("First 10 values:")
    print(human_max_probs[:10])
    print("Min probability:", human_max_probs.min())
    print("Max probability:", human_max_probs.max())

    np.save("human_max_probs.npy", human_max_probs)
    print("Saved to: human_max_probs.npy")

    plt.figure()

    plt.scatter(
        model_true_label_probs,
        human_max_probs,
        alpha=0.3,
        s=3
    )

    plt.xlabel("Model Probability for True Label")
    plt.ylabel("Human Highest Probability")
    plt.title("Model True-Label Probability vs Human Highest Probability")
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()


#-------------------- bonus -----------------
df = pd.read_csv(
    r"C:\Users\User\Desktop\דברים\לימודים\שנה ג\מבוא ללמידה עמוקה\Assignment_1\cifar-10h-master\data\cifar10h-raw\cifar10h-raw.csv"
)
df = df[df["reaction_time"] <= 40000]

# Load model probabilities file
model_probs = np.load("model_true_label_probs.npy")

# Compute average reaction time per image
avg_reaction_time = df.groupby("cifar10_test_test_idx")["reaction_time"].mean()

# Keep only CIFAR-10 test images 0–9999 and sort by image index
avg_reaction_time = avg_reaction_time.loc[0:9999].sort_index()
reaction_times = avg_reaction_time.values

print("Number of reaction times:", len(reaction_times))
print("Number of model probabilities:", len(model_probs))

# Correlation metrics
pearson_corr, pearson_p = pearsonr(model_probs, reaction_times)
spearman_corr, spearman_p = spearmanr(model_probs, reaction_times)

print("\n===== Correlation Results =====")
print(f"Pearson correlation:  {pearson_corr:.4f}")
print(f"Pearson p-value:      {pearson_p:.4e}")

print(f"Spearman correlation: {spearman_corr:.4f}")
print(f"Spearman p-value:     {spearman_p:.4e}")

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(model_probs, reaction_times, alpha=0.3)
plt.xlabel("Model probability for true label")
plt.ylabel("Average human reaction time")
plt.title("Model Confidence vs. Human Reaction Time")
plt.grid(True)

n_bins = 20
bins = np.linspace(0, 1, n_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2

bin_means = []
for i in range(n_bins):

    mask = (model_probs >= bins[i]) & (model_probs < bins[i + 1])
    if mask.any():
        bin_means.append(reaction_times[mask].mean())
    else:
        bin_means.append(np.nan)

plt.plot(bin_centers, bin_means, color='red', linewidth=3, label='Average Trend')
plt.legend()

plt.show()