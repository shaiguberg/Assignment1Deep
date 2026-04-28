import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


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