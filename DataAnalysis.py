import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict

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