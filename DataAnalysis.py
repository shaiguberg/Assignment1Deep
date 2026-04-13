import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from collections import Counter

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