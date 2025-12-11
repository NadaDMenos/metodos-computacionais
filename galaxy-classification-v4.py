import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
import kornia.augmentation as K


beginning = time.time() 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
h5_path = "C:/Users/roger/OneDrive/Área de Trabalho/coisas da universidade/2-2025/Metodos Computacionais B/3rd PROJECT/Galaxy10_DECals.h5"

# ============================================================
# 1. Dataset Loader for Galaxy10 DECals (.h5 file)
# ============================================================

def SplitDataset(h5_path, ratio=0.8):
    with h5py.File(h5_path, "r") as f: 
        images = f["images"][:]
        labels = f["ans"][:]

    train_idx, val_idx = train_test_split(np.arange(labels.shape[0]), train_size=ratio, shuffle=True, stratify=labels)

    train_images = images[train_idx]
    val_images  = images[val_idx]

    train_labels = labels[train_idx]
    val_labels  = labels[val_idx]
    return train_images, train_labels, val_images, val_labels

class Galaxy10DECals(Dataset): 
    def __init__(self, images, labels): 
        self.images = images
        self.labels = labels

    def __len__(self): 
        return len(self.labels) 
    
    def __getitem__(self, idx): 
        return ToTensor()(self.images[idx]), self.labels[idx]

# ============================================================
# 2. CNN Model
# ============================================================

class GalaxyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def ConvolutionBlock(in_channels, out_channels, kernel_size, dropout):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same', bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),

                nn.Dropout(dropout),
                nn.MaxPool2d(2)
            )
        
        def BottleneckBlock(in_channel, out_channel, kernel_size, dropout):
            in_between = out_channel // 4

            return nn.Sequential(
                nn.Conv2d(in_channel, in_between, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_between),
                nn.ReLU(),

                nn.Conv2d(in_between, in_between, kernel_size=kernel_size, padding='same', bias=False),
                nn.BatchNorm2d(in_between),
                nn.ReLU(),

                nn.Conv2d(in_between, out_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),

                nn.Dropout(dropout),
                nn.MaxPool2d(2)
    )
        
        def OutputBlock(in_channels, out_channels, dropout):
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_channels, num_classes)
            )
        
        self.featuremap1 = ConvolutionBlock(3, 32, 3, 0.02)
        self.featuremap2 = BottleneckBlock(32, 64, 5, 0.03)
        self.featuremap3 = BottleneckBlock(64, 128, 3, 0.04)
        self.featuremap4 = ConvolutionBlock(128, 256, 3, 0.06)
        self.featuremap5 = BottleneckBlock(256, 512, 3, 0.07)
        self.featuremap6 = BottleneckBlock(512, 1024, 3, 0.08)

        self.classification = OutputBlock(1024, 512, 0.25)

    def forward(self, x):
        x = self.featuremap1(x)
        x = self.featuremap2(x)
        x = self.featuremap3(x)
        x = self.featuremap4(x)
        x = self.featuremap5(x)
        x = self.featuremap6(x)
        
        x = self.classification(x)
        return x

# ============================================================
# 3. Transformations
# ============================================================

class DataAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=150., p=0.5),
            K.RandomResizedCrop(size=(256, 256), scale=(0.75, 1.0), p=0.5),
            K.ColorJiggle(brightness=0.15, contrast=0.15, saturation=0.15, p=0.5)
        )

    def forward(self, x):
        return self.transforms(x)

data_augmenter = DataAugmentation().to(device)
normalize = K.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]).to(device)

# ============================================================
# 4. Subsets definition
# ============================================================

train_images, train_labels, val_images, val_labels = SplitDataset(h5_path, ratio=0.7)

train_dataset = Galaxy10DECals(train_images, train_labels)
val_dataset = Galaxy10DECals(val_images, val_labels)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

model = GalaxyCNN(num_classes=10)
model = model.to(device)

print(model)
print("\nUsing device:", device,'\n')

# ============================================================
# 5. Main Execution
# ============================================================

def Learning(epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    for epoch in range(epochs):
        # -----------------------
        # Training mode
        # -----------------------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs = data_augmenter(imgs)
            imgs = normalize(imgs)

            optimizer.zero_grad()
            outputs = model(imgs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # -----------------------
        # Validation mode
        # -----------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                imgs = normalize(imgs)

                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # -----------------------
        # Epoch summary
        # -----------------------
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

Learning(epochs=30)
print("Training complete.")

print(f'Time elapsed: {time.strftime("%M:%S", time.gmtime(time.time() - beginning))}') 