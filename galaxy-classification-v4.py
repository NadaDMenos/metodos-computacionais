import time
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor
import kornia.augmentation as K


beginning = time.time() 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
h5_path = "C:/Users/roger/OneDrive/Área de Trabalho/coisas da universidade/2-2025/Metodos Computacionais B/3rd PROJECT/Galaxy10_DECals.h5"

# ============================================================
# 1. Dataset Loader for Galaxy10 DECals (.h5 file)
# ============================================================

class Galaxy10DECals(Dataset): 
    def __init__(self, h5_path): 
        super().__init__() 
        with h5py.File(h5_path, "r") as f: 
            self.images = f["images"][:]
            self.labels = f["ans"][:] 

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
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool2d(2)
            )
        
        def BottleneckBlock(in_ch, out_ch, kernel_size, dropout):
            mid = out_ch // 4

            return nn.Sequential(
                nn.Conv2d(in_ch, mid, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(),

                nn.Conv2d(mid, mid, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(),

                nn.Conv2d(mid, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),

                nn.MaxPool2d(2),
                nn.Dropout(dropout)
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
        
        self.featuremap1 = ConvolutionBlock(3, 32, 4, 0.1)
        self.featuremap2 = BottleneckBlock(32, 64, 4, 0.1)
        self.featuremap3 = ConvolutionBlock(64, 128, 3, 0.15)
        self.featuremap4 = BottleneckBlock(128, 256, 4, 0.15)
        self.featuremap5 = ConvolutionBlock(256, 512, 3, 0.2)
        self.featuremap6 = BottleneckBlock(512, 1024, 3, 0.2)

        self.classification = OutputBlock(1024, 1024, 0.4)

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
            K.RandomRotation(degrees=180., p=0.5),
            K.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), p=0.5),
            K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=0.5),
            K.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        )

    def forward(self, x):
        return self.transforms(x)

data_augmenter = DataAugmentation().to(device)
normalize = K.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]).to(device)

# ============================================================
# 4. Subsets definition
# ============================================================

def DatasetSplit(dataset, split):
    data_length = len(dataset)
    train_size = int(split * data_length)
    val_size = data_length - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

dataset = Galaxy10DECals(h5_path)
train_dataset, val_dataset = DatasetSplit(dataset, 0.8)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
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
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

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

Learning(epochs=20)
print("Training complete.")

print(f'Time elapsed: {time.strftime("%M:%S", time.gmtime(time.time() - beginning))}')
