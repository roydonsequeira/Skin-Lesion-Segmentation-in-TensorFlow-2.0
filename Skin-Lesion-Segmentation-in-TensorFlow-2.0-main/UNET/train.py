import os
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim import Adam
from torchmetrics import Precision, Dice
from torchmetrics.detection import IntersectionOverUnion as IoU
from torchmetrics.classification import Recall
from model import build_unet
from metrics import dice_coef
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time  # Import the time module

import csv

dataset_path = "C:\\Users\\roy\\OneDrive\\Desktop\\FYP"

H = 256
W = 256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_data(dataset_path, split=0.2):
    images = sorted(glob(os.path.join(dataset_path, "ISIC2018_Task1-2_Training_Input", "*.jpg")))
    masks = sorted(glob(os.path.join(dataset_path, "ISIC2018_Task1_Training_GroundTruth", "*.png")))

    if len(images) == 0 or len(masks) == 0:
        raise ValueError("No images or masks found in the specified dataset path.")

    test_size = int(len(images) * split)

    if test_size == 0:
        test_size = 1  # Ensure test_size is at least 1

    train_x, valid_x = train_test_split(images, test_size=test_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=test_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

class CustomDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_path, y_path = self.X[idx], self.Y[idx]
        x, y = read_image(x_path), read_mask(y_path)

        if self.transform:
            x, y = self.transform(x, y)

        return x, y

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)  # Add a batch dimension for PyTorch
    return x

def transform(x, y):
    x = torch.from_numpy(x.transpose((2, 0, 1)))  # Channels first for PyTorch
    if y.shape[1] == 1:
        y = torch.from_numpy(y.squeeze(1))  # Squeeze the extra dimension
    return x, y

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    create_dir("files")

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 4
    lr = 1e-4
    num_epoch = 100
    model_path = "C:\\Users\\roy\\OneDrive\\Desktop\\FYP\\model.pth"
    csv_path = "C:\\Users\\roy\\OneDrive\\Desktop\\FYP\\data.csv"

    train_losses = [] 


    dataset_path = "C:\\Users\\roy\\OneDrive\\Desktop\\FYP"

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    train_dataset = CustomDataset(train_x, train_y, transform=transform)
    valid_dataset = CustomDataset(valid_x, valid_y, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_unet((3, H, W))
    model.to(device)  # Move model to GPU if available

    metrics = [dice_coef, IoU(), Recall(task="binary"), Precision(task="binary")]  # Assuming 2 classes
    optimizer = Adam(model.parameters(), lr=lr)

    writer = SummaryWriter()  # TensorBoard equivalent

    start_time = time.time()  # Record the start time

    for epoch in range(num_epoch):
        model.train()

        # Use tqdm to create a progress bar for training
        train_loader = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epoch}, Training', dynamic_ncols=True)

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)  # Move data to GPU if available

            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.BCEWithLogitsLoss()(outputs, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()

        # Use tqdm to create a progress bar for validation
        with torch.no_grad():
            total_val_loss = 0.0
            valid_loader = tqdm(valid_loader, desc=f'Epoch {epoch + 1}/{num_epoch}, Validation', dynamic_ncols=True)
            for val_images, val_masks in valid_loader:
                val_images, val_masks = val_images.to(device), val_masks.to(device)  # Move data to GPU if available

                val_outputs = model(val_images)
                val_loss = nn.BCEWithLogitsLoss()(val_outputs, val_masks)
                total_val_loss += val_loss.item()

        average_val_loss = total_val_loss / len(valid_loader)
        print(f"Epoch {epoch + 1}/{num_epoch}, Validation Loss: {average_val_loss}")

        # Logging to TensorBoard
        writer.add_scalar('Loss/Validation', average_val_loss, epoch)

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time

    print(f"Training complete. Elapsed time: {elapsed_time:.2f} seconds")

    # Save the model
    torch.save(model.state_dict(), model_path)

    csv_file_path = "C:\\Users\\roy\\OneDrive\\Desktop\\FYP\\training_losses.csv"
    with open(csv_file_path, mode='w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Training Loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for epoch, loss in enumerate(train_losses):
            writer.writerow({'Epoch': epoch + 1, 'Training Loss': loss})

    print(f"Training losses saved to {csv_file_path}")