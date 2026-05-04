import os
import cv2
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

DATA_DIR = "/kaggle/input/datasets/kipshidze/shoplifting-video-dataset"

BATCH_SIZE = 8
FRAMES_PER_VIDEO = 10
EPOCHS = 10
LEARNING_RATE = 1e-4
IMG_SIZE = 224
NUM_WORKERS = 4


class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, num_frames=10):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]

        cap = cv2.VideoCapture(path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count > 0:
            frame_indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
        else:
            frame_indices = np.zeros(self.num_frames, dtype=int)

        frames = []
        current_frame = 0
        grabbed_count = 0

        while cap.isOpened() and grabbed_count < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame == frame_indices[grabbed_count]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
                grabbed_count += 1

            current_frame += 1

        cap.release()

        while len(frames) < self.num_frames:
            empty_frame = torch.zeros((3, IMG_SIZE, IMG_SIZE))
            frames.append(empty_frame)

        video_tensor = torch.stack(frames)
        return video_tensor, torch.tensor(label, dtype=torch.long)


class VideoEfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super(VideoEfficientNet, self).__init__()

        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        B, F, C, H, W = x.shape

        x = x.view(B * F, C, H, W)

        out = self.backbone(x)

        out = out.view(B, F, -1)
        out = out.mean(dim=1)

        return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device} with {num_gpus} GPUs")

    normal_dir = os.path.join(DATA_DIR, "normal", "*.mp4")
    shoplifting_dir = os.path.join(DATA_DIR, "shoplifting", "*.mp4")

    normal_videos = glob.glob(normal_dir)
    shoplifting_videos = glob.glob(shoplifting_dir)

    if not normal_videos and not shoplifting_videos:
        print("Error: No videos found. Please check your dataset path.")
        return

    print(f"Found {len(normal_videos)} normal videos and {len(shoplifting_videos)} shoplifting videos.")

    all_videos = normal_videos + shoplifting_videos
    all_labels = [0] * len(normal_videos) + [1] * len(shoplifting_videos)

    X_train, X_val, y_train, y_val = train_test_split(
        all_videos,
        all_labels,
        test_size=0.2,
        random_state=42,
        stratify=all_labels
    )

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = VideoDataset(
        X_train,
        y_train,
        transform=transform,
        num_frames=FRAMES_PER_VIDEO
    )

    val_dataset = VideoDataset(
        X_val,
        y_val,
        transform=transform,
        num_frames=FRAMES_PER_VIDEO
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = VideoEfficientNet(num_classes=2).to(device)

    if num_gpus > 1:
        print("Wrapping model in DataParallel to utilize both GPUs...")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")

        model.train()
        train_loss, train_correct = 0.0, 0

        loop = tqdm(train_loader, desc="Training")

        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

            loop.set_postfix(loss=loss.item())

        train_acc = train_correct.double() / len(train_dataset)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss, val_correct = 0.0, 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)

        val_acc = val_correct.double() / len(val_dataset)
        val_loss /= len(val_dataset)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), "best_shoplifting_efficientnet.pth")
            print(">>> Saved new best model!")


if __name__ == '__main__':
    main()
