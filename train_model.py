import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms
from PIL import Image
import argparse

class CalligraphyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.char_to_idx = {}
        self.idx_to_char = {}

        supported_exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
        char_idx = 0
        for char_name in sorted(os.listdir(data_dir)):
            char_dir = os.path.join(data_dir, char_name)
            if os.path.isdir(char_dir):
                if char_name not in self.char_to_idx:
                    self.char_to_idx[char_name] = char_idx
                    self.idx_to_char[char_idx] = char_name
                    char_idx += 1

                for root, _, files in os.walk(char_dir):
                    for file in files:
                        if file.lower().endswith(supported_exts):
                            self.image_paths.append(os.path.join(root, file))
                            self.labels.append(self.char_to_idx[char_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Skipping corrupted image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_model(num_classes):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    # 解冻后半部分参数
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

        if i % 50 == 49:
            print(f"  Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description='Train a calligraphy recognition model.')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the chinese_fonts directory.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate.')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Training on CPU will be very slow.")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.7, 1.3)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    full_dataset = CalligraphyDataset(args.data_dir, transform=data_transforms['train'])
    num_classes = len(full_dataset.char_to_idx)
    print(f"Found {len(full_dataset)} images belonging to {num_classes} classes.")

    with open('char_map.json', 'w', encoding='utf-8') as f:
        json.dump(full_dataset.char_to_idx, f, ensure_ascii=False, indent=4)
    print("Character map saved to char_map.json")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset = CalligraphyDataset(args.data_dir, transform=data_transforms['val'])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    }

    model = get_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print('-' * 10)
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        val_loss, val_acc = evaluate(model, dataloaders['val'], criterion, device)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved to best_model.pth")

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    main()
