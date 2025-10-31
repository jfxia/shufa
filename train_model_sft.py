'''
模型微调程序

# 轻度微调（仅训练layer4和fc层）
python train_model_sft.py --data-dir ./data --epochs 30 --lr 0.0001 --fine-tune-level 1

# 深度微调（训练所有层）
python train_model_sft.py --data-dir ./data --epochs 50 --lr 0.00005 --fine-tune-level 4 --pretrained-model best_model.pth
'''
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
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

def get_model(num_classes, fine_tune_level=0):
    """
    获取模型并根据微调级别设置参数是否可训练
    fine_tune_level: 0-仅训练最后一层, 1-训练layer4和fc, 2-训练layer3-4和fc, 3-训练所有层
    """
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    
    # 根据微调级别设置参数是否可训练
    for name, param in model.named_parameters():
        param.requires_grad = False  # 默认冻结所有层
        
        if fine_tune_level >= 1 and ("layer4" in name or "fc" in name):
            param.requires_grad = True
        if fine_tune_level >= 2 and "layer3" in name:
            param.requires_grad = True
        if fine_tune_level >= 3 and ("layer1" in name or "layer2" in name):
            param.requires_grad = True
        if fine_tune_level >= 4:  # 训练包括conv1和bn1在内的所有层
            param.requires_grad = True

    # 替换最后一层
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # 微调时增大dropout防止过拟合
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        loss = checkpoint['loss']
        
        print(f"Resuming training from epoch {start_epoch + 1}, best accuracy: {best_acc:.4f}")
        return start_epoch, best_acc, loss
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting training from scratch.")
        return 0, 0.0, float('inf')

def train_one_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
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

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = correct_predictions.double() / total_samples if total_samples > 0 else 0.0
    
    # 学习率调度器步骤（基于训练损失）
    if scheduler and isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(epoch_loss)
        
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

def get_optimizer(model, lr, fine_tune_level):
    """根据微调级别设置不同的学习率"""
    # 分层设置学习率：深层网络参数使用较小学习率，浅层（如果解冻）使用更小学习率
    if fine_tune_level >= 3:
        # 对不同层设置不同学习率
        params = [
            {'params': model.conv1.parameters(), 'lr': lr * 0.1},
            {'params': model.bn1.parameters(), 'lr': lr * 0.1},
            {'params': model.layer1.parameters(), 'lr': lr * 0.2},
            {'params': model.layer2.parameters(), 'lr': lr * 0.4},
            {'params': model.layer3.parameters(), 'lr': lr * 0.6},
            {'params': model.layer4.parameters(), 'lr': lr * 0.8},
            {'params': model.fc.parameters(), 'lr': lr}
        ]
    elif fine_tune_level >= 2:
        params = [
            {'params': model.layer3.parameters(), 'lr': lr * 0.5},
            {'params': model.layer4.parameters(), 'lr': lr * 0.8},
            {'params': model.fc.parameters(), 'lr': lr}
        ]
    elif fine_tune_level >= 1:
        params = [
            {'params': model.layer4.parameters(), 'lr': lr * 0.5},
            {'params': model.fc.parameters(), 'lr': lr}
        ]
    else:
        params = model.fc.parameters()
        
    return optim.Adam(params, lr=lr, weight_decay=1e-5)  # 增加权重衰减

def main():
    parser = argparse.ArgumentParser(description='Fine-tune a calligraphy recognition model.')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the chinese_fonts directory.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of fine-tuning epochs.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for fine-tuning.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (smaller for fine-tuning).')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint-freq', type=int, default=3, help='Save checkpoint every N epochs')
    parser.add_argument('--fine-tune-level', type=int, default=1, 
                      help='Fine-tuning level (0-4, higher means more layers unfrozen)')
    parser.add_argument('--pretrained-model', type=str, default='', 
                      help='Path to pretrained model weights for initialization')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Training on CPU will be very slow.")

    # 微调的数据增强可以更温和
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),  # 减小旋转角度
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # 减小变换幅度
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 减小颜色抖动
            transforms.RandomGrayscale(p=0.05),
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

    # 保存字符映射
    if not os.path.exists('char_map_finetune.json'):
        with open('char_map_finetune.json', 'w', encoding='utf-8') as f:
            json.dump(full_dataset.char_to_idx, f, ensure_ascii=False, indent=4)
        print("Character map saved to char_map_finetune.json")
    else:
        print("Character map already exists, skipping creation.")

    # 分割数据集，微调可以使用更小的训练集比例
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset = CalligraphyDataset(args.data_dir, transform=data_transforms['val'])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    }

    # 初始化模型
    model = get_model(num_classes, args.fine_tune_level).to(device)
    
    # 如果提供了预训练模型，加载它
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"Loading pretrained model from {args.pretrained_model}")
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device), strict=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args.lr, args.fine_tune_level)
    
    # 学习率调度器：当损失不再下降时降低学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-7)

    # 断点续训
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        start_epoch, best_acc, _ = load_checkpoint(args.resume, model, optimizer)
    
    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}/{total_params} ({trainable_params/total_params:.1%})")

    # 微调循环
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print('-' * 10)
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")
        
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, device, scheduler)
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        
        val_loss, val_acc = evaluate(model, dataloaders['val'], criterion, device)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_finetuned_model.pth')
            print("Best fine-tuned model saved to best_finetuned_model.pth")
        
        # 定期保存断点
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = f'finetune_checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'loss': val_loss,
                'fine_tune_level': args.fine_tune_level
            }, checkpoint_path)
            
        # 保存最新断点
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'loss': val_loss,
            'fine_tune_level': args.fine_tune_level
        }, 'latest_finetune_checkpoint.pth')

    print(f"\nFine-tuning complete. Best validation accuracy: {best_acc:.4f}")

if __name__ == '__main__':

    main()
