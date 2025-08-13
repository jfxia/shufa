# 本程序为train_model.py的分布式训练版本
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
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import socket

# SE注意力模块
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 双任务模型 (DMTL框架)
class DMTLModel(nn.Module):
    def __init__(self, num_chars, num_styles):
        super(DMTLModel, self).__init__()
        weights = ResNet50_Weights.DEFAULT
        self.base_model = resnet50(weights=weights)
        
        # 解冻ResNet50后半部分参数
        for name, param in self.base_model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # 在stage4后加入SE模块
        self.se = SEBlock(2048)
        
        # 获取特征维度
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        
        # 双任务头
        self.char_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_chars)
        )
        self.style_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_styles)
        )
        
        # 任务不确定性参数
        self.log_vars = nn.Parameter(torch.zeros(2))
        self.dynamic_weighted_loss = DynamicWeightedLoss()  # 初始化损失权重模块

    def forward(self, x):
        features = self.base_model(x)
        features = self.se(features.unsqueeze(-1).unsqueeze(-1))
        features = features.squeeze(-1).squeeze(-1)
        char_output = self.char_head(features)
        style_output = self.style_head(features)
        return char_output, style_output

# 动态加权损失
class DynamicWeightedLoss(nn.Module):
    def __init__(self):
        super(DynamicWeightedLoss, self).__init__()

    def forward(self, losses, log_vars):
        loss1, loss2 = losses
        precision1 = torch.exp(-log_vars[0])
        precision2 = torch.exp(-log_vars[1])
        return precision1 * loss1 + precision2 * loss2 + log_vars.sum()

# 梯度手术 (缓解任务冲突)
def gradient_surgery(parameters, char_loss, style_loss):
    char_grads = torch.autograd.grad(char_loss, parameters, retain_graph=True, create_graph=True)
    style_grads = torch.autograd.grad(style_loss, parameters, retain_graph=True, create_graph=True)
    
    new_grads = []
    for char_g, style_g in zip(char_grads, style_grads):
        if char_g is None or style_g is None:
            new_grads.append(char_g if char_g is not None else style_g)
        else:
            cos_sim = F.cosine_similarity(char_g.flatten(), style_g.flatten(), dim=0)
            if cos_sim < 0:
                # 投影风格梯度
                proj = (torch.sum(style_g * char_g) / (torch.norm(char_g)**2 + 1e-8)) * char_g
                new_style_g = style_g - proj
                new_grads.append(char_g + new_style_g)
            else:
                new_grads.append(char_g + style_g)
    
    # 手动设置梯度
    for param, grad in zip(parameters, new_grads):
        if grad is not None:
            param.grad = grad

# 数据集类
class CalligraphyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.char_labels = []
        self.style_labels = []  # 风格标签
        self.char_to_idx = {}
        self.style_to_idx = {}  # 风格映射
        
        supported_exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
        
        # 遍历数据集
        char_idx = 0
        style_idx = 0
        for style_name in sorted(os.listdir(data_dir)):  # 风格作为一级目录
            style_dir = os.path.join(data_dir, style_name)
            if os.path.isdir(style_dir):
                # 添加风格映射
                if style_name not in self.style_to_idx:
                    self.style_to_idx[style_name] = style_idx
                    style_idx += 1
                
                for char_name in sorted(os.listdir(style_dir)):
                    char_dir = os.path.join(style_dir, char_name)
                    if os.path.isdir(char_dir):
                        # 添加字符映射
                        if char_name not in self.char_to_idx:
                            self.char_to_idx[char_name] = char_idx
                            char_idx += 1
                        
                        # 遍历图片
                        for file in os.listdir(char_dir):
                            if file.lower().endswith(supported_exts):
                                img_path = os.path.join(char_dir, file)
                                self.image_paths.append(img_path)
                                self.char_labels.append(self.char_to_idx[char_name])
                                self.style_labels.append(self.style_to_idx[style_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Skipping corrupted image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        char_label = self.char_labels[idx]
        style_label = self.style_labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, char_label, style_label

def train_one_epoch(model, dataloader, char_criterion, style_criterion, optimizer, device, epoch, rank):
    model.train()
    running_char_loss = 0.0
    running_style_loss = 0.0
    char_correct = 0
    style_correct = 0
    total_samples = 0
    
    for i, (inputs, char_labels, style_labels) in enumerate(dataloader):
        inputs, char_labels, style_labels = inputs.to(device), char_labels.to(device), style_labels.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        char_outputs, style_outputs = model(inputs)
        
        # 计算损失
        char_loss = char_criterion(char_outputs, char_labels)
        style_loss = style_criterion(style_outputs, style_labels)
        
        # 动态加权损失（使用模型的log_vars参数）
        weighted_loss = model.module.dynamic_weighted_loss([char_loss, style_loss], model.module.log_vars)
        
        # 梯度手术 - 使用原始模型参数
        shared_params = [param for name, param in model.module.named_parameters() 
                         if 'base_model' in name and param.requires_grad]
        gradient_surgery(shared_params, char_loss, style_loss)
        
        # 反向传播
        weighted_loss.backward()
        optimizer.step()
        
        # 更新统计信息
        running_char_loss += char_loss.item() * inputs.size(0)
        running_style_loss += style_loss.item() * inputs.size(0)
        _, char_preds = torch.max(char_outputs, 1)
        _, style_preds = torch.max(style_outputs, 1)
        char_correct += torch.sum(char_preds == char_labels.data)
        style_correct += torch.sum(style_preds == style_labels.data)
        total_samples += inputs.size(0)
        
        # 仅主进程打印批次日志
        if rank == 0 and i % 50 == 49:
            print(f"Epoch {epoch} Batch {i+1}: Char Loss: {char_loss.item():.4f}, "
                  f"Style Loss: {style_loss.item():.4f}, "
                  f"Char Acc: {char_correct.double() / total_samples:.4f}, "
                  f"Style Acc: {style_correct.double() / total_samples:.4f}")
    
    # 准备聚合的张量
    total_samples_tensor = torch.tensor(total_samples, device=device, dtype=torch.float64)
    running_char_loss_tensor = torch.tensor(running_char_loss, device=device, dtype=torch.float64)
    running_style_loss_tensor = torch.tensor(running_style_loss, device=device, dtype=torch.float64)
    char_correct_tensor = char_correct.double().to(device)
    style_correct_tensor = style_correct.double().to(device)
    
    # 聚合所有进程的数据
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_char_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_style_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(char_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(style_correct_tensor, op=dist.ReduceOp.SUM)
    
    epoch_char_loss = running_char_loss_tensor.item() / total_samples_tensor.item()
    epoch_style_loss = running_style_loss_tensor.item() / total_samples_tensor.item()
    epoch_char_acc = char_correct_tensor.item() / total_samples_tensor.item()
    epoch_style_acc = style_correct_tensor.item() / total_samples_tensor.item()
    
    return epoch_char_loss, epoch_style_loss, epoch_char_acc, epoch_style_acc

def evaluate(model, dataloader, char_criterion, style_criterion, device):
    model.eval()
    running_char_loss = 0.0
    running_style_loss = 0.0
    char_correct = 0
    style_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, char_labels, style_labels in dataloader:
            inputs, char_labels, style_labels = inputs.to(device), char_labels.to(device), style_labels.to(device)
            
            char_outputs, style_outputs = model(inputs)
            
            char_loss = char_criterion(char_outputs, char_labels)
            style_loss = style_criterion(style_outputs, style_labels)
            
            running_char_loss += char_loss.item() * inputs.size(0)
            running_style_loss += style_loss.item() * inputs.size(0)
            _, char_preds = torch.max(char_outputs, 1)
            _, style_preds = torch.max(style_outputs, 1)
            char_correct += torch.sum(char_preds == char_labels.data)
            style_correct += torch.sum(style_preds == style_labels.data)
            total_samples += inputs.size(0)
    
    # 准备聚合的张量
    total_samples_tensor = torch.tensor(total_samples, device=device, dtype=torch.float64)
    running_char_loss_tensor = torch.tensor(running_char_loss, device=device, dtype=torch.float64)
    running_style_loss_tensor = torch.tensor(running_style_loss, device=device, dtype=torch.float64)
    char_correct_tensor = char_correct.double().to(device)
    style_correct_tensor = style_correct.double().to(device)
    
    # 聚合所有进程的数据
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_char_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_style_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(char_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(style_correct_tensor, op=dist.ReduceOp.SUM)
    
    epoch_char_loss = running_char_loss_tensor.item() / total_samples_tensor.item()
    epoch_style_loss = running_style_loss_tensor.item() / total_samples_tensor.item()
    epoch_char_acc = char_correct_tensor.item() / total_samples_tensor.item()
    epoch_style_acc = style_correct_tensor.item() / total_samples_tensor.item()
    
    return epoch_char_loss, epoch_style_loss, epoch_char_acc, epoch_style_acc

def init_distributed(rank, world_size, args):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',  # GPU使用nccl通信
        rank=rank,
        world_size=world_size
    )
    
    # 设置当前设备
    torch.cuda.set_device(rank % torch.cuda.device_count())

def train(rank, world_size, args):
    # 初始化分布式环境
    init_distributed(rank, world_size, args)
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    
    # 数据增强
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.7, 1.3)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # 加载数据集
    full_dataset = CalligraphyDataset(args.data_dir, transform=data_transforms['train'])
    num_chars = len(full_dataset.char_to_idx)
    num_styles = len(full_dataset.style_to_idx)
    
    # 仅在主进程保存映射表
    if rank == 0:
        print(f"Found {len(full_dataset)} images, {num_chars} characters, {num_styles} styles")
        with open('char_map.json', 'w', encoding='utf-8') as f:
            json.dump(full_dataset.char_to_idx, f, ensure_ascii=False, indent=4)
        with open('style_map.json', 'w', encoding='utf-8') as f:
            json.dump(full_dataset.style_to_idx, f, ensure_ascii=False, indent=4)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = data_transforms['val']
    
    # 使用DistributedSampler进行数据分片
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型并包装为DDP
    model = DMTLModel(num_chars, num_styles).to(device)
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    
    # 损失函数
    char_criterion = nn.CrossEntropyLoss()
    style_criterion = nn.CrossEntropyLoss()
    
    # 学习率根据总GPU数量缩放
    scaled_lr = args.lr * world_size
    optimizer = optim.AdamW([
        {'params': model.module.base_model.parameters(), 'lr': scaled_lr * 0.1},
        {'params': model.module.char_head.parameters()},
        {'params': model.module.style_head.parameters()},
        {'params': model.module.log_vars}
    ], lr=scaled_lr, weight_decay=1e-5)
    
    best_char_acc = 0.0
    
    # 训练循环
    for epoch in range(args.epochs):
        # 设置sampler的epoch，确保每个epoch数据打乱方式一致
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print('-' * 10)
        
        # 训练一个epoch
        train_char_loss, train_style_loss, train_char_acc, train_style_acc = train_one_epoch(
            model, train_loader, char_criterion, style_criterion, optimizer, device, epoch+1, rank
        )
        
        # 验证
        val_char_loss, val_style_loss, val_char_acc, val_style_acc = evaluate(
            model, val_loader, char_criterion, style_criterion, device
        )
        
        # 仅主节点打印日志和保存模型
        if rank == 0:
            print(f"Train Char: Loss {train_char_loss:.4f} Acc {train_char_acc:.4f}")
            print(f"Train Style: Loss {train_style_loss:.4f} Acc {train_style_acc:.4f}")
            print(f"Val Char: Loss {val_char_loss:.4f} Acc {val_char_acc:.4f}")
            print(f"Val Style: Loss {val_style_loss:.4f} Acc {val_style_acc:.4f}")
            
            if val_char_acc > best_char_acc:
                best_char_acc = val_char_acc
                torch.save(model.module.state_dict(), 'best_model.pth')
                print("Saved best model")
    
    # 清理分布式环境
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='Distributed DMTL calligraphy recognition model')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to chinese_fonts dataset (must be accessible by all nodes)')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.0005, help='Base learning rate (will be scaled by number of GPUs)')
    parser.add_argument('--world-size', type=int, default=16, help='Total number of GPUs (4 servers × 4 GPUs)')
    parser.add_argument('--rank', type=int, default=0, help='Rank of current process (auto-set by launcher)')
    parser.add_argument('--master-addr', type=str, default='127.0.0.1', help='Master node IP address')
    parser.add_argument('--master-port', type=str, default='23456', help='Master node port number')
    args = parser.parse_args()
    
    # 启动分布式训练
    if args.rank == 0:
        print(f"Starting distributed training with {args.world_size} GPUs")
    mp.spawn(train, args=(args.world_size, args), nprocs=args.world_size, join=True)

if __name__ == '__main__':
    # 确保CUDA可见性（如果需要指定GPU）
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 每台服务器上的GPU编号
    main()

'''
##分布式训练启动说明

假设有4台服务器，每台服务器有4块GPU，IP地址分别为：

```
主节点 ：192.168.1.101  端口23456
从节点1：192.168.1.102
从节点2：192.168.1.103
从节点3：192.168.1.104
```

各服务器运行如下命令：

**主节点**
```
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=0 --master_addr="192.168.1.101" --master_port=23456 train_model.py --data-dir=/path/to/chinese_fonts_dataset --epochs=50 --batch-size=32
```

**从节点1**
```
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=1 --master_addr="192.168.1.101" --master_port=23456 train_model.py --data-dir=/path/to/chinese_fonts_dataset --epochs=50 --batch-size=32
```

**从节点2**
```
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=2 --master_addr="192.168.1.101" --master_port=23456 train_model.py --data-dir=/path/to/chinese_fonts_dataset --epochs=50 --batch-size=32
```

**从节点3**
```
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=3 --master_addr="192.168.1.101" --master_port=23456 train_model.py --data-dir=/path/to/chinese_fonts_dataset --epochs=50 --batch-size=32
```

**注意事项**

(1)确保所有服务器之间网络通畅，主节点的23456端口可访问

(2)数据集需要在所有服务器上**保持相同路径**（可通过NFS、SMB等共享存储实现）

(3)首次运行时建议先测试小批量数据和少量epoch，确认分布式环境正常工作

(4)可根据服务器性能调整num_workers参数（建议设置为 CPU 核心数的 1-2 倍）

(5)训练过程中只有主节点会打印日志和保存模型，其他节点仅进行计算

'''


