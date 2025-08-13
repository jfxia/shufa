# 汉字书法字体识别模型DMTL分布式训练程序
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms
import cv2  # 替换PIL为OpenCV加速图片加载
import argparse
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import logging
from datetime import datetime
import random
from torch.optim.lr_scheduler import CosineAnnealingLR

# 设置日志格式
def setup_logging(rank, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(f"Rank_{rank}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(f'%(asctime)s [Rank {rank}] %(message)s')
    
    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件 handler (仅主进程)
    if rank == 0:
        file_handler = logging.FileHandler(
            os.path.join(output_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

# 固定随机种子确保可复现性
def set_seed(seed, rank):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭自动优化以确保可复现

# SE注意力模块 (支持1D特征)
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 1D池化更高效
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()  # 直接处理[B, C]形状特征
        y = self.avg_pool(x.unsqueeze(1)).squeeze(1)  # [B, C] -> [B, 1, C] -> 池化 -> [B, 1, 1] -> [B, 1]
        y = self.fc(y)  # [B, C]
        return x * y  # 元素级乘法

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
        
        # 获取特征维度
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # 移除原fc层
        
        # 加入SE模块 (处理[B, 2048]特征)
        self.se = SEBlock(2048)
        
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
        self.dynamic_weighted_loss = DynamicWeightedLoss()

    def forward(self, x):
        features = self.base_model(x)  # [B, 2048]
        features = self.se(features)  # 直接处理1D特征，无需维度转换
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

# 梯度手术
def gradient_surgery(parameters, char_loss, style_loss):
    char_grads = torch.autograd.grad(char_loss, parameters, retain_graph=True)
    style_grads = torch.autograd.grad(style_loss, parameters, retain_graph=True)
    
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

# 数据集
class CalligraphyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.char_labels = []
        self.style_labels = []
        self.char_to_idx = {}
        self.style_to_idx = {}
        
        supported_exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
        
        # 预过滤损坏图片路径
        self._load_and_validate_data(supported_exts)
        
        # 确保数据集不为空
        if not self.image_paths:
            raise ValueError(f"No valid images found in {data_dir}")

    def _load_and_validate_data(self, supported_exts):
        char_idx = 0
        style_idx = 0
        for style_name in sorted(os.listdir(self.data_dir)):
            style_dir = os.path.join(self.data_dir, style_name)
            if not os.path.isdir(style_dir):
                continue
                
            if style_name not in self.style_to_idx:
                self.style_to_idx[style_name] = style_idx
                style_idx += 1
            
            for char_name in sorted(os.listdir(style_dir)):
                char_dir = os.path.join(style_dir, char_name)
                if not os.path.isdir(char_dir):
                    continue
                    
                if char_name not in self.char_to_idx:
                    self.char_to_idx[char_name] = char_idx
                    char_idx += 1
                
                for file in os.listdir(char_dir):
                    if file.lower().endswith(supported_exts):
                        img_path = os.path.join(char_dir, file)
                        # 预验证图片完整性
                        if self._is_image_valid(img_path):
                            self.image_paths.append(img_path)
                            self.char_labels.append(self.char_to_idx[char_name])
                            self.style_labels.append(self.style_to_idx[style_name])

    def _is_image_valid(self, img_path):
        try:
            # 使用OpenCV快速验证图片
            img = cv2.imread(img_path)
            return img is not None
        except:
            return False
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 使用OpenCV加载图片并转换为RGB
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV默认BGR，转换为RGB
        image = Image.fromarray(img)  # 转为PIL格式以便应用transforms
        
        char_label = self.char_labels[idx]
        style_label = self.style_labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, char_label, style_label

def train_one_epoch(model, dataloader, char_criterion, style_criterion, optimizer, 
                   device, epoch, rank, logger, scaler=None):
    model.train()
    running_stats = torch.zeros(5, device=device, dtype=torch.float64)  # [char_loss, style_loss, char_correct, style_correct, total]
    
    for i, (inputs, char_labels, style_labels) in enumerate(dataloader):
        inputs, char_labels, style_labels = inputs.to(device), char_labels.to(device), style_labels.to(device)
        optimizer.zero_grad()
        
        # 混合精度训练上下文
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            char_outputs, style_outputs = model(inputs)
            char_loss = char_criterion(char_outputs, char_labels)
            style_loss = style_criterion(style_outputs, style_labels)
            weighted_loss = model.module.dynamic_weighted_loss([char_loss, style_loss], model.module.log_vars)
        
        # 梯度手术
        shared_params = [param for name, param in model.module.named_parameters() 
                         if 'base_model' in name and param.requires_grad]
        gradient_surgery(shared_params, char_loss, style_loss)
        
        # 反向传播 (支持混合精度)
        if scaler:
            scaler.scale(weighted_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            weighted_loss.backward()
            optimizer.step()
        
        # 更新统计信息
        batch_size = inputs.size(0)
        _, char_preds = torch.max(char_outputs, 1)
        _, style_preds = torch.max(style_outputs, 1)
        
        running_stats[0] += char_loss.item() * batch_size
        running_stats[1] += style_loss.item() * batch_size
        running_stats[2] += torch.sum(char_preds == char_labels.data).double()
        running_stats[3] += torch.sum(style_preds == style_labels.data).double()
        running_stats[4] += batch_size
        
        # 打印批次日志
        if rank == 0 and i % 50 == 49:
            batch_char_acc = running_stats[2] / running_stats[4]
            batch_style_acc = running_stats[3] / running_stats[4]
            logger.info(f"Epoch {epoch} Batch {i+1}: "
                        f"Char Loss: {char_loss.item():.4f}, Style Loss: {style_loss.item():.4f}, "
                        f"Char Acc: {batch_char_acc:.4f}, Style Acc: {batch_style_acc:.4f}")
    
    # 合并所有进程的统计信息 (单步all_reduce优化)
    dist.all_reduce(running_stats, op=dist.ReduceOp.SUM)
    total = running_stats[4].item()
    return (
        running_stats[0].item() / total,
        running_stats[1].item() / total,
        running_stats[2].item() / total,
        running_stats[3].item() / total
    )

def evaluate(model, dataloader, char_criterion, style_criterion, device):
    model.eval()
    running_stats = torch.zeros(5, device=device, dtype=torch.float64)  # [char_loss, style_loss, char_correct, style_correct, total]
    
    with torch.inference_mode():  # 更高效的推理模式
        for inputs, char_labels, style_labels in dataloader:
            inputs, char_labels, style_labels = inputs.to(device), char_labels.to(device), style_labels.to(device)
            char_outputs, style_outputs = model(inputs)
            
            char_loss = char_criterion(char_outputs, char_labels)
            style_loss = style_criterion(style_outputs, style_labels)
            
            batch_size = inputs.size(0)
            _, char_preds = torch.max(char_outputs, 1)
            _, style_preds = torch.max(style_outputs, 1)
            
            running_stats[0] += char_loss.item() * batch_size
            running_stats[1] += style_loss.item() * batch_size
            running_stats[2] += torch.sum(char_preds == char_labels.data).double()
            running_stats[3] += torch.sum(style_preds == style_labels.data).double()
            running_stats[4] += batch_size
    
    # 合并所有进程的统计信息
    dist.all_reduce(running_stats, op=dist.ReduceOp.SUM)
    total = running_stats[4].item()
    return (
        running_stats[0].item() / total,
        running_stats[1].item() / total,
        running_stats[2].item() / total,
        running_stats[3].item() / total
    )

def init_distributed(rank, world_size, args):
    """初始化分布式环境 (从环境变量获取参数)"""
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())

def save_checkpoint(model, optimizer, scheduler, epoch, best_char_acc, output_dir, logger):
    """保存训练检查点"""
    checkpoint = {
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'best_char_acc': best_char_acc
    }
    # 保存最新检查点
    latest_path = os.path.join(output_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    logger.info(f"Saved latest checkpoint to {latest_path}")
    
    # 保存最佳检查点
    best_path = os.path.join(output_dir, 'best_model.pth')
    torch.save(model.module.state_dict(), best_path)
    logger.info(f"Saved best model to {best_path}")

def load_checkpoint(model, optimizer, scheduler, resume_path, device, logger):
    """加载检查点恢复训练"""
    if not os.path.exists(resume_path):
        logger.error(f"Checkpoint {resume_path} not found!")
        return 0, 0.0
    
    checkpoint = torch.load(resume_path, map_location=device)
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
    best_char_acc = checkpoint['best_char_acc']
    logger.info(f"Resumed from checkpoint: epoch {checkpoint['epoch']}, best char acc {best_char_acc:.4f}")
    return start_epoch, best_char_acc

def train(rank, world_size, args):
    # 初始化分布式环境
    init_distributed(rank, world_size, args)
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    
    # 初始化日志
    logger = setup_logging(rank, args.output_dir)
    
    # 固定随机种子
    set_seed(args.seed, rank)
    
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
    try:
        full_dataset = CalligraphyDataset(args.data_dir, transform=data_transforms['train'])
    except ValueError as e:
        logger.error(e)
        dist.destroy_process_group()
        return
    
    num_chars = len(full_dataset.char_to_idx)
    num_styles = len(full_dataset.style_to_idx)
    
    # 仅主进程保存映射表
    if rank == 0:
        logger.info(f"Found {len(full_dataset)} images, {num_chars} characters, {num_styles} styles")
        with open(os.path.join(args.output_dir, 'char_map.json'), 'w', encoding='utf-8') as f:
            json.dump(full_dataset.char_to_idx, f, ensure_ascii=False, indent=4)
        with open(os.path.join(args.output_dir, 'style_map.json'), 'w', encoding='utf-8') as f:
            json.dump(full_dataset.style_to_idx, f, ensure_ascii=False, indent=4)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = data_transforms['val']
    
    # 数据加载器
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True  # 丢弃最后一个不完整批次
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 初始化模型
    model = DMTLModel(num_chars, num_styles).to(device)
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    
    # 损失函数和优化器
    char_criterion = nn.CrossEntropyLoss()
    style_criterion = nn.CrossEntropyLoss()
    
    scaled_lr = args.lr * world_size
    optimizer = optim.AdamW([
        {'params': model.module.base_model.parameters(), 'lr': scaled_lr * 0.1},
        {'params': model.module.char_head.parameters()},
        {'params': model.module.style_head.parameters()},
        {'params': model.module.log_vars}
    ], lr=scaled_lr, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    # 断点续训
    start_epoch = 0
    best_char_acc = 0.0
    if args.resume:
        start_epoch, best_char_acc = load_checkpoint(
            model, optimizer, scheduler, args.resume, device, logger
        )
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        if rank == 0:
            logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
            logger.info('-' * 40)
        
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, char_criterion, style_criterion, optimizer, 
            device, epoch+1, rank, logger, scaler
        )
        train_char_loss, train_style_loss, train_char_acc, train_style_acc = train_metrics
        
        # 验证
        val_metrics = evaluate(
            model, val_loader, char_criterion, style_criterion, device
        )
        val_char_loss, val_style_loss, val_char_acc, val_style_acc = val_metrics
        
        # 学习率调度
        scheduler.step()
        
        # 主进程日志和保存
        if rank == 0:
            logger.info(f"Train: Char Loss {train_char_loss:.4f} Acc {train_char_acc:.4f}; "
                        f"Style Loss {train_style_loss:.4f} Acc {train_style_acc:.4f}")
            logger.info(f"Val:   Char Loss {val_char_loss:.4f} Acc {val_char_acc:.4f}; "
                        f"Style Loss {val_style_loss:.4f} Acc {val_style_acc:.4f}")
            logger.info(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存检查点
            if val_char_acc > best_char_acc:
                best_char_acc = val_char_acc
                logger.info(f"New best char accuracy: {best_char_acc:.4f}")
            
            # 每5个epoch或最后一个epoch保存检查点
            if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
                save_checkpoint(model, optimizer, scheduler, epoch, best_char_acc, args.output_dir, logger)
    
    # 清理
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='Distributed DMTL Calligraphy Recognition')
    # 数据与训练参数
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset (shared across nodes)')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.0005, help='Base learning rate (scaled by world size)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers per process')
    
    # 分布式参数
    parser.add_argument('--world-size', type=int, required=True, help='Total number of GPUs across all nodes')
    parser.add_argument('--master-addr', type=str, default='127.0.0.1', help='Master node IP')
    parser.add_argument('--master-port', type=str, default='23456', help='Master node port')
    
    # 优化相关参数
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--mixed-precision', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save logs and checkpoints')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint for resume training')
    
    args = parser.parse_args()
    
    # 从环境变量获取rank (兼容torch.distributed.launch)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank == 0:
        print(f"Starting distributed training with {args.world_size} GPUs")
        print(f"Outputs will be saved to {args.output_dir}")
    
    # 启动分布式训练
    mp.spawn(
        train, 
        args=(args.world_size, args), 
        nprocs=args.world_size, 
        join=True
    )

if __name__ == '__main__':
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
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=0 \
  --master_addr="192.168.1.101" --master_port=23456 train_model_ddp.py \
  --data-dir=/path/to/dataset --world-size=16 --epochs=100 \
  --batch-size=32 --num-workers=8 --mixed-precision --output-dir=./training_results
```

**从节点1**
```
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=1 \
  --master_addr="192.168.1.101" --master_port=23456 train_model_ddp.py \
  --data-dir=/path/to/dataset --world-size=16 --epochs=100 \
  --batch-size=32 --num-workers=8 --mixed-precision --output-dir=./training_results
```

**从节点2**
```
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=2 \
  --master_addr="192.168.1.101" --master_port=23456 train_model_ddp.py \
  --data-dir=/path/to/dataset --world-size=16 --epochs=100 \
  --batch-size=32 --num-workers=8 --mixed-precision --output-dir=./training_results
```

**从节点3**
```
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=3 \
  --master_addr="192.168.1.101" --master_port=23456 train_model_ddp.py \
  --data-dir=/path/to/dataset --world-size=16 --epochs=100 \
  --batch-size=32 --num-workers=8 --mixed-precision --output-dir=./training_results
```

**注意事项**

(1)确保所有服务器之间网络通畅，主节点的23456端口可访问

(2)数据集需要在所有服务器上**保持相同路径**（可通过NFS、SMB等共享存储实现）

(3)首次运行时建议先测试小批量数据和少量epoch，确认分布式环境正常工作

(4)可根据服务器性能调整num_workers参数（建议设置为 CPU 核心数的 1-2 倍）

(5)训练过程中只有主节点会打印日志和保存模型，其他节点仅进行计算

(6)如需从断点恢复训练，添加--resume=./training_results/latest_checkpoint.pth参数即可。
'''
