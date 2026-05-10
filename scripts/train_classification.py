import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import random

# 导入现有的数据集类
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.landslide4sense import LandslideDataset, create_dataloader

# 设置随机种子
def set_seed(seed: int = 42):
    """设置所有随机数生成器的种子以确保可重复性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# 辅助函数：检查目录结构
# 使用demo.py中的目录结构检查功能

# 分类任务的数据集适配器
class ClassificationLandslideDataset(LandslideDataset):
    """为分类任务适配的数据集类，继承自现有的LandslideDataset"""
    def __init__(self, root, split="train", augment=False, resize=(256, 256), in_channels=3):
        # LandslideDataset内部已有路径验证逻辑，无需额外检查
        # 根据demo.py中的LandslideDataset参数调整
        super().__init__(data_root=root, split=split, transform=None, target_transform=None)
        # 预处理标签为分类任务
        self.labels = []
        # 修复：使用父类的image_files和mask_files属性，而不是不存在的self.pairs
        for image_path, mask_path in zip(self.image_files, self.mask_files):
            # 计算掩码中是否包含滑坡（非零像素）作为标签
            # 修复：使用父类的_load_h5方法读取掩码，并且需要构建完整路径
            full_mask_path = os.path.join(self.masks_dir, mask_path)
            mask = super()._load_h5(full_mask_path)
            # 修复：根据父类的逻辑，掩码值>127的为滑坡区域
            has_landslide = 1 if np.sum(mask > 127) > 0 else 0
            self.labels.append(has_landslide)
        
        # 根据split参数筛选数据
        if split in ['train', 'validation', 'test']:
            # 使用固定比例划分数据集
            train_indices, test_indices = train_test_split(
                range(len(self)), test_size=0.2, random_state=42,
                stratify=self.labels
            )
            
            if split == 'train':
                self.indices = train_indices
            elif split == 'validation':
                # 从测试集中再分割出验证集
                val_indices, test_indices = train_test_split(
                    test_indices, test_size=0.5, random_state=42,
                    stratify=[self.labels[i] for i in test_indices]
                )
                self.indices = val_indices
            else:  # test
                self.indices = test_indices
        else:
            self.indices = list(range(len(self)))
            
        # 调整数据集大小为筛选后的大小
        # 修复：更新image_files和mask_files，而不是不存在的self.pairs
        self.image_files = [self.image_files[i] for i in self.indices]
        self.mask_files = [self.mask_files[i] for i in self.indices]
        self.labels = [self.labels[i] for i in self.indices]
    
    def __getitem__(self, idx):
        # 获取原始数据
        data = super().__getitem__(idx)
        # 返回图像和对应的分类标签
        return data['image'], torch.tensor(self.labels[idx], dtype=torch.long)
    
    def __len__(self):
        # 修复：返回image_files或labels的长度，而不是不存在的self.pairs
        return len(self.image_files)

# 数据加载和准备
def get_datasets_and_loaders(root_dir=None, batch_size=16, num_workers=4):
    """获取数据集和数据加载器"""
    # 设置数据根目录
    if root_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 假设数据在项目根目录下的landslide4sense文件夹中
        root_dir = os.path.join(os.path.dirname(current_dir), 'landslide4sense')
    
    # 创建训练集、验证集和测试集
    train_dataset = ClassificationLandslideDataset(
        root=root_dir,
        split='train',
        augment=True,  # 训练集应用数据增强
        resize=(256, 256)
    )
    
    # 修复：使用'validation'而不是'val'作为划分名称
    val_dataset = ClassificationLandslideDataset(
        root=root_dir,
        split='validation',
        augment=False,
        resize=(256, 256)
    )
    
    test_dataset = ClassificationLandslideDataset(
        root=root_dir,
        split='test',
        augment=False,
        resize=(256, 256)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 打印数据集信息
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")
    print(f"数据集根目录: {root_dir}")
    
    return {
        'train': {'dataset': train_dataset, 'loader': train_loader},
        'val': {'dataset': val_dataset, 'loader': val_loader},
        'test': {'dataset': test_dataset, 'loader': test_loader}
    }


# 由于原始的LandslideDataset期望images和masks文件夹，但我们的数据可能以其他方式组织
# 让我们修改策略，直接使用一个简单的分类数据集实现

# 使用src.data.demo中的LandslideDataset类

# 修改get_datasets_and_loaders函数以使用我们的简化数据集
def get_datasets_and_loaders(data_root=None, batch_size=32, num_workers=4):
    """
    获取数据集和数据加载器
    使用demo.py中提供的create_dataloader函数
    """
    # 设置数据根目录
    if data_root is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_root = os.path.join(os.path.dirname(current_dir), 'landslide4sense')
    
    print(f"使用数据集根目录: {data_root}")
    
    # 创建训练集加载器
    train_loader = create_dataloader(
        data_root=data_root,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    # 尝试创建验证集加载器，如果失败则使用测试集
    try:
        val_loader = create_dataloader(
            data_root=data_root,
            split="validation",
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    except (ValueError, FileNotFoundError):
        print("验证集不存在，使用测试集作为验证集")
        val_loader = create_dataloader(
            data_root=data_root,
            split="test",
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    
    # 尝试创建测试集加载器
    try:
        test_loader = create_dataloader(
            data_root=data_root,
            split="test",
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    except (ValueError, FileNotFoundError):
        print("测试集不存在，使用验证集作为测试集")
        test_loader = val_loader
    
    print(f"数据集加载完成: 训练集 {len(train_loader.dataset)} 样本")
    print(f"验证集 {len(val_loader.dataset)} 样本")
    print(f"测试集 {len(test_loader.dataset)} 样本")
    
    return {
        'train': {'dataset': train_loader.dataset, 'loader': train_loader},
        'val': {'dataset': val_loader.dataset, 'loader': val_loader},
        'test': {'dataset': test_loader.dataset, 'loader': test_loader}
    }

# 图像特征编码器
class ImageEncoder(nn.Module):
    """统一的图像编码器，适用于分类任务"""
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256]):
        super().__init__()
        layers = []
        prev_dim = in_channels
        
        # 构建卷积层序列
        for dim in hidden_dims:
            layers.extend([
                nn.Conv2d(prev_dim, dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            prev_dim = dim
        
        # 最后添加一个全局池化层
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # 展平特征

# 分类头部模块
class ClassificationHead(nn.Module):
    """分类任务的头部模块"""
    def __init__(self, in_features=256, num_classes=2, dropout_rate=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# 主分类模型
class LandslideClassifier(nn.Module):
    """简化的滑坡分类模型，基于单模态输入"""
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        # 图像编码器
        self.encoder = ImageEncoder(in_channels=in_channels)
        
        # 分类头
        self.classification_head = ClassificationHead(
            in_features=256,  # 与ImageEncoder输出维度匹配
            num_classes=num_classes
        )

    def forward(self, x):
        # 提取图像特征
        features = self.encoder(x)
        
        # 分类
        return self.classification_head(features)



# 训练和评估函数
def train_model(model, loader, criterion, optimizer, device):
    """训练模型一个epoch，适配字典格式数据"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in loader:
        # 从字典中提取图像和掩码
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # 从掩码生成二分类标签：如果掩码中有任何非零值，则为滑坡(1)，否则为非滑坡(0)
        labels = (masks.sum(dim=(1, 2, 3)) > 0).long().squeeze()
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # 确保输出和标签形状匹配
        if outputs.dim() > 1 and outputs.size(1) > 1:
            _, predicted = torch.max(outputs, 1)
        else:
            predicted = (outputs > 0.5).long().squeeze()
        
        # 确保标签形状正确
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        if predicted.dim() == 0:
            predicted = predicted.unsqueeze(0)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    """评估模型性能，适配字典格式数据"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            # 从字典中提取图像和掩码
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # 从掩码生成二分类标签：如果掩码中有任何非零值，则为滑坡(1)，否则为非滑坡(0)
            labels = (masks.sum(dim=(1, 2, 3)) > 0).long().squeeze()
            
            outputs = model(images)
            
            # 确保输出和标签形状匹配
            if outputs.dim() > 1 and outputs.size(1) > 1:
                _, predicted = torch.max(outputs, 1)
            else:
                predicted = (outputs > 0.5).long().squeeze()
            
            # 确保标签形状正确
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            if predicted.dim() == 0:
                predicted = predicted.unsqueeze(0)
            
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), correct / total

# 训练配置和过程
def train_classification_model(model, train_loader, val_loader, test_loader, 
                              criterion, optimizer, scheduler, device, 
                              num_epochs=50, save_dir='models'):
    """训练分类模型的完整流程"""
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 初始化记录
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # 训练模型
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        
        # 在验证集上评估
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # 调整学习率
        scheduler.step(val_acc)
        
        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 打印训练进度
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(save_dir, 'best_landslide_classifier.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Best model saved to {model_path}!')
    
    # 训练结束后在测试集上评估
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'\nFinal Test Performance:')
    print(f'Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f}')
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'final_landslide_classifier.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_loss': test_loss,
        'test_acc': test_acc
    }


# 绘制训练结果函数
def plot_results(train_losses, test_losses, train_accs, test_accs):
    """
    绘制训练和测试的损失曲线以及准确率曲线
    
    Args:
        train_losses: 训练损失列表
        test_losses: 测试损失列表
        train_accs: 训练准确率列表
        test_accs: 测试准确率列表
    """
    import matplotlib.pyplot as plt
    import os
    
    # 创建结果目录
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 设置中文字体（如果需要显示中文）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 绘制损失曲线
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='训练损失')
    ax1.plot(epochs, test_losses, 'r-', label='测试损失')
    ax1.set_title('训练和测试损失')
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('损失值')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制准确率曲线
    ax2.plot(epochs, train_accs, 'b-', label='训练准确率')
    ax2.plot(epochs, test_accs, 'r-', label='测试准确率')
    ax2.set_title('训练和测试准确率')
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('准确率')
    ax2.set_ylim(0, 1.05)  # 设置y轴范围，确保能够看到完整的准确率变化
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('results/training_results.png', dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()
    
    print(f"训练结果图表已保存至: results/training_results.png")

def main():
    """主函数入口，执行训练分类模型的完整流程"""
    # 获取数据集和加载器
    data_root = 'C:\\Users\\hongbo_0\\Desktop\\U_net_new\\Landslide4Sense-2022-main\\landslide4sense'
    data_loaders = get_datasets_and_loaders(data_root=data_root)
    train_loader = data_loaders['train']['loader']
    val_loader = data_loaders['val']['loader']
    test_loader = data_loaders['test']['loader']

    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"测试样本数: {len(test_loader.dataset)}")

    # 初始化模型
    num_classes = 2  # 二分类任务：有滑坡/无滑坡
    model = LandslideClassifier(in_channels=14, num_classes=num_classes)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )

    # 训练模型
    training_results = train_classification_model(
        model, train_loader, val_loader, test_loader,
        criterion, optimizer, scheduler, device,
        num_epochs=50
    )

    # 绘制训练结果
    plot_results(
        training_results['train_losses'], 
        training_results['val_losses'], 
        training_results['train_accs'], 
        training_results['val_accs']
    )

# 程序入口
if __name__ == '__main__':
    # Windows系统上多进程数据加载需要这行代码
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    # 调用主函数
    main()