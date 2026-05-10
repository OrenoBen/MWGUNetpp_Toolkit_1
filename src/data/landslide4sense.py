import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

class LandslideDataset(Dataset):
    """适配Landslide4Sense数据集的Dataset类，支持14波段读取、H5格式解析、数据增强"""
    def __init__(self, data_root, split="train", transform=None, target_transform=None):
        """
        Args:
            data_root (str): 数据集根目录（即项目中创建的data/文件夹路径）
            split (str): 数据集划分，可选"train"/"validation"/"test"
            transform: 图像（14波段）的预处理/增强变换
            target_transform: 掩码的预处理变换
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # 定义图像和掩码的路径
        self.images_dir = os.path.join(data_root, "images", split)
        self.masks_dir = os.path.join(data_root, "annotations", split)
        
        # 校验路径有效性
        if not os.path.exists(self.images_dir) or not os.path.exists(self.masks_dir):
            raise ValueError(f"{split}划分的图像/掩码路径不存在，请检查数据集目录结构")
        
        # 获取所有文件名称（按序号排序，确保图像与掩码一一对应）
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith(".h5")])
        self.mask_files = sorted([f for f in os.listdir(self.masks_dir) if f.endswith(".h5")])
        
        # 校验文件数量匹配
        if len(self.image_files) != len(self.mask_files):
            raise RuntimeError(f"{split}划分的图像数量（{len(self.image_files)}）与掩码数量（{len(self.mask_files)}）不匹配")
        
        # 预计算训练集的均值和方差（用于标准化，14波段分别计算）
        self.mean = np.array([423.1, 541.5, 629.8, 673.9, 832.4, 1648.2, 1874.2, 
                             1997.9, 2076.4, 2110.2, 2203.5, 2326.1, 1758.3, 956.2])  # 示例值，建议用真实训练集计算
        self.std = np.array([85.2, 98.7, 112.3, 135.6, 156.8, 234.5, 289.1, 
                            312.4, 328.7, 335.1, 356.2, 389.7, 301.2, 218.5])   # 示例值，建议用真实训练集计算

    def __len__(self):
        """返回数据集样本数量"""
        return len(self.image_files)

    def _load_h5(self, file_path):
        """读取H5文件，返回numpy数组。自动检测HDF5文件结构"""
        with h5py.File(file_path, "r") as f:
            # 尝试多种可能的数据键名
            possible_keys = ["array", "image", "data", "img", "dataset", "mask"]
            
            # 首先尝试用户在possible_keys列表中指定的键
            for key in possible_keys:
                if key in f:
                    return f[key][:]
            
            # 如果指定的键都不存在，尝试使用第一个可用的键
            keys = list(f.keys())
            if keys:
                default_key = keys[0]
                return f[default_key][:]
            
            # 如果文件中没有任何键，抛出错误
            raise ValueError(f"HDF5文件 {file_path} 中不包含任何可访问的数据键")
    def _standardize(self, image):
        """对14波段图像进行标准化（每个波段单独归一化）"""
        # 图像维度：(H, W, 14) → 转为(14, H, W)，适配PyTorch通道优先格式
        image = np.transpose(image, (2, 0, 1))
        # 逐波段标准化：(x - mean) / std
        for i in range(14):
            image[i] = (image[i] - self.mean[i]) / (self.std[i] + 1e-8)  # 避免除零
        return image

    def _random_augment(self, image, mask):
        """训练集随机数据增强（仅对训练集生效）"""
        if self.split != "train":
            return image, mask
        
        # 1. 随机水平翻转（50%概率）
        if random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        
        # 2. 随机垂直翻转（50%概率）
        if random.random() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
        
        # 3. 随机90度旋转（0/90/180/270度）
        rotate_k = random.choice([0, 1, 2, 3])
        if rotate_k > 0:
            image = np.rot90(image, k=rotate_k, axes=(0, 1))
            mask = np.rot90(mask, k=rotate_k, axes=(0, 1))
        
        return image, mask

    def __getitem__(self, idx):
        """读取单样本（图像+掩码），返回字典格式"""
        # 1. 读取图像和掩码文件
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        
        image = self._load_h5(image_path)  # 输出形状：(128, 128, 14)（H, W, 波段数）
        mask = self._load_h5(mask_path)    # 输出形状：(128, 128)（二值掩码，0=非滑坡，255=滑坡）
        
        # 2. 掩码预处理：转为0/1二值图
        mask = (mask > 127).astype(np.float32)  # 255→1，0→0
        
        # 3. 数据增强（仅训练集）
        image, mask = self._random_augment(image, mask)
        
        # 4. 标准化（14波段单独处理）
        image = self._standardize(image)
        
        # 5. 转为Tensor
        image = torch.from_numpy(image.copy()).float()  # 添加copy()解决负步长问题
        mask = torch.from_numpy(mask.copy()).unsqueeze(0).float()  # 添加copy()解决负步长问题
        
        # 6. 应用外部变换（如额外的裁剪、归一化等）
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        # 返回字典格式，与landslide4sense.py保持接口一致
        return {
            "image": image,
            "mask": mask,
            "path": image_path
        }


# ------------------------------
# 辅助函数：创建数据加载器（直接调用即可）
# ------------------------------
def create_dataloader(data_root, split="train", batch_size=32, shuffle=True, num_workers=4):
    """
    创建Landslide4Sense数据集的数据加载器
    Args:
        data_root (str): 数据集根目录（如"../data"）
        split (str): 数据集划分
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据
        num_workers (int): 多进程数量
    Returns:
        DataLoader: PyTorch数据加载器
    """
    dataset = LandslideDataset(
        data_root=data_root,
        split=split,
        transform=None,  # 若需额外变换，可在此添加（如transforms.Resize）
        target_transform=None
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train")  # 训练集丢弃最后不完整批次
    )
    
    return dataloader


# ------------------------------
# 测试代码（验证数据集是否能正常加载）
# ------------------------------
if __name__ == "__main__":
    # 测试路径（需根据实际项目结构调整，此处为项目根目录下的data文件夹）
    DATA_ROOT = "C:\\Users\\hongbo_0\\Desktop\\U_net_new\\Landslide4Sense-2022-main\\landslide4sense"  # 从src/data/目录指向项目根目录的data/
    
    # 创建训练集加载器
    train_loader = create_dataloader(
        data_root=DATA_ROOT,
        split="train",
        batch_size=8,
        shuffle=True
    )
    
    # 验证数据维度
    for batch in train_loader:
        images = batch["image"]
        masks = batch["mask"]
        print(f"图像批次形状: {images.shape} → (batch, 14波段, H, W)")
        print(f"掩码批次形状: {masks.shape} → (batch, 1通道, H, W)")
        print(f"图像数值范围: {images.min():.2f} ~ {images.max():.2f}")
        print(f"掩码数值范围: {masks.min():.0f} ~ {masks.max():.0f}")
        break