import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.demo import LandslideDataset


def count_dataset_sizes(data_root):
    """统计各个数据集划分的大小"""
    splits = ["train", "validation", "test"]
    sizes = {}
    
    for split in splits:
        try:
            dataset = LandslideDataset(data_root, split=split)
            sizes[split] = len(dataset)
            print(f"{split}集大小: {len(dataset)} 样本")
        except (ValueError, FileNotFoundError) as e:
            sizes[split] = 0
            print(f"{split}集: 未找到或无法访问 - {str(e)}")
    
    return sizes


if __name__ == "__main__":
    # 使用与demo.py相同的数据路径
    DATA_ROOT = "C:\\Users\\hongbo_0\\Desktop\\U_net_new\\Landslide4Sense-2022-main\\landslide4sense"
    
    print("正在统计数据集大小...")
    print(f"数据集根目录: {DATA_ROOT}")
    print("=" * 50)
    
    sizes = count_dataset_sizes(DATA_ROOT)
    
    print("=" * 50)
    print("数据集统计总结:")
    for split, size in sizes.items():
        print(f"{split}: {size} 样本")
    print(f"总样本数: {sum(sizes.values())}")