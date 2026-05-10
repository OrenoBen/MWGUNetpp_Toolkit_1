import os
import sys
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_classification import ClassificationLandslideDataset

def count_positive_negative():
    # 打印数据集根目录
    # 直接指定数据集的绝对路径，使用原始字符串避免转义错误
    root_dir = r"C:\Users\hongbo_0\Desktop\U_net_new\Landslide4Sense-2022-main\landslide4sense"
    print(f"数据集根目录: {root_dir}")
    print("\n正在统计正样例和负样例数量...")
    print("--------------------------------------------------")
    
    # 修复：同时检查所有三个划分
    splits = ['train', 'validation', 'test']
    
    # 遍历所有划分
    for split in splits:
        try:
            print(f"\n处理 {split} 集...")
            
            # 创建数据集实例
            dataset = ClassificationLandslideDataset(
                root=root_dir,
                split=split
            )
            
            # 统计正样例和负样例数量
            total_samples = len(dataset)
            positive_samples = sum(dataset.labels)
            negative_samples = total_samples - positive_samples
            
            # 计算百分比
            positive_percentage = (positive_samples / total_samples) * 100 if total_samples > 0 else 0
            negative_percentage = (negative_samples / total_samples) * 100 if total_samples > 0 else 0
            
            # 打印结果
            print(f"{split} 集总样本数: {total_samples}")
            print(f"正样例数量 (含滑坡): {positive_samples} ({positive_percentage:.2f}%)")
            print(f"负样例数量 (无滑坡): {negative_samples} ({negative_percentage:.2f}%)")
            
        except Exception as e:
            print(f"处理 {split} 集时出错: {str(e)}")
            print("可能原因: 该划分下没有数据或路径错误")
            continue

if __name__ == "__main__":
    count_positive_negative()