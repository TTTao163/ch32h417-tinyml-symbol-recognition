import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import shutil

class DataQualityChecker:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.train_dir = os.path.join(base_dir, 'training')
        self.test_dir = os.path.join(base_dir, 'testing')
        self.quality_report = defaultdict(list)
        
    def check_image_quality(self, image_path):
        """检查单张图像的质量"""
        try:
            # 读取图像
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {'status': 'error', 'reason': 'cannot_read'}
            
            # 检查尺寸
            if img.shape != (28, 28):
                return {'status': 'error', 'reason': f'wrong_size_{img.shape}'}
            
            # 检查是否为空白图像
            if np.all(img == 0) or np.all(img == 255):
                return {'status': 'warning', 'reason': 'blank_image'}
            
            # 检查对比度
            contrast = img.std()
            if contrast < 10:
                return {'status': 'warning', 'reason': f'low_contrast_{contrast:.2f}'}
            
            # 检查亮度分布
            mean_brightness = img.mean()
            if mean_brightness < 20 or mean_brightness > 235:
                return {'status': 'warning', 'reason': f'extreme_brightness_{mean_brightness:.2f}'}
            
            # 检查是否有异常像素值
            unique_values = len(np.unique(img))
            if unique_values < 5:
                return {'status': 'warning', 'reason': f'few_unique_values_{unique_values}'}
            
            return {'status': 'ok', 'contrast': contrast, 'brightness': mean_brightness}
            
        except Exception as e:
            return {'status': 'error', 'reason': f'exception_{str(e)}'}
    
    def analyze_dataset(self, dataset_dir, dataset_name):
        """分析整个数据集的质量"""
        print(f"\n--- 分析 {dataset_name} 数据集 ---")
        
        class_stats = {}
        total_images = 0
        error_count = 0
        warning_count = 0
        
        for class_name in sorted(os.listdir(dataset_dir)):
            class_path = os.path.join(dataset_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            print(f"检查类别: {class_name}")
            
            class_images = os.listdir(class_path)
            class_stats[class_name] = {
                'total': len(class_images),
                'ok': 0,
                'warnings': 0,
                'errors': 0,
                'error_files': [],
                'warning_files': []
            }
            
            for img_file in class_images:
                img_path = os.path.join(class_path, img_file)
                result = self.check_image_quality(img_path)
                
                if result['status'] == 'ok':
                    class_stats[class_name]['ok'] += 1
                elif result['status'] == 'warning':
                    class_stats[class_name]['warnings'] += 1
                    class_stats[class_name]['warning_files'].append((img_file, result['reason']))
                    warning_count += 1
                else:  # error
                    class_stats[class_name]['errors'] += 1
                    class_stats[class_name]['error_files'].append((img_file, result['reason']))
                    error_count += 1
                
                total_images += 1
        
        # 打印统计结果
        print(f"\n{dataset_name} 数据集质量报告:")
        print(f"总图像数: {total_images}")
        print(f"正常图像: {total_images - error_count - warning_count}")
        print(f"警告图像: {warning_count}")
        print(f"错误图像: {error_count}")
        
        for class_name, stats in class_stats.items():
            print(f"\n类别 {class_name}:")
            print(f"  总数: {stats['total']}")
            print(f"  正常: {stats['ok']}")
            print(f"  警告: {stats['warnings']}")
            print(f"  错误: {stats['errors']}")
            
            if stats['error_files']:
                print(f"  错误文件: {stats['error_files'][:5]}...")  # 只显示前5个
            if stats['warning_files']:
                print(f"  警告文件: {stats['warning_files'][:3]}...")  # 只显示前3个
        
        return class_stats
    
    def clean_dataset(self, dataset_dir, dataset_name, remove_errors=True, remove_warnings=False):
        """清理数据集，移除有问题的图像"""
        print(f"\n--- 清理 {dataset_name} 数据集 ---")
        
        cleaned_dir = os.path.join(self.base_dir, f'{dataset_name}_cleaned')
        if os.path.exists(cleaned_dir):
            shutil.rmtree(cleaned_dir)
        os.makedirs(cleaned_dir)
        
        total_removed = 0
        total_kept = 0
        
        for class_name in sorted(os.listdir(dataset_dir)):
            class_path = os.path.join(dataset_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            
            # 创建清理后的类别目录
            clean_class_path = os.path.join(cleaned_dir, class_name)
            os.makedirs(clean_class_path)
            
            removed_count = 0
            kept_count = 0
            
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                result = self.check_image_quality(img_path)
                
                should_remove = False
                if result['status'] == 'error' and remove_errors:
                    should_remove = True
                elif result['status'] == 'warning' and remove_warnings:
                    should_remove = True
                
                if should_remove:
                    removed_count += 1
                    total_removed += 1
                else:
                    # 复制到清理后的目录
                    shutil.copy2(img_path, os.path.join(clean_class_path, img_file))
                    kept_count += 1
                    total_kept += 1
            
            print(f"类别 {class_name}: 保留 {kept_count}, 移除 {removed_count}")
        
        print(f"\n清理完成:")
        print(f"总保留: {total_kept}")
        print(f"总移除: {total_removed}")
        print(f"清理后数据保存在: {cleaned_dir}")
        
        return cleaned_dir
    
    def balance_dataset(self, dataset_dir):
        """平衡数据集，确保每个类别的样本数量相近"""
        print(f"\n--- 平衡数据集 ---")
        
        # 统计每个类别的样本数
        class_counts = {}
        for class_name in sorted(os.listdir(dataset_dir)):
            class_path = os.path.join(dataset_dir, class_name)
            if os.path.isdir(class_path):
                class_counts[class_name] = len(os.listdir(class_path))
        
        # 找到最小样本数
        min_samples = min(class_counts.values())
        print(f"最小样本数: {min_samples}")
        
        # 创建平衡后的数据集目录
        balanced_dir = os.path.join(self.base_dir, 'balanced_dataset')
        if os.path.exists(balanced_dir):
            shutil.rmtree(balanced_dir)
        os.makedirs(balanced_dir)
        
        for class_name, count in class_counts.items():
            print(f"类别 {class_name}: {count} -> {min_samples}")
            
            class_path = os.path.join(dataset_dir, class_name)
            balanced_class_path = os.path.join(balanced_dir, class_name)
            os.makedirs(balanced_class_path)
            
            # 随机选择样本
            all_files = os.listdir(class_path)
            np.random.shuffle(all_files)
            selected_files = all_files[:min_samples]
            
            for file_name in selected_files:
                shutil.copy2(
                    os.path.join(class_path, file_name),
                    os.path.join(balanced_class_path, file_name)
                )
        
        print(f"平衡后数据保存在: {balanced_dir}")
        return balanced_dir
    
    def visualize_samples(self, dataset_dir, samples_per_class=5):
        """可视化每个类别的样本"""
        class_names = sorted(os.listdir(dataset_dir))
        
        fig, axes = plt.subplots(len(class_names), samples_per_class, 
                                figsize=(samples_per_class*2, len(class_names)*2))
        
        for i, class_name in enumerate(class_names):
            class_path = os.path.join(dataset_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            files = os.listdir(class_path)[:samples_per_class]
            
            for j, file_name in enumerate(files):
                img_path = os.path.join(class_path, file_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if len(class_names) == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]
                
                ax.imshow(img, cmap='gray')
                ax.set_title(f'{class_name}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, 'dataset_samples.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"样本可视化保存为: dataset_samples.png")
    
    def generate_quality_report(self):
        """生成完整的数据质量报告"""
        print("\n=== 数据质量检查开始 ===")
        
        # 分析训练集
        train_stats = self.analyze_dataset(self.train_dir, "训练")
        
        # 分析测试集
        test_stats = self.analyze_dataset(self.test_dir, "测试")
        
        # 可视化样本
        self.visualize_samples(self.train_dir)
        
        # 生成报告文件
        report_content = f"""# 数据质量检查报告

## 训练集统计
"""
        for class_name, stats in train_stats.items():
            report_content += f"""
### 类别: {class_name}
- 总数: {stats['total']}
- 正常: {stats['ok']}
- 警告: {stats['warnings']}
- 错误: {stats['errors']}
"""

        report_content += f"""
## 测试集统计
"""
        for class_name, stats in test_stats.items():
            report_content += f"""
### 类别: {class_name}
- 总数: {stats['total']}
- 正常: {stats['ok']}
- 警告: {stats['warnings']}
- 错误: {stats['errors']}
"""
        
        with open(os.path.join(self.base_dir, 'data_quality_report.md'), 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n数据质量报告已保存: data_quality_report.md")
        
        return train_stats, test_stats

def main():
    # 使用示例
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checker = DataQualityChecker(base_dir)
    
    # 生成质量报告
    train_stats, test_stats = checker.generate_quality_report()
    
    # 询问是否清理数据
    print("\n是否要清理数据集? (y/n): ", end="")
    if input().lower() == 'y':
        # 清理训练集（移除错误，保留警告）
        cleaned_train = checker.clean_dataset(checker.train_dir, "training", 
                                            remove_errors=True, remove_warnings=False)
        
        # 清理测试集
        cleaned_test = checker.clean_dataset(checker.test_dir, "testing", 
                                           remove_errors=True, remove_warnings=False)
        
        print(f"\n清理完成！")
        print(f"使用清理后的数据进行训练:")
        print(f"训练集: {cleaned_train}")
        print(f"测试集: {cleaned_test}")

if __name__ == "__main__":
    main()