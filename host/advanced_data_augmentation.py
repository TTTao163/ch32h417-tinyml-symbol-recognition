import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class AdvancedMathSymbolAugmenter:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.train_dir = os.path.join(base_dir, 'training')
        self.augmented_dir = os.path.join(base_dir, 'training_augmented')
        
    def apply_noise(self, img, noise_type='gaussian', intensity=0.1):
        """添加各种类型的噪声"""
        if noise_type == 'gaussian':
            noise = np.random.normal(0, intensity * 255, img.shape)
            noisy_img = img + noise
        elif noise_type == 'salt_pepper':
            noisy_img = img.copy()
            # Salt noise
            salt_coords = tuple([np.random.randint(0, i - 1, int(intensity * img.size * 0.5)) 
                               for i in img.shape])
            noisy_img[salt_coords] = 255
            # Pepper noise
            pepper_coords = tuple([np.random.randint(0, i - 1, int(intensity * img.size * 0.5)) 
                                 for i in img.shape])
            noisy_img[pepper_coords] = 0
        elif noise_type == 'speckle':
            noise = np.random.randn(*img.shape)
            noisy_img = img + img * noise * intensity
        else:
            return img
            
        return np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    def apply_blur(self, img, blur_type='gaussian', kernel_size=3):
        """应用不同类型的模糊"""
        if blur_type == 'gaussian':
            return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        elif blur_type == 'motion':
            # 创建运动模糊核
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            return cv2.filter2D(img, -1, kernel)
        elif blur_type == 'median':
            return cv2.medianBlur(img, kernel_size)
        else:
            return img
    
    def apply_morphological_ops(self, img, operation='erosion', kernel_size=2):
        """应用形态学操作"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if operation == 'erosion':
            return cv2.erode(img, kernel, iterations=1)
        elif operation == 'dilation':
            return cv2.dilate(img, kernel, iterations=1)
        elif operation == 'opening':
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        elif operation == 'closing':
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        else:
            return img
    
    def apply_elastic_deformation(self, img, alpha=10, sigma=3):
        """应用弹性变形"""
        random_state = np.random.RandomState(None)
        shape = img.shape
        
        dx = random_state.rand(*shape) * 2 - 1
        dy = random_state.rand(*shape) * 2 - 1
        
        dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        return cv2.remap(img, indices[1].astype(np.float32), 
                        indices[0].astype(np.float32), cv2.INTER_LINEAR)
    
    def apply_perspective_transform(self, img, max_warp=0.1):
        """应用透视变换"""
        h, w = img.shape
        
        # 定义源点和目标点
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # 随机扰动目标点
        dst_points = src_points.copy()
        for i in range(4):
            dst_points[i][0] += random.uniform(-max_warp * w, max_warp * w)
            dst_points[i][1] += random.uniform(-max_warp * h, max_warp * h)
        
        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 应用变换
        return cv2.warpPerspective(img, matrix, (w, h))
    
    def apply_brightness_contrast(self, img, brightness=0, contrast=1.0):
        """调整亮度和对比度"""
        return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    
    def apply_rotation_with_crop(self, img, angle, crop_center=True):
        """旋转图像并智能裁剪"""
        h, w = img.shape
        center = (w // 2, h // 2)
        
        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 旋转图像
        rotated = cv2.warpAffine(img, rotation_matrix, (w, h), 
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                               borderValue=0)
        
        if crop_center:
            # 裁剪中心区域以避免黑边
            crop_size = int(min(w, h) * 0.8)
            start_x = (w - crop_size) // 2
            start_y = (h - crop_size) // 2
            cropped = rotated[start_y:start_y+crop_size, start_x:start_x+crop_size]
            # 调整回原始大小
            return cv2.resize(cropped, (w, h))
        
        return rotated
    
    def create_augmentation_pipeline(self, img, augmentation_level='medium'):
        """创建增强管道"""
        augmented_images = []
        
        if augmentation_level == 'light':
            augmentations = [
                lambda x: self.apply_rotation_with_crop(x, random.uniform(-10, 10)),
                lambda x: self.apply_brightness_contrast(x, random.randint(-20, 20), random.uniform(0.8, 1.2)),
                lambda x: self.apply_noise(x, 'gaussian', 0.05),
            ]
        elif augmentation_level == 'medium':
            augmentations = [
                lambda x: self.apply_rotation_with_crop(x, random.uniform(-15, 15)),
                lambda x: self.apply_brightness_contrast(x, random.randint(-30, 30), random.uniform(0.7, 1.3)),
                lambda x: self.apply_noise(x, 'gaussian', 0.1),
                lambda x: self.apply_blur(x, 'gaussian', random.choice([3, 5])),
                lambda x: self.apply_morphological_ops(x, random.choice(['erosion', 'dilation']), 2),
                lambda x: self.apply_perspective_transform(x, 0.05),
            ]
        else:  # heavy
            augmentations = [
                lambda x: self.apply_rotation_with_crop(x, random.uniform(-20, 20)),
                lambda x: self.apply_brightness_contrast(x, random.randint(-40, 40), random.uniform(0.6, 1.4)),
                lambda x: self.apply_noise(x, random.choice(['gaussian', 'salt_pepper']), 0.15),
                lambda x: self.apply_blur(x, random.choice(['gaussian', 'motion']), random.choice([3, 5, 7])),
                lambda x: self.apply_morphological_ops(x, random.choice(['erosion', 'dilation', 'opening']), random.choice([2, 3])),
                lambda x: self.apply_perspective_transform(x, 0.08),
                lambda x: self.apply_elastic_deformation(x, random.uniform(5, 15), random.uniform(2, 4)),
            ]
        
        # 应用随机组合的增强
        for _ in range(len(augmentations)):
            aug_img = img.copy()
            
            # 随机选择2-4个增强操作
            selected_augs = random.sample(augmentations, random.randint(2, min(4, len(augmentations))))
            
            for aug_func in selected_augs:
                try:
                    aug_img = aug_func(aug_img)
                    # 确保图像值在有效范围内
                    aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)
                except:
                    continue
            
            augmented_images.append(aug_img)
        
        return augmented_images
    
    def augment_class(self, class_name, target_count=None, augmentation_level='medium'):
        """增强特定类别的数据"""
        class_path = os.path.join(self.train_dir, class_name)
        if not os.path.exists(class_path):
            print(f"类别 {class_name} 不存在")
            return
        
        # 创建输出目录
        output_class_path = os.path.join(self.augmented_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)
        
        # 获取原始图像
        original_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        original_count = len(original_files)
        
        if target_count is None:
            target_count = original_count * 3  # 默认增加到3倍
        
        print(f"增强类别 {class_name}: {original_count} -> {target_count}")
        
        # 复制原始图像
        for file_name in original_files:
            src_path = os.path.join(class_path, file_name)
            dst_path = os.path.join(output_class_path, f"orig_{file_name}")
            cv2.imwrite(dst_path, cv2.imread(src_path, cv2.IMREAD_GRAYSCALE))
        
        # 生成增强图像
        augmented_count = 0
        needed_count = target_count - original_count
        
        with tqdm(total=needed_count, desc=f"增强 {class_name}") as pbar:
            while augmented_count < needed_count:
                # 随机选择一个原始图像
                source_file = random.choice(original_files)
                source_path = os.path.join(class_path, source_file)
                img = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                
                # 生成增强图像
                augmented_images = self.create_augmentation_pipeline(img, augmentation_level)
                
                for i, aug_img in enumerate(augmented_images):
                    if augmented_count >= needed_count:
                        break
                    
                    # 保存增强图像
                    base_name = os.path.splitext(source_file)[0]
                    aug_filename = f"aug_{augmented_count:06d}_{base_name}_{i}.png"
                    aug_path = os.path.join(output_class_path, aug_filename)
                    
                    cv2.imwrite(aug_path, aug_img)
                    augmented_count += 1
                    pbar.update(1)
    
    def augment_all_classes(self, target_count_per_class=None, augmentation_level='medium'):
        """增强所有类别的数据"""
        print("开始数据增强...")
        
        # 创建输出目录
        if os.path.exists(self.augmented_dir):
            import shutil
            shutil.rmtree(self.augmented_dir)
        os.makedirs(self.augmented_dir)
        
        # 获取所有类别
        class_names = [d for d in os.listdir(self.train_dir) 
                      if os.path.isdir(os.path.join(self.train_dir, d))]
        
        # 如果没有指定目标数量，找到最大的类别数量
        if target_count_per_class is None:
            max_count = 0
            for class_name in class_names:
                class_path = os.path.join(self.train_dir, class_name)
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                max_count = max(max_count, count)
            target_count_per_class = max_count * 2  # 增加到2倍
        
        print(f"目标每类样本数: {target_count_per_class}")
        
        # 增强每个类别
        for class_name in class_names:
            self.augment_class(class_name, target_count_per_class, augmentation_level)
        
        print(f"数据增强完成！增强后的数据保存在: {self.augmented_dir}")
    
    def visualize_augmentations(self, class_name, num_samples=3):
        """可视化增强效果"""
        class_path = os.path.join(self.train_dir, class_name)
        if not os.path.exists(class_path):
            print(f"类别 {class_name} 不存在")
            return
        
        # 获取原始图像
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        sample_files = random.sample(files, min(num_samples, len(files)))
        
        fig, axes = plt.subplots(num_samples, 7, figsize=(21, num_samples * 3))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, file_name in enumerate(sample_files):
            img_path = os.path.join(class_path, file_name)
            original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # 显示原始图像
            axes[i, 0].imshow(original_img, cmap='gray')
            axes[i, 0].set_title('原始')
            axes[i, 0].axis('off')
            
            # 生成并显示不同的增强效果
            augmentations = [
                ('旋转', lambda x: self.apply_rotation_with_crop(x, 15)),
                ('噪声', lambda x: self.apply_noise(x, 'gaussian', 0.1)),
                ('模糊', lambda x: self.apply_blur(x, 'gaussian', 3)),
                ('形态学', lambda x: self.apply_morphological_ops(x, 'erosion', 2)),
                ('透视', lambda x: self.apply_perspective_transform(x, 0.05)),
                ('弹性', lambda x: self.apply_elastic_deformation(x, 10, 3)),
            ]
            
            for j, (name, aug_func) in enumerate(augmentations):
                try:
                    aug_img = aug_func(original_img.copy())
                    axes[i, j+1].imshow(aug_img, cmap='gray')
                    axes[i, j+1].set_title(name)
                    axes[i, j+1].axis('off')
                except:
                    axes[i, j+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, f'augmentation_preview_{class_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"增强预览保存为: augmentation_preview_{class_name}.png")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    augmenter = AdvancedMathSymbolAugmenter(base_dir)
    
    print("高级数学符号数据增强工具")
    print("1. 预览增强效果")
    print("2. 增强所有类别 (轻度)")
    print("3. 增强所有类别 (中度)")
    print("4. 增强所有类别 (重度)")
    print("5. 自定义增强")
    
    choice = input("请选择操作 (1-5): ")
    
    if choice == '1':
        # 预览增强效果
        class_names = os.listdir(augmenter.train_dir)
        for class_name in class_names[:3]:  # 只预览前3个类别
            if os.path.isdir(os.path.join(augmenter.train_dir, class_name)):
                augmenter.visualize_augmentations(class_name)
    
    elif choice in ['2', '3', '4']:
        levels = {'2': 'light', '3': 'medium', '4': 'heavy'}
        level = levels[choice]
        
        target_count = int(input(f"每个类别的目标样本数 (默认20000): ") or "20000")
        augmenter.augment_all_classes(target_count, level)
    
    elif choice == '5':
        class_name = input("要增强的类别名称: ")
        target_count = int(input("目标样本数: "))
        level = input("增强级别 (light/medium/heavy): ") or "medium"
        
        augmenter.augment_class(class_name, target_count, level)
    
    print("操作完成！")

if __name__ == "__main__":
    main()