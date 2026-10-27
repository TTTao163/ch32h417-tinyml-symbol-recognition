import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

class ModelAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.test_dir = os.path.join(base_dir, 'testing')
        self.model = None
        self.class_names = None
        
    def load_model(self, model_path):
        """加载训练好的模型"""
        self.model = tf.keras.models.load_model(model_path)
        print(f"模型已加载: {model_path}")
        return self.model
    
    def prepare_test_data(self):
        """准备测试数据"""
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_gen = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(28, 28),
            batch_size=32,
            color_mode='grayscale',
            class_mode='sparse',
            shuffle=False
        )
        
        self.class_names = list(test_gen.class_indices.keys())
        return test_gen
    
    def analyze_model_performance(self, test_gen):
        """分析模型性能"""
        print("\n=== 模型性能分析 ===")
        
        # 基本评估
        test_loss, test_accuracy = self.model.evaluate(test_gen, verbose=0)
        print(f"测试损失: {test_loss:.4f}")
        print(f"测试准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # 生成预测
        test_gen.reset()
        predictions = self.model.predict(test_gen, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_gen.classes
        
        # 详细分类报告
        report = classification_report(true_classes, predicted_classes, 
                                     target_names=self.class_names, 
                                     output_dict=True)
        
        # 转换为DataFrame便于分析
        df_report = pd.DataFrame(report).transpose()
        print("\n详细分类报告:")
        print(df_report.round(4))
        
        # 保存报告
        df_report.to_csv(os.path.join(self.base_dir, 'classification_report.csv'))
        
        return predictions, predicted_classes, true_classes, report
    
    def analyze_confusion_matrix(self, true_classes, predicted_classes):
        """分析混淆矩阵"""
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # 计算每个类别的准确率
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        print("\n=== 混淆矩阵分析 ===")
        for i, (class_name, acc) in enumerate(zip(self.class_names, class_accuracy)):
            print(f"类别 {class_name}: {acc:.4f} ({acc*100:.2f}%)")
        
        # 可视化混淆矩阵
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, 'detailed_confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 分析最容易混淆的类别对
        confusion_pairs = []
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append({
                        'true_class': self.class_names[i],
                        'predicted_class': self.class_names[j],
                        'count': cm[i, j],
                        'percentage': cm[i, j] / cm[i].sum() * 100
                    })
        
        # 按混淆程度排序
        confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
        
        print("\n最容易混淆的类别对:")
        for pair in confusion_pairs[:10]:  # 显示前10个
            print(f"{pair['true_class']} -> {pair['predicted_class']}: "
                  f"{pair['count']} 次 ({pair['percentage']:.2f}%)")
        
        return cm, confusion_pairs
    
    def analyze_prediction_confidence(self, predictions, true_classes, predicted_classes):
        """分析预测置信度"""
        print("\n=== 预测置信度分析 ===")
        
        # 计算每个预测的最大置信度
        max_confidences = np.max(predictions, axis=1)
        
        # 分析正确和错误预测的置信度分布
        correct_mask = (predicted_classes == true_classes)
        correct_confidences = max_confidences[correct_mask]
        incorrect_confidences = max_confidences[~correct_mask]
        
        print(f"正确预测平均置信度: {correct_confidences.mean():.4f}")
        print(f"错误预测平均置信度: {incorrect_confidences.mean():.4f}")
        
        # 可视化置信度分布
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(correct_confidences, bins=50, alpha=0.7, label='正确预测', color='green')
        plt.hist(incorrect_confidences, bins=50, alpha=0.7, label='错误预测', color='red')
        plt.xlabel('预测置信度')
        plt.ylabel('频次')
        plt.title('预测置信度分布')
        plt.legend()
        
        # 按类别分析置信度
        plt.subplot(1, 2, 2)
        class_confidences = []
        for i, class_name in enumerate(self.class_names):
            class_mask = (true_classes == i)
            class_conf = max_confidences[class_mask]
            class_confidences.append(class_conf.mean())
        
        plt.bar(range(len(self.class_names)), class_confidences)
        plt.xlabel('类别')
        plt.ylabel('平均置信度')
        plt.title('各类别平均预测置信度')
        plt.xticks(range(len(self.class_names)), self.class_names, rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, 'confidence_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return max_confidences, correct_confidences, incorrect_confidences
    
    def analyze_difficult_samples(self, test_gen, predictions, true_classes, predicted_classes, top_k=20):
        """分析最难分类的样本"""
        print(f"\n=== 分析最难分类的 {top_k} 个样本 ===")
        
        # 计算预测置信度
        max_confidences = np.max(predictions, axis=1)
        
        # 找到错误预测中置信度最高的样本（最"自信"的错误）
        incorrect_mask = (predicted_classes != true_classes)
        incorrect_indices = np.where(incorrect_mask)[0]
        incorrect_confidences = max_confidences[incorrect_mask]
        
        # 按置信度排序（降序）
        sorted_indices = np.argsort(incorrect_confidences)[::-1]
        most_confident_errors = incorrect_indices[sorted_indices[:top_k]]
        
        # 获取测试图像
        test_gen.reset()
        all_images = []
        all_labels = []
        
        for i in range(len(test_gen)):
            batch_images, batch_labels = test_gen[i]
            all_images.extend(batch_images)
            all_labels.extend(batch_labels)
            if len(all_images) >= len(true_classes):
                break
        
        all_images = np.array(all_images[:len(true_classes)])
        
        # 可视化最难分类的样本
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, idx in enumerate(most_confident_errors):
            if i >= 20:
                break
                
            img = all_images[idx].squeeze()
            true_label = self.class_names[true_classes[idx]]
            pred_label = self.class_names[predicted_classes[idx]]
            confidence = max_confidences[idx]
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'真实: {true_label}\n预测: {pred_label}\n置信度: {confidence:.3f}', 
                            fontsize=8)
            axes[i].axis('off')
        
        plt.suptitle('最难分类的样本（高置信度错误预测）')
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, 'difficult_samples.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 分析低置信度正确预测
        correct_mask = (predicted_classes == true_classes)
        correct_indices = np.where(correct_mask)[0]
        correct_confidences = max_confidences[correct_mask]
        
        # 找到置信度最低的正确预测
        sorted_correct_indices = np.argsort(correct_confidences)
        least_confident_correct = correct_indices[sorted_correct_indices[:top_k]]
        
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, idx in enumerate(least_confident_correct):
            if i >= 20:
                break
                
            img = all_images[idx].squeeze()
            true_label = self.class_names[true_classes[idx]]
            confidence = max_confidences[idx]
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'类别: {true_label}\n置信度: {confidence:.3f}', fontsize=8)
            axes[i].axis('off')
        
        plt.suptitle('低置信度正确预测样本')
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, 'low_confidence_correct.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return most_confident_errors, least_confident_correct
    
    def analyze_feature_maps(self, test_gen, layer_names=None):
        """分析特征图"""
        if layer_names is None:
            # 自动选择一些卷积层
            layer_names = []
            for layer in self.model.layers:
                if 'conv' in layer.name.lower():
                    layer_names.append(layer.name)
        
        print(f"\n=== 分析特征图 (层: {layer_names}) ===")
        
        # 创建特征提取模型
        feature_extractor = tf.keras.Model(
            inputs=self.model.input,
            outputs=[self.model.get_layer(name).output for name in layer_names]
        )
        
        # 获取一个测试样本
        test_gen.reset()
        sample_batch, _ = next(test_gen)
        sample_image = sample_batch[0:1]  # 取第一个样本
        
        # 提取特征
        features = feature_extractor(sample_image)
        
        # 可视化特征图
        for i, (feature, layer_name) in enumerate(zip(features, layer_names)):
            feature = feature[0]  # 移除batch维度
            
            # 选择前16个特征图进行可视化
            n_features = min(16, feature.shape[-1])
            
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            axes = axes.flatten()
            
            for j in range(n_features):
                axes[j].imshow(feature[:, :, j], cmap='viridis')
                axes[j].set_title(f'特征图 {j+1}')
                axes[j].axis('off')
            
            # 隐藏多余的子图
            for j in range(n_features, 16):
                axes[j].axis('off')
            
            plt.suptitle(f'层 {layer_name} 的特征图')
            plt.tight_layout()
            plt.savefig(os.path.join(self.base_dir, f'feature_maps_{layer_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("\n=== 生成综合分析报告 ===")
        
        # 准备测试数据
        test_gen = self.prepare_test_data()
        
        # 性能分析
        predictions, predicted_classes, true_classes, report = self.analyze_model_performance(test_gen)
        
        # 混淆矩阵分析
        cm, confusion_pairs = self.analyze_confusion_matrix(true_classes, predicted_classes)
        
        # 置信度分析
        max_confidences, correct_confidences, incorrect_confidences = self.analyze_prediction_confidence(
            predictions, true_classes, predicted_classes)
        
        # 难分类样本分析
        difficult_errors, low_conf_correct = self.analyze_difficult_samples(
            test_gen, predictions, true_classes, predicted_classes)
        
        # 特征图分析
        self.analyze_feature_maps(test_gen)
        
        # 生成文本报告
        report_content = f"""# 模型性能综合分析报告

## 基本性能指标
- 测试准确率: {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)
- 宏平均F1分数: {report['macro avg']['f1-score']:.4f}
- 加权平均F1分数: {report['weighted avg']['f1-score']:.4f}

## 各类别性能
"""
        
        for class_name in self.class_names:
            if class_name in report:
                class_report = report[class_name]
                report_content += f"""
### 类别: {class_name}
- 精确率: {class_report['precision']:.4f}
- 召回率: {class_report['recall']:.4f}
- F1分数: {class_report['f1-score']:.4f}
- 支持样本数: {class_report['support']}
"""

        report_content += f"""
## 置信度分析
- 正确预测平均置信度: {correct_confidences.mean():.4f}
- 错误预测平均置信度: {incorrect_confidences.mean():.4f}
- 置信度差异: {correct_confidences.mean() - incorrect_confidences.mean():.4f}

## 最容易混淆的类别对
"""
        
        for pair in confusion_pairs[:5]:
            report_content += f"- {pair['true_class']} -> {pair['predicted_class']}: {pair['count']} 次 ({pair['percentage']:.2f}%)\n"
        
        report_content += f"""
## 生成的文件
- classification_report.csv: 详细分类报告
- detailed_confusion_matrix.png: 混淆矩阵
- confidence_analysis.png: 置信度分析
- difficult_samples.png: 最难分类样本
- low_confidence_correct.png: 低置信度正确预测
- feature_maps_*.png: 各层特征图
"""
        
        with open(os.path.join(self.base_dir, 'comprehensive_analysis_report.md'), 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print("综合分析报告已生成: comprehensive_analysis_report.md")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    analyzer = ModelAnalyzer(base_dir)
    
    # 查找模型文件
    model_files = [f for f in os.listdir(base_dir) if f.endswith('.h5')]
    
    if not model_files:
        print("未找到模型文件(.h5)，请先训练模型")
        return
    
    print("找到的模型文件:")
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    
    choice = input(f"请选择模型文件 (1-{len(model_files)}): ")
    try:
        model_index = int(choice) - 1
        model_path = os.path.join(base_dir, model_files[model_index])
    except:
        model_path = os.path.join(base_dir, model_files[0])
    
    # 加载模型
    analyzer.load_model(model_path)
    
    # 生成综合分析报告
    analyzer.generate_comprehensive_report()
    
    print("\n模型分析完成！")

if __name__ == "__main__":
    main()