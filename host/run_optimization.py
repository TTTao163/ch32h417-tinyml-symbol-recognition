#!/usr/bin/env python3
"""
数学符号识别模型优化一键运行脚本
自动执行数据质量检查、数据增强、模型训练和性能分析的完整流程
"""

import os
import sys
import subprocess
import time
from datetime import datetime

class OptimizationRunner:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_file = os.path.join(self.base_dir, f'optimization_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def run_script(self, script_name, description, timeout=3600):
        """运行Python脚本"""
        self.log(f"开始执行: {description}")
        self.log(f"运行脚本: {script_name}")
        
        script_path = os.path.join(self.base_dir, script_name)
        if not os.path.exists(script_path):
            self.log(f"错误: 脚本文件不存在 - {script_path}")
            return False
        
        try:
            start_time = time.time()
            result = subprocess.run([sys.executable, script_path], 
                                  cwd=self.base_dir,
                                  timeout=timeout,
                                  capture_output=True,
                                  text=True,
                                  encoding='utf-8')
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                self.log(f"✓ {description} 完成 (耗时: {duration:.1f}秒)")
                if result.stdout:
                    self.log(f"输出: {result.stdout[-500:]}")  # 只记录最后500字符
                return True
            else:
                self.log(f"✗ {description} 失败 (返回码: {result.returncode})")
                if result.stderr:
                    self.log(f"错误: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log(f"✗ {description} 超时 (超过{timeout}秒)")
            return False
        except Exception as e:
            self.log(f"✗ {description} 异常: {str(e)}")
            return False
    
    def check_prerequisites(self):
        """检查前置条件"""
        self.log("检查前置条件...")
        
        # 检查数据目录
        train_dir = os.path.join(self.base_dir, 'training')
        test_dir = os.path.join(self.base_dir, 'testing')
        
        if not os.path.exists(train_dir):
            self.log(f"错误: 训练数据目录不存在 - {train_dir}")
            return False
            
        if not os.path.exists(test_dir):
            self.log(f"错误: 测试数据目录不存在 - {test_dir}")
            return False
        
        # 检查数据是否存在
        train_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        test_classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        
        if not train_classes:
            self.log("错误: 训练数据目录为空")
            return False
            
        if not test_classes:
            self.log("错误: 测试数据目录为空")
            return False
        
        self.log(f"✓ 发现训练类别: {train_classes}")
        self.log(f"✓ 发现测试类别: {test_classes}")
        
        # 检查必要的Python包
        required_packages = ['tensorflow', 'opencv-python', 'matplotlib', 'seaborn', 'scikit-learn', 'pandas', 'tqdm']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.log(f"错误: 缺少必要的Python包: {missing_packages}")
            self.log("请运行: pip install " + " ".join(missing_packages))
            return False
        
        self.log("✓ 所有前置条件满足")
        return True
    
    def interactive_mode(self):
        """交互模式"""
        self.log("=== 数学符号识别模型优化工具 ===")
        self.log("选择运行模式:")
        print("1. 完整优化流程 (推荐)")
        print("2. 仅数据质量检查")
        print("3. 仅数据增强")
        print("4. 仅模型训练")
        print("5. 仅性能分析")
        print("6. 自定义流程")
        
        choice = input("请选择 (1-6): ").strip()
        
        if choice == '1':
            return self.run_full_pipeline()
        elif choice == '2':
            return self.run_data_quality_check()
        elif choice == '3':
            return self.run_data_augmentation()
        elif choice == '4':
            return self.run_model_training()
        elif choice == '5':
            return self.run_performance_analysis()
        elif choice == '6':
            return self.run_custom_pipeline()
        else:
            self.log("无效选择")
            return False
    
    def run_full_pipeline(self):
        """运行完整优化流程"""
        self.log("开始完整优化流程...")
        
        steps = [
            ("data_quality_checker.py", "数据质量检查", 600),
            ("advanced_data_augmentation.py", "高级数据增强", 1800),
            ("train_optimized.py", "优化模型训练", 7200),
            ("model_analyzer.py", "模型性能分析", 600),
        ]
        
        success_count = 0
        for script, description, timeout in steps:
            if self.run_script(script, description, timeout):
                success_count += 1
            else:
                self.log(f"流程在 {description} 步骤失败，是否继续？")
                continue_choice = input("继续执行后续步骤？(y/n): ").strip().lower()
                if continue_choice != 'y':
                    break
        
        self.log(f"完整流程结束，成功执行 {success_count}/{len(steps)} 个步骤")
        return success_count == len(steps)
    
    def run_data_quality_check(self):
        """运行数据质量检查"""
        return self.run_script("data_quality_checker.py", "数据质量检查", 600)
    
    def run_data_augmentation(self):
        """运行数据增强"""
        return self.run_script("advanced_data_augmentation.py", "高级数据增强", 1800)
    
    def run_model_training(self):
        """运行模型训练"""
        return self.run_script("train_optimized.py", "优化模型训练", 7200)
    
    def run_performance_analysis(self):
        """运行性能分析"""
        return self.run_script("model_analyzer.py", "模型性能分析", 600)
    
    def run_custom_pipeline(self):
        """自定义流程"""
        self.log("自定义流程配置:")
        
        available_steps = [
            ("data_quality_checker.py", "数据质量检查"),
            ("advanced_data_augmentation.py", "高级数据增强"),
            ("train_optimized.py", "优化模型训练"),
            ("model_analyzer.py", "模型性能分析"),
        ]
        
        print("可用步骤:")
        for i, (script, description) in enumerate(available_steps, 1):
            print(f"{i}. {description}")
        
        selected_steps = input("请选择要执行的步骤 (用逗号分隔，如: 1,3,4): ").strip()
        
        try:
            step_indices = [int(x.strip()) - 1 for x in selected_steps.split(',')]
            selected_steps = [available_steps[i] for i in step_indices if 0 <= i < len(available_steps)]
        except:
            self.log("无效的步骤选择")
            return False
        
        success_count = 0
        for script, description in selected_steps:
            timeout = 7200 if 'train' in script else 1800 if 'augmentation' in script else 600
            if self.run_script(script, description, timeout):
                success_count += 1
        
        self.log(f"自定义流程结束，成功执行 {success_count}/{len(selected_steps)} 个步骤")
        return success_count == len(selected_steps)
    
    def generate_summary_report(self):
        """生成总结报告"""
        self.log("生成总结报告...")
        
        report_content = f"""# 模型优化流程总结报告

## 执行信息
- 执行时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 工作目录: {self.base_dir}
- 日志文件: {self.log_file}

## 生成的文件

### 数据处理文件
- data_quality_report.md: 数据质量分析报告
- dataset_samples.png: 数据集样本可视化
- training_augmented/: 增强后的训练数据 (如果执行了数据增强)

### 模型文件
- best_model.h5: 最佳训练模型
- optimized_model_quant.tflite: 量化后的TFLite模型
- optimized_model_data.h: 嵌入式C语言头文件

### 分析文件
- training_history.png: 训练历史曲线
- confusion_matrix.png: 混淆矩阵
- confidence_analysis.png: 置信度分析
- difficult_samples.png: 难分类样本分析
- comprehensive_analysis_report.md: 综合性能分析报告

## 使用建议

1. 查看数据质量报告，了解数据集状况
2. 检查训练历史曲线，确认模型收敛情况
3. 分析混淆矩阵，识别需要改进的类别
4. 查看难分类样本，了解模型局限性
5. 使用生成的.h文件部署到嵌入式设备

## 下一步优化建议

- 如果某些类别准确率较低，考虑增加该类别的训练数据
- 如果出现过拟合，可以增加数据增强或正则化
- 如果训练速度慢，可以调整批次大小或模型复杂度
- 定期重新评估模型性能，持续优化

详细信息请查看各个步骤生成的具体报告文件。
"""
        
        summary_file = os.path.join(self.base_dir, 'optimization_summary.md')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.log(f"总结报告已生成: {summary_file}")

def main():
    runner = OptimizationRunner()
    
    # 检查前置条件
    if not runner.check_prerequisites():
        print("前置条件检查失败，请解决问题后重新运行")
        return
    
    # 运行优化流程
    try:
        success = runner.interactive_mode()
        
        # 生成总结报告
        runner.generate_summary_report()
        
        if success:
            runner.log("🎉 优化流程成功完成！")
            print(f"\n详细日志请查看: {runner.log_file}")
            print("总结报告: optimization_summary.md")
        else:
            runner.log("⚠️ 优化流程部分失败，请查看日志了解详情")
            
    except KeyboardInterrupt:
        runner.log("用户中断执行")
    except Exception as e:
        runner.log(f"执行过程中发生异常: {str(e)}")

if __name__ == "__main__":
    main()