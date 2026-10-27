import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 环境与路径配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 优先使用增强后的数据，如果不存在则使用原始数据
AUGMENTED_TRAIN_DIR = os.path.join(BASE_DIR, 'training_augmented')
ORIGINAL_TRAIN_DIR = os.path.join(BASE_DIR, 'training')
TEST_DIR = os.path.join(BASE_DIR, 'testing')

# 选择训练数据目录
if os.path.exists(AUGMENTED_TRAIN_DIR):
    TRAIN_DIR = AUGMENTED_TRAIN_DIR
    print("使用增强后的训练数据")
else:
    TRAIN_DIR = ORIGINAL_TRAIN_DIR
    print("使用原始训练数据")

# 检查数据集目录是否存在
if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
    print(f"错误: 找不到数据集目录。请确保 {BASE_DIR} 下有 'training' 和 'testing' 文件夹。")
    exit()

# ==========================================
# 2. 自动类别检测与数据统计
# ==========================================
class_names = sorted(os.listdir(TRAIN_DIR))
num_classes = len(class_names)
print(f"--- [检测结果] 识别类别总数: {num_classes} ---")
print(f"--- [类别映射] {class_names} ---")

# 统计每个类别的样本数量
print("\n--- [数据集统计] ---")
for i, class_name in enumerate(class_names):
    train_count = len(os.listdir(os.path.join(TRAIN_DIR, class_name)))
    test_count = len(os.listdir(os.path.join(TEST_DIR, class_name)))
    print(f"类别 [{i}:{class_name}] - 训练: {train_count}, 测试: {test_count}")

# ==========================================
# 3. 优化的数据预处理与增强
# ==========================================
IMG_SHAPE = (28, 28, 1)
BATCH_SIZE = 64  # 增加批次大小以提高训练效率

# 训练数据增强 - 增加数据多样性
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,          # 旋转角度
    width_shift_range=0.1,      # 水平平移
    height_shift_range=0.1,     # 垂直平移
    shear_range=0.1,           # 剪切变换
    zoom_range=0.1,            # 缩放
    horizontal_flip=False,      # 数学符号不适合水平翻转
    fill_mode='constant',       # 填充模式
    cval=0,                    # 填充值（黑色背景）
    validation_split=0.2       # 从训练集中分出20%作为验证集
)

# 测试数据只做归一化
test_datagen = ImageDataGenerator(rescale=1./255)

# 创建数据生成器
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(28, 28),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='sparse',
    subset='training',          # 使用训练子集
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(28, 28),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='sparse',
    subset='validation',        # 使用验证子集
    shuffle=False
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(28, 28),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='sparse',
    shuffle=False
)

# ==========================================
# 4. 构建优化的CNN模型
# ==========================================
def create_optimized_model(num_classes):
    model = models.Sequential([
        # 第一个卷积块
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SHAPE, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 第二个卷积块
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 第三个卷积块
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # 全连接层
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        
        # 输出层
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

model = create_optimized_model(num_classes)

# 使用更好的优化器配置
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']  # 只使用基本准确率
)

# 显示模型结构
print("\n--- [模型结构] ---")
model.summary()

# ==========================================
# 5. 设置训练回调函数
# ==========================================
callbacks = [
    # 早停：验证损失不再改善时停止训练
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    
    # 学习率调度：验证损失停滞时降低学习率
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    
    # 模型检查点：保存最佳模型
    ModelCheckpoint(
        os.path.join(BASE_DIR, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ==========================================
# 6. 模型训练
# ==========================================
print("\n--- [开始训练] 正在优化网络权重... ---")

# 计算训练步数
steps_per_epoch = train_gen.samples // BATCH_SIZE
validation_steps = val_gen.samples // BATCH_SIZE

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=50,  # 增加训练轮数，配合早停使用
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

# ==========================================
# 7. 模型评估与可视化
# ==========================================
print("\n--- [模型评估] ---")

# 在测试集上评估
test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)
print(f"测试集准确率: {test_accuracy:.4f}")

# 生成预测结果
test_gen.reset()
predictions = model.predict(test_gen, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

# 获取真实标签
true_classes = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

# 生成分类报告
print("\n--- [详细分类报告] ---")
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# 生成混淆矩阵
print("\n--- [混淆矩阵] ---")
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# 绘制训练历史
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 准确率曲线
    ax1.plot(history.history['accuracy'], label='训练准确率')
    ax1.plot(history.history['val_accuracy'], label='验证准确率')
    ax1.set_title('模型准确率')
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('准确率')
    ax1.legend()
    ax1.grid(True)
    
    # 损失曲线
    ax2.plot(history.history['loss'], label='训练损失')
    ax2.plot(history.history['val_loss'], label='验证损失')
    ax2.set_title('模型损失')
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('损失')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

plot_training_history(history)

# ==========================================
# 8. 优化的TFLite量化
# ==========================================
print("\n--- [模型量化] 正在执行优化的Int8转换... ---")

def representative_data_gen():
    """代表性数据生成器，用于量化校准"""
    dataset = tf.data.Dataset.from_generator(
        lambda: train_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )
    
    for i, (images, _) in enumerate(dataset.take(100)):
        yield [images]

# 创建转换器
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# 设置量化参数
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# 执行转换
try:
    tflite_model_quant = converter.convert()
    print("✓ 量化转换成功")
except Exception as e:
    print(f"量化转换失败: {e}")
    # 降级到float16量化
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model_quant = converter.convert()
    print("✓ 使用Float16量化作为备选方案")

# 保存量化模型
tflite_file = os.path.join(BASE_DIR, "optimized_model_quant.tflite")
with open(tflite_file, "wb") as f:
    f.write(tflite_model_quant)

# ==========================================
# 9. 生成优化的C头文件
# ==========================================
print("\n--- [导出数据] 正在生成优化的嵌入式C语言代码... ---")

def convert_to_c_header(tflite_content, array_name="g_optimized_model_data"):
    hex_lines = [f"0x{byte:02x}" for byte in tflite_content]
    
    # 优化格式化，每行16个十六进制数
    formatted_hex = ""
    for i in range(0, len(hex_lines), 16):
        line = "  " + ", ".join(hex_lines[i:i+16])
        if i + 16 < len(hex_lines):
            line += ","
        formatted_hex += line + "\n"
    
    # 生成详细的类别映射注释
    class_map_note = "// 类别映射关系:\n"
    for i, name in enumerate(class_names):
        class_map_note += f"// [{i}] = {name}\n"
    
    # 添加模型信息注释
    model_info = f"""// 模型信息:
// - 输入尺寸: 28x28x1 (灰度图像)
// - 输出类别: {num_classes}
// - 量化类型: INT8
// - 模型大小: {len(tflite_content)} bytes
"""
    
    header_text = f"""/* 优化的TensorFlow Lite Micro模型数据 */
/* 自动生成，请勿手动修改 */
/* 生成时间: {tf.timestamp()} */

#ifndef OPTIMIZED_MODEL_DATA_H
#define OPTIMIZED_MODEL_DATA_H

#include <stdint.h>

{model_info}
{class_map_note}

// 模型数据数组
const unsigned char {array_name}[] __attribute__((aligned(16))) = {{
{formatted_hex}}};

// 模型大小
const unsigned int {array_name}_len = {len(tflite_content)};

// 类别数量
const unsigned int num_classes = {num_classes};

// 输入维度
const unsigned int input_width = 28;
const unsigned int input_height = 28;
const unsigned int input_channels = 1;

#endif /* OPTIMIZED_MODEL_DATA_H */
"""
    return header_text

header_file = os.path.join(BASE_DIR, "optimized_model_data.h")
with open(header_file, "w", encoding='utf-8') as f:
    f.write(convert_to_c_header(tflite_model_quant))

# ==========================================
# 10. 生成训练报告
# ==========================================
def generate_training_report():
    report_content = f"""
# 训练报告

## 模型配置
- 类别数量: {num_classes}
- 输入尺寸: 28x28x1
- 批次大小: {BATCH_SIZE}
- 训练轮数: {len(history.history['accuracy'])}

## 性能指标
- 最终训练准确率: {history.history['accuracy'][-1]:.4f}
- 最终验证准确率: {history.history['val_accuracy'][-1]:.4f}
- 测试集准确率: {test_accuracy:.4f}

## 类别映射
{chr(10).join([f"[{i}] {name}" for i, name in enumerate(class_names)])}

## 文件输出
- 最佳模型: best_model.h5
- 量化模型: optimized_model_quant.tflite
- C头文件: optimized_model_data.h
- 混淆矩阵: confusion_matrix.png
- 训练历史: training_history.png
"""
    
    with open(os.path.join(BASE_DIR, "training_report.md"), "w", encoding='utf-8') as f:
        f.write(report_content)

generate_training_report()

print(f"\n--- [训练完成] ---")
print(f"✓ 最佳模型已保存: best_model.h5")
print(f"✓ 量化模型已保存: {tflite_file}")
print(f"✓ C头文件已生成: {header_file}")
print(f"✓ 训练报告已生成: training_report.md")
print(f"✓ 可视化图表已保存: confusion_matrix.png, training_history.png")
print(f"\n最终测试准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print("提示: 可以直接将生成的.h文件用于嵌入式项目中!")