import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ==========================================
# 1. 环境与路径配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 优先使用增强后的数据
AUGMENTED_TRAIN_DIR = os.path.join(BASE_DIR, 'training_augmented')
ORIGINAL_TRAIN_DIR = os.path.join(BASE_DIR, 'training')
TEST_DIR = os.path.join(BASE_DIR, 'testing')

if os.path.exists(AUGMENTED_TRAIN_DIR):
    TRAIN_DIR = AUGMENTED_TRAIN_DIR
    print("✓ 使用增强后的训练数据")
else:
    TRAIN_DIR = ORIGINAL_TRAIN_DIR
    print("✓ 使用原始训练数据")

if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
    print(f"错误: 找不到数据集目录。请确保 {BASE_DIR} 下有 'training' 和 'testing' 文件夹。")
    exit()

# ==========================================
# 2. 自动类别检测 
# ==========================================
# 通过扫描子文件夹名称自动获取分类列表（如 0-9）
# 文件夹的数量直接决定了模型最后一层输出神经元的个数
class_names = sorted(os.listdir(TRAIN_DIR))
num_classes = len(class_names)
print(f"--- [检测结果] 识别类别总数: {num_classes} ---")
print(f"--- [类别映射] {class_names} ---")

# ==========================================
# 3. 数据预处理与增强
# ==========================================
IMG_SHAPE = (28, 28, 1)
BATCH_SIZE = 64  # 增加批次大小提高训练效率

# 训练数据增强
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    validation_split=0.2  # 分出20%作为验证集
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(28, 28),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='sparse',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(28, 28),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='sparse',
    subset='validation',
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
# 4. 构建优化的轻量级CNN (适合嵌入式)
# ==========================================
model = models.Sequential([
    # 第一个卷积块 - 使用深度可分离卷积减少参数
    layers.Conv2D(8, (3, 3), activation='relu', input_shape=IMG_SHAPE, padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # 第二个卷积块 - 深度可分离卷积
    layers.SeparableConv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # 第三个卷积块
    layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same'),
    
    # 全连接层 - 精简设计
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- [模型结构] ---")
model.summary()

# ==========================================
# 5. 模型训练 (带早停和学习率调度)
# ==========================================
print("\n--- [开始训练] 正在优化网络权重... ---")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

steps_per_epoch = train_gen.samples // BATCH_SIZE
validation_steps = val_gen.samples // BATCH_SIZE

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

# 评估模型
test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)
print(f"\n✓ 测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# ==========================================
# 6. TFLite Int8 全整型量化
# ==========================================
print("\n--- [模型量化] 正在执行全整型 Int8 转换... ---")

def representative_data_gen():
    """代表性数据生成器，用于量化校准"""
    train_gen.reset()
    for _ in range(100):
        img, _ = next(train_gen)
        yield [img.astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

try:
    tflite_model_quant = converter.convert()
    print("✓ Int8量化成功")
except Exception as e:
    print(f"警告: Int8量化失败 ({e})，尝试Float16量化...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model_quant = converter.convert()
    print("✓ Float16量化成功")

tflite_file = os.path.join(BASE_DIR, "mnist_model_quant.tflite")
with open(tflite_file, "wb") as f:
    f.write(tflite_model_quant)

# ==========================================
# 7. 导出优化的C语言数组头文件
# ==========================================
print("\n--- [导出数据] 正在生成嵌入式 C 语言代码... ---")

def convert_to_c_header(tflite_content, array_name="g_mnist_model_data"):
    hex_lines = [f"0x{byte:02x}" for byte in tflite_content]
    
    # 每行16个十六进制数，增强可读性
    formatted_hex = ""
    for i in range(0, len(hex_lines), 16):
        line = "  " + ", ".join(hex_lines[i:i+16])
        if i + 16 < len(hex_lines):
            line += ","
        formatted_hex += line + "\n"
    
    # 生成类别映射注释
    class_map_note = "// 类别映射:\n"
    for i, name in enumerate(class_names):
        class_map_note += f"//   [{i}] = {name}\n"
    
    # 模型信息
    model_info = f"""// 模型信息:
//   输入: 28x28x1 (灰度图像)
//   输出: {num_classes} 类别
//   量化: INT8
//   大小: {len(tflite_content)} bytes
//   准确率: {test_accuracy*100:.2f}%
"""
    
    header_text = f"""/* TensorFlow Lite Micro 模型数据 */
/* 自动生成 - 请勿手动修改 */

#ifndef MNIST_MODEL_DATA_H
#define MNIST_MODEL_DATA_H

#include <stdint.h>

{model_info}
{class_map_note}

const unsigned char {array_name}[] __attribute__((aligned(16))) = {{
{formatted_hex}}};

const unsigned int {array_name}_len = {len(tflite_content)};
const unsigned int num_classes = {num_classes};
const unsigned int input_width = 28;
const unsigned int input_height = 28;
const unsigned int input_channels = 1;

#endif /* MNIST_MODEL_DATA_H */
"""
    return header_text

header_file = os.path.join(BASE_DIR, "mnist_model_data.h")
with open(header_file, "w", encoding='utf-8') as f:
    f.write(convert_to_c_header(tflite_model_quant))

print(f"\n--- [任务完成] ---")
print(f"✓ 量化模型: {tflite_file}")
print(f"✓ C头文件: {header_file}")
print(f"✓ 模型大小: {len(tflite_model_quant)} bytes")
print(f"✓ 测试准确率: {test_accuracy*100:.2f}%")
print("\n提示: 将 .h 文件包含到 RT-Thread 工程中即可使用！")