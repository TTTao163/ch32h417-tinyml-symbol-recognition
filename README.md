# CH32H417 + TinyML 手写符号识别

在国产双核 RISC-V MCU **CH32H417** 上，用 **RT-Thread + LVGL + TensorFlow Lite Micro** 跑通「摄像头取图 → 传统图像预处理 → INT8 量化 CNN 推理 → 界面显示」的完整边缘 AI 链路，全程无操作系统级 AI 框架、无 OpenCV、无浮点运算。

> 2026 年江苏省职业院校技能大赛 · 嵌入式系统应用开发竞赛 · 嵌入式 AI 应用任务

---

## 一句话架构

```
OV5640 (DVP, RGB565 224×168)
        │  DMA 双缓冲
        ▼
   ┌─────────────────────────────┐
   │  预处理链路（纯定点整数运算） │
   │  灰度 → 对比度拉伸 → Otsu    │
   │  → 中心包围盒 → 等比缩放     │
   │  → 28×28 int8 张量           │
   └─────────────────────────────┘
        │
        ▼
 TensorFlow Lite Micro（INT8 全整型量化，25 KB arena）
        │  10 类 argmax
        ▼
   LVGL 界面：状态 / 结果 / 28×28 特征图放大预览
```

---

## 硬件与软件栈

| 项目 | 选型 |
|---|---|
| 主控 | CH32H417（沁恒，双核 RISC-V：V3F + V5F，本工程运行在 V5F 核） |
| 操作系统 | RT-Thread 4.x |
| GUI | LVGL 8.2 |
| 摄像头 | OV5640，DVP 接口，RGB565 224×168 |
| 显示屏 | SPI/RGB LCD（LVGL 直绘） |
| 推理框架 | TensorFlow Lite Micro（INT8） |
| 模型 | 自训练深度可分离卷积网络，**23,744 字节** |
| 工具链 | MounRiver Studio / RISC-V Embedded GCC 12 |

---

## 目录结构

```
firmware/                                   # 固件应用层（官方模板之外的本人代码）
└── V5F/
    ├── User/main.c                         # 线程创建、任务调度入口
    ├── THREAD/
    │   ├── task.h                          # 任务 ID 枚举与任务函数声明
    │   ├── task0.c                         # LVGL 任务主菜单
    │   └── tflm_cam_task.c                 # 摄像头采集 + 推理线程与 UI 构建
    └── tflm/examples/handwritten_digit/
        ├── tflm_handwritten.cc             # TFLM 解释器初始化、推理循环、结果显示
        ├── img_process.c / .h              # 图像预处理链路（本人实现的重点）
        └── models/
            ├── mnist_model_data.h          # 自训练 INT8 模型（23.7 KB，部署用）
            └── baseline_mnist_model_data.h # 官方基线模型（270 KB，对照用）

host/                                       # 上位机模型训练工具链（Python）
├── train_to_tflm_DEPTHWISE_CONV_2D.py      # 训练 + INT8 量化 + 导出 C 数组
├── train_optimized.py                      # 高精度版训练（BN + Dropout + 学习率调度）
├── advanced_data_augmentation.py           # 旋转/噪声/模糊/形态学/弹性变形增强
├── data_quality_checker.py                 # 数据集质量体检与清洗
├── model_analyzer.py                       # 逐层算子与参数量分析
├── run_optimization.py                     # 一键流水线编排
├── augment_mnist.py                        # MNIST 数字 1 定向增强
├── generate_symbols.py                     # +/- 符号合成数据生成器
└── docs/                                   # 训练说明与优化报告

model/mnist_model_quant.tflite              # 量化模型产物（113 KB）
```

---

## 关键技术点

### 1. 预处理链路：把「AI 该看到的那 28×28」喂准

摄像头给的是 224×168 的整幅画面，而模型只要 28×28 的二值化手写符号。中间这一步做不好，模型再准也没用。整条链路全部是定点整数运算，跑在 MCU 上没有任何浮点库依赖：

| 步骤 | 做法 | 为什么这么做 |
|---|---|---|
| 灰度化 | RGB565 → 整数 BT.601 加权 `(30R + 59G + 11B) / 100` | 避免 `float` 乘法开销 |
| 对比度拉伸 | 以 8 像素步长采样求 min/max，线性映射到 0–255 | 现场光照不均，先拉开动态范围 |
| 自适应二值化 | Otsu 大津法，直方图按步长 4 采样以省时间 | 固定阈值换个现场就废 |
| 包围盒提取 | 只在画面中心 ±80 px 区域扫描黑像素，取最小外接框 | 抗边缘杂物的强干扰，同时把笔迹居中 |
| 等比缩放 | 长边缩放到 20 px（MNIST 标准留白比例），居中嵌入 28×28 | 直接拉伸会让扁符号变形，实测掉点明显 |
| 量化映射 | 背景 `-128`、笔迹 `127`；二值阈值额外 +15 偏置 | int8 量化的零点在 0，笔画虚边会被判成背景 |
| 形态学（可选） | 3×3 中值滤波 / 膨胀，冒泡排序取中值 | 摄像头噪点会打断笔画 |

入口函数 `process_mnist_optimized()` 把上面 6 步串成一次调用，输出直接写进解释器的 `input->data.int8`，不需要额外中转缓冲区。

### 2. TFLM 部署：把 RAM 抠出来

CH32H417 的片上 RAM 要同时喂 LVGL、DMA 双缓冲和 RT-Thread 线程栈，留给推理的余量很小。

- **算子白名单只注册 9 个**（`Conv2D`、`DepthwiseConv2D`、`MaxPool2D`、`FullyConnected`、`Softmax`、`Reshape`、`Shape`、`StridedSlice`、`Pack`），砍掉整张算子表省 flash
- **`tensor_arena` 固定 25 KB 静态全局**，配 `RecordingMicroAllocator`；数组用 `__attribute__((aligned(16)))` 保证 16 字节对齐。放在全局而不是线程栈里 —— 否则光 arena 就能把 2.5 KB 的 LVGL 线程栈打爆
- **模型放 `.text_data` 段**，直接 XIP 从 flash 读，不占 RAM
- **INT8 输入输出**，argmax 找最大 score 之外还加了 `score > 0` 的有效性判据（int8 量化下 0 是零概率的对应值），避免全空画面时硬输出一个符号

自训练模型只有 **23.7 KB**，比官方基线模型的 270 KB 小了 11 倍，识别效果反而更贴合现场：训练集是自己采集的符号图像，类别映射直接对齐到 `0 2 4 6 8 + - × ÷ %`。

### 3. RT-Thread 线程模型：状态机而不是动态创建

LVGL 的线程安全是这块最容易被坑的地方 —— 界面控件被推理线程和按键线程同时改就会花屏甚至死机。

- **LVGL 线程**（栈 2560 B）+ **LCD 互斥量**：所有 `lv_*` 调用前后加锁
- **按键扫描线程**（栈 1536 B）：独立轮询，不阻塞画面刷新
- **TFLM 推理线程**：不在中断里做推理，用 4 状态状态机 `空闲 → 已捕获 → 推理 → 回位` 驱动，每 10 ms 轮询一次
- 任务切换沿用模板的 `task_running_info` 结构，退出任务时统一 `rt_thread_delete` 收敛，避免线程句柄泄漏

### 4. UI：让模型「看得见」

调试阶段最大的帮助是把模型真正看到的 28×28 特征图放大 2 倍（2×2 像素块）以 RGB565 画到屏上。人一眼就能判断是「预处理糊了」还是「模型认错了」，比对着串口打印的 `##` 字符块快得多。

---

## 模型训练（host/）

数据集：**约 10 万张** 28×28 灰度图（训练集 8.6 万 + 测试集 1.3 万），10 类（`0 2 4 6 8 + - × ÷ %`）。运算符类别样本较少，由 `generate_symbols.py` 按线宽、位置偏移、旋转角、噪声强度四维参数合成扩充。

| 指标 | 数值 |
|---|---|
| 测试集准确率（训练日志记录） | **99.69%** |
| 模型参数量 | 102,050 |
| 量化方式 | INT8 全整型（代表性数据集校准） |
| tflite 体积 | 113,104 字节 |
| 部署用 C 数组 | 23,744 字节 |

```bash
cd host
pip install -r requirements.txt

# 一键：数据体检 → 增强 → 训练 → 量化 → 导出 C 数组
python run_optimization.py

# 或者分步
python data_quality_checker.py          # 数据集体检
python advanced_data_augmentation.py    # 高级数据增强
python train_optimized.py               # 高精度训练
python train_to_tflm_DEPTHWISE_CONV_2D.py   # 训练 + INT8 量化导出
python model_analyzer.py                # 逐层算子分析
```

导出后的 `mnist_model_data.h` 替换到 `firmware/V5F/tflm/examples/handwritten_digit/models/` 即可重新编译。

---

## 复现说明

本仓库**只包含本人实现的应用层代码**。底层外设驱动（GPIO/TIM/SPI/DVP/SDMMC）、LVGL 移植层、TFLM 运行时、RT-Thread 内核均来自沁恒官方 EVT 模板与各自开源项目，未重复提交。

完整编译步骤：

1. 从沁恒官网获取 CH32H417 EVT 与 `Template V2.9` 工程模板
2. 用本仓库 `firmware/V5F/` 下的文件覆盖模板中同名路径
3. MounRiver Studio 打开 `V5F.wvproj`，编译下载
4. 串口命令行输入任务启动命令，进入 LVGL 任务菜单

---

## 致谢

- 沁恒微电子 CH32H417 EVT、TFLM 移植教程
- TensorFlow Lite Micro、LVGL 8.2、RT-Thread 开源社区
