#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// ================= 宏定义配置 =================
#define IMG_W              224
#define IMG_H              168
#define CROP_SIZE          168      // 采集中心区域大小 (224*224)
#define MODEL_SIZE         28       // MNIST模型输入大小
#define TARGET_SIZE        20       // 数字缩放后的目标边长 (建议18-22)

#define ENABLE_STRETCH     1        // 是否开启自动对比度拉伸
#define ENABLE_FILTER      0        // 是否开启中值滤波
#define FILTER_DIM         3        // 滤波窗口大小 (3x3 或 5x5)
#define OTSU_STEP          4        // 大津法采样步长,不用改

typedef struct { int x, y, w, h; } BBox;

// ================= 内部辅助函数 =================

// 快速排序 (用于中值滤波)
static void simple_sort(uint8_t *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                uint8_t t = arr[j]; arr[j] = arr[j + 1]; arr[j + 1] = t;
            }
        }
    }
}

// ================= 核心算法步骤 =================

// 1. 灰度转换与对比度拉伸 (原位处理)
void img_preprocess(uint16_t *src_buf) {
    uint8_t *gray_ptr = (uint8_t *)src_buf;
    int off_x = (IMG_W - CROP_SIZE) / 2;
    int off_y = (IMG_H - CROP_SIZE) / 2;
    uint8_t min_g = 255, max_g = 0;

#if ENABLE_STRETCH
    // 采样找极值
    for (int y = 0; y < CROP_SIZE; y += 8) {
        uint16_t *line = &src_buf[(y + off_y) * IMG_W + off_x];
        for (int x = 0; x < CROP_SIZE; x += 8) {
            uint16_t p = line[x];
            uint8_t g = ((((p >> 8) & 0xF8) * 30) + (((p >> 3) & 0xFC) * 59) + (((p << 3) & 0xF8) * 11)) / 100;
            if (g < min_g) min_g = g; if (g > max_g) max_g = g;
        }
    }
    int diff = (max_g - min_g > 15) ? (max_g - min_g) : 1;
#endif

    // 全量转换：必须从左上到右下，确保不破坏未读取的像素
    for (int y = 0; y < CROP_SIZE; y++) {
        uint16_t *src_line = &src_buf[(y + off_y) * IMG_W + off_x];
        for (int x = 0; x < CROP_SIZE; x++) {
            uint16_t p = src_line[x];
            uint8_t g = ((((p >> 8) & 0xF8) * 30) + (((p >> 3) & 0xFC) * 59) + (((p << 3) & 0xF8) * 11)) / 100;
#if ENABLE_STRETCH
            int s = ((int)g - min_g) * 255 / diff;
            gray_ptr[y * CROP_SIZE + x] = (uint8_t)(s > 255 ? 255 : (s < 0 ? 0 : s));
#else
            gray_ptr[y * CROP_SIZE + x] = g;
#endif
        }
    }
}

// 2. 大津法自动阈值
uint8_t img_get_otsu_threshold(uint8_t *gray_buf) {
    uint32_t hist[256] = {0};
    for (int i = 0; i < CROP_SIZE * CROP_SIZE; i += OTSU_STEP) hist[gray_buf[i]]++;
    float sum = 0, sumB = 0, varMax = 0;
    for (int i = 0; i < 256; i++) sum += (float)i * hist[i];
    uint32_t wB = 0, total = (CROP_SIZE * CROP_SIZE) / OTSU_STEP;
    uint8_t threshold = 127;
    for (int i = 0; i < 256; i++) {
        wB += hist[i]; if (wB == 0) continue;
        uint32_t wF = total - wB; if (wF == 0) break;
        sumB += (float)(i * hist[i]);
        float mB = sumB / (float)wB, mF = (sum - sumB) / (float)wF;
        float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);
        if (varBetween > varMax) { varMax = varBetween; threshold = i; }
    }
    return threshold;
}

// 3. 中心向四周搜索边界 (BBox)
bool img_get_bounding_box_center_out(uint8_t *gray_buf, uint8_t thresh, BBox *box) {
    int cx = CROP_SIZE / 2;
    int cy = CROP_SIZE / 2;
    int min_x = CROP_SIZE, max_x = 0, min_y = CROP_SIZE, max_y = 0;
    bool found = false;

    // 从中心向外扩散扫描 (十字星型搜索 + 局部包围)
    // 逻辑：寻找离中心点 100 像素范围内的所有有效像素
    for (int y = cy - 80; y < cy + 80; y++) {
        if (y < 0 || y >= CROP_SIZE) continue;
        uint8_t *line = &gray_buf[y * CROP_SIZE];
        for (int x = cx - 80; x < cx + 80; x++) {
            if (x < 0 || x >= CROP_SIZE) continue;
            // MNIST默认白纸黑字：像素 < 阈值
            if (line[x] < thresh) { 
                if (x < min_x) min_x = x; if (x > max_x) max_x = x;
                if (y < min_y) min_y = y; if (y > max_y) max_y = y;
                found = true;
            }
        }
    }

    if (found) {
        box->x = min_x; box->y = min_y;
        box->w = max_x - min_x + 1; box->h = max_y - min_y + 1;
        // 过滤极其微小的噪点
        if (box->w < 3 || box->h < 3) return false;
    }
    return found;
}

// 增加一个简单的膨胀函数，用于连接断裂笔画
void img_dilate(int8_t *model_input) {
    int8_t temp[MODEL_SIZE * MODEL_SIZE];
    memcpy(temp, model_input, sizeof(temp));

    for (int y = 1; y < MODEL_SIZE - 1; y++) {
        for (int x = 1; x < MODEL_SIZE - 1; x++) {
            // 如果原始像素是背景(-128)，但四周有数字(127)
            if (temp[y * MODEL_SIZE + x] < 0) {
                if (temp[(y-1)*MODEL_SIZE + x] > 0 || temp[(y+1)*MODEL_SIZE + x] > 0 ||
                    temp[y*MODEL_SIZE + (x-1)] > 0 || temp[y*MODEL_SIZE + (x+1)] > 0) {
                    model_input[y * MODEL_SIZE + x] = 127; // 膨胀
                }
            }
        }
    }
}

// 4. 等比例缩放映射与中值滤波处理
void img_final_map(uint8_t *gray_buf, BBox *box, uint8_t thresh, int8_t *model_input) {
    // 初始化模型背景为黑 (-128)
    memset(model_input, -128, MODEL_SIZE * MODEL_SIZE);

   // --- 阈值补偿 ---
    // 稍微调高阈值（+15），让笔画变粗
    int adjusted_thresh = (int)thresh + 15; 
    if (adjusted_thresh > 250) adjusted_thresh = 250;

    int max_side = (box->w > box->h) ? box->w : box->h;
    float scale = (float)TARGET_SIZE / max_side;
    int nw = box->w * scale;
    int nh = box->h * scale;
    int ox = (MODEL_SIZE - nw) / 2;
    int oy = (MODEL_SIZE - nh) / 2;

    for (int y = 0; y < nh; y++) {
        for (int x = 0; x < nw; x++) {
            int sx = box->x + (int)(x / scale);
            int sy = box->y + (int)(y / scale);
            if (gray_buf[sy * CROP_SIZE + sx] < adjusted_thresh) {
                model_input[(y + oy) * MODEL_SIZE + (x + ox)] = 127;
            }
        }
    }

    // 执行一次膨胀，连接断裂处
    // img_dilate(model_input);

#if ENABLE_FILTER
    // 中值滤波：滤除单点噪声
    int8_t temp[MODEL_SIZE * MODEL_SIZE];
    memcpy(temp, model_input, sizeof(temp));
    uint8_t win[FILTER_DIM * FILTER_DIM];
    int edge = FILTER_DIM / 2;

    for (int y = edge; y < MODEL_SIZE - edge; y++) {
        for (int x = edge; x < MODEL_SIZE - edge; x++) {
            int count = 0;
            for (int fy = -edge; fy <= edge; fy++) {
                for (int fx = -edge; fx <= edge; fx++) {
                    win[count++] = (uint8_t)temp[(y + fy) * MODEL_SIZE + (x + fx)];
                }
            }
            simple_sort(win, FILTER_DIM * FILTER_DIM);
            model_input[y * MODEL_SIZE + x] = (int8_t)win[(FILTER_DIM * FILTER_DIM) / 2];
        }
    }
#endif
}

// ================= 最终封装接口 =================

void process_mnist_optimized(uint16_t *src_buf, int8_t *model_input) {
    if (!src_buf || !model_input) return;

    // 1. 预处理
    img_preprocess(src_buf);
    uint8_t *gray_data = (uint8_t *)src_buf;

    // 2. 自动计算阈值
    uint8_t thresh = img_get_otsu_threshold(gray_data);

    // 3. 寻找数字边界 (中心扫描)
    BBox box;
    if (img_get_bounding_box_center_out(gray_data, thresh, &box)) {
        // 4. 映射到 28x28
        img_final_map(gray_data, &box, thresh, model_input);

        // 5. 串口打印字符画 (调试用)
        #if 0
        printf("\n--- Detected Digital (BBox: %dx%d) ---\n", box.w, box.h);
        for (int y = 0; y < MODEL_SIZE; y++) {
            for (int x = 0; x < MODEL_SIZE; x++) {
                printf(model_input[y * MODEL_SIZE + x] > 0 ? "##" : "  ");
            }
            printf("\n");
        }
        #endif
    } else {
        printf("Digit not found in center area.\n");
        memset(model_input, -128, MODEL_SIZE * MODEL_SIZE);
    }
}