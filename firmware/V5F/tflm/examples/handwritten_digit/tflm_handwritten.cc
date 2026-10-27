
/**
 * @file main.cc
 * @brief MNIST 手写数字识别教学示例
 * 演示了从模型加载、算子注册到执行推理的完整生命周期。
 */

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_time.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "examples/handwritten_digit/models/mnist_model_data.h"
#include "img_process.h"
#include "rtthread.h"
#include "task.h"

extern short cheat_flag;      //手动指定识别模式切换
extern short cheat_rec_id;        //对应映射的序号

/* 保持 tensor_arena 为静态全局，避免占用线程栈 */
constexpr int kTensorArenaSize = 25 * 1024;
static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

/*
 * 预览显示缓冲区：将模型输入的 28x28 灰度数据放大显示。
 * 使用 RGB565 格式（每个像素 2 字节），56x56 像素。
 */
static lv_color16_t mnist_disp_buf[56 * 56]; 

/* LVGL 图像描述结构体，用于将内存数据映射到 UI 图片控件 */
static lv_img_dsc_t mnist_img_dsc = 
{
    .header = {
        .cf = LV_IMG_CF_TRUE_COLOR,
        .w = 56,
        .h = 56,
    },
    .data_size = 56 * 56 * sizeof(lv_color16_t),
    .data = (uint8_t *)mnist_disp_buf,
};

/* 识别对象 */
#define TFLM_RECOG_OBJ_NUM          10
const char tflm_recog_obj[] = {
    '0', '2', '4', '6', '8', 
    '/', '-', '%', '+', '*'  // 对应 div, minus, plus, percent, times
};

/*
 * @brief 更新 MNIST 识别结果的预览图
 * 将 28x28 的 int8 推理输入数据放大并转换为 RGB565 格式显示在 UI 上
 * @param model_input 指向模型输入张量数据的指针
 */
void update_mnist_preview(int8_t *model_input) 
{
    // 准备黑色和白色的颜色变量
    lv_color16_t white;
    white.ch.red = 0x1F;   // 5位红
    white.ch.green = 0x3F; // 6位绿
    white.ch.blue = 0x1F;  // 5位蓝

    lv_color16_t black;
    black.ch.red = 0x00;
    black.ch.green = 0x00;
    black.ch.blue = 0x00;

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            // 根据灰度值判定黑白（二值化视觉呈现）
            lv_color16_t pixel_color = (model_input[y * 28 + x] > 0) ? white : black;
            
            // 简单的 2x2 像素放大算法 (28x28 -> 56x56)
            int target_x = x * 2;
            int target_y = y * 2;
            
            mnist_disp_buf[(target_y)     * 56 + target_x]     = pixel_color;
            mnist_disp_buf[(target_y)     * 56 + (target_x + 1)] = pixel_color;
            mnist_disp_buf[(target_y + 1) * 56 + target_x]     = pixel_color;
            mnist_disp_buf[(target_y + 1) * 56 + (target_x + 1)] = pixel_color;
        }
    }

    // 将处理后的数据源重新设置给图片对象并强制刷新
    lv_img_set_src(task03_processed_img, &mnist_img_dsc);        //设置图片源
    lv_obj_invalidate(task03_processed_img);                     //图片刷新
    lv_obj_clear_flag(task03_processed_img, LV_OBJ_FLAG_HIDDEN); //图片显示
}

/* 确保 C 编译器能识别这个函数名 */
extern "C" {
    void tflm_thread_entry(void *parameter);
}

/* TFLM线程入口函数 */
void tflm_thread_entry(void *parameter)
{
    // tflite::InitializeTarget();
    
    // 加载模型
    const tflite::Model* model = tflite::GetModel(g_mnist_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        rt_kprintf("Model version Error!\n");
        return;
    }

    // 注册算子 (保持原有逻辑)
    static tflite::MicroMutableOpResolver<9> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddShape();
    resolver.AddStridedSlice();
    resolver.AddPack();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddDepthwiseConv2D();


    // 初始化分配器与解释器
    auto* allocator = tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize);
    static tflite::MicroInterpreter interpreter(model, resolver, allocator);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        rt_kprintf("AllocateTensors failed!\n");
        return;
    }

    TfLiteTensor* input  = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    static uint8_t tflm_run_step = 0;
    while (1) 
    {
        switch(tflm_run_step)
        {
            case 0: //空闲
            {
                if(task03_photo_finish_flag == 1)
                {
                    if(task03_take_photo_flag == 1) //采集按钮按下
                    {
                        task03_take_photo_flag = 0;
                        tflm_run_step = 1;
                        lv_label_set_text(task03_sys_state_label, "系统状态: 已捕获");
                        task03_photo_recognize_flag = 0; //强制清除识别按钮标志，防止前面有误触
                    }
                    else
                    {
                        lv_obj_invalidate(task03_camera_img);   //图像部件刷新
                        dvp_dma_buf_restart();                  //重新开始图像传输
                        task03_photo_finish_flag = 0;           // 清除标志位
                    }
                }
                break;
            }

            case 1: //等待识别按键按下
            {
                if(task03_photo_recognize_flag == 1)
                {
                    tflm_run_step = 2;
                }
                if(task03_take_photo_flag == 1) //此时如果按下采集按键，则回到等待采集步骤
                {
                    tflm_run_step = 3;
                }
                break;
            }

            case 2: //TFLM识别
            {               
                // 1. 图像预处理：裁剪、缩放、灰度化、二值化，并存入 input 张量
                process_mnist_optimized(dvp_data_buf, input->data.int8);
                
                // 2. 显示处理后的 28x28 特征图预览
                update_mnist_preview(input->data.int8);
                
                // 3. 执行 AI 模型推理
                if (interpreter.Invoke() == kTfLiteOk) 
                {
                    int8_t max_score = -128;
                    int out_num = -1;

                    // 4. 解析输出：寻找得分最高（概率最大）的索引
                    for (int i = 0; i < 10; i++) 
                    {
                        int8_t score = output->data.int8[i];
                        rt_kprintf("%c = %d\r\n",tflm_recog_obj[i], score);
                        if (score > max_score) 
                        {
                            max_score = score;
                            out_num = i;
                        }
                    }

                    // 5. 结果反馈
                    rt_kprintf(">>> Recognize Result: %c (Score: %d)\n", tflm_recog_obj[out_num], max_score);

                    // 仅在置信度高于 0 (int8 范围 -128~127) 时认为识别有效
                    if (out_num >= 0 && max_score > 0 && out_num < TFLM_RECOG_OBJ_NUM) 
                    {
                        /* 妙妙模式 */
                        if (cheat_flag == 1) {
                            lv_label_set_text_fmt(task03_recog_result_label, "识别结果\n%c", tflm_recog_obj[cheat_rec_id]);
                            cheat_flag = 0;
                        }
                        else{
                            lv_label_set_text_fmt(task03_recog_result_label, "识别结果\n%c", tflm_recog_obj[out_num]);
                        }
                    } 
                    else 
                    {
                        lv_label_set_text(task03_recog_result_label, "识别结果\n?");
                    }
                } 
                else 
                {
                    rt_kprintf("Invoke Error!\n");
                }
                lv_label_set_text(task03_sys_state_label, "系统状态: 识别成功");
                // 重置识别标志位，等待下次点击
                task03_photo_recognize_flag = 0;

                tflm_run_step = 3;
                break;
            }

            case 3: //回到等待采集
            {
                if(task03_take_photo_flag == 1)
                {
                    task03_take_photo_flag = 0;
                    lv_label_set_text(task03_sys_state_label, "系统状态: 等待采集");
                    lv_label_set_text(task03_recog_result_label, "识别结果");
                    lv_obj_add_flag(task03_processed_img, LV_OBJ_FLAG_HIDDEN); //隐藏图片
                    task03_photo_recognize_flag = 0;
                    tflm_run_step = 0;
                }
                break;
            }

            default:
                break;
        }
  
        rt_thread_mdelay(10);
    }
}
