/********************************** (C) COPYRIGHT *******************************
 * File Name          : task.h
 * Author             : WCH
 * Version            : V1.0
 * Date               : 2023/10/9
 * Description        :
 * Copyright (c) 2021 Nanjing Qinheng Microelectronics Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

/******************************************************************************/
#ifndef __TASK_H
#define __TASK_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ch32h417.h"
#include <rtthread.h>
#include <rthw.h>
#include "lvgl.h"
#include "hal_config.h"
#include "qfl.h"

/*********************************************************************
 * CONSTANTS
 */

#define TASK_THREAD_NUM                 6

/*********************************************************************
 * TYPEDEFS
 */

 /* 定义任务序号 */
typedef enum
{
    kNone = 0,
    kTask1,
    kTask2,
    kTask3,
    kTask4,
    kTask5,
    kTask6,
    kTask01,
    kTask02,
    kTask03
}task_id_t;

/* 定义正在运行的任务的信息 */
typedef struct
{
    rt_thread_t running_thread[TASK_THREAD_NUM];
    uint8_t running_thread_num;
    task_id_t running_task;
}task_running_info_t;

// #define FFT_MEM_SIZE                    4608    //FFT算法堆栈,大小根据kiss_fft_alloc中memneeded变量的需求来设置
// #define FFT_SIGNAL_LEN                  512     // FFT输入信号的长度（必须是2的幂次方）
// /* 自定义复数结构体（替代 CMSIS 的 float32_t 复杂类型） */
// typedef struct {
//     float r;  // 实部
//     float i;  // 虚部
// } fft_complex_t;

// typedef struct
// {
//     uint16_t scope_adc_buf1[FFT_SIGNAL_LEN];        //ADC采集数据缓冲区1
//     uint16_t scope_adc_buf2[FFT_SIGNAL_LEN];        //ADC采集数据缓冲区2
//     uint16_t scope_vol_buf[FFT_SIGNAL_LEN];         //电压数据缓冲区
//     uint16_t scope_show_buf[FFT_SIGNAL_LEN];        //屏幕显示数据缓冲区
//     uint16_t waveform_output_buf1[FFT_SIGNAL_LEN];  //DAC输出同步信号数据缓冲区
//     uint16_t waveform_output_buf2[FFT_SIGNAL_LEN];  //DAC输出同步信号数据缓冲区

//     uint8_t fft_memory[FFT_MEM_SIZE];               //FFT算法堆栈区
//     kiss_fft_cpx fft_input[FFT_SIGNAL_LEN];         //FFT输入信号的复数形式
//     kiss_fft_cpx fft_output[FFT_SIGNAL_LEN];        //FFT输出信号的复数形式
//     float fft_amplitude[FFT_SIGNAL_LEN];            //FFT输出信号的幅度值
// }scope_data_buf_t;

typedef struct
{
    // __attribute__((aligned(4))) uint16_t mic_adc_buf1[MIC_ADC_BUF_LEN]; //存放MIC ADC采集的数据
    // __attribute__((aligned(4))) uint16_t mic_adc_buf2[MIC_ADC_BUF_LEN]; //存放MIC ADC采集的数据
    // uint16_t mic_show_buf[MIC_SHOW_BUF_LEN]; //存放MIC ADC采集的数据

    // __attribute__((aligned(4))) uint8_t fft_memory[FFT_MEM_SIZE];               //FFT算法堆栈区 //幅度值*2
    // __attribute__((aligned(4))) kiss_fft_cpx fft_input[FFT_SIGNAL_LEN];         //FFT输入信号的复数形式
    // __attribute__((aligned(4))) kiss_fft_cpx fft_output[FFT_SIGNAL_LEN];        //FFT输出信号的复数形式
    // float fft_amplitude[FFT_SIGNAL_LEN];            //FFT输出信号的幅度值
}mic_data_buf_t;

/* 定义共用体，用于需要大内存任务的使用 */
typedef union
{
    uint8_t figure_canvas_buf[LV_CANVAS_BUF_SIZE_TRUE_COLOR(60, 60)]; //60*60真彩色画布，7200B，用于绘制图形
    // mic_data_buf_t mic_data_buf;
    // scope_data_buf_t scope_data_buf; //示波器任务数据缓冲区
}task_union_mem_t;

/*********************************************************************
 * GLOBAL VARIABLES
 */

/* 中文字库声明 */
LV_FONT_DECLARE(chinese_font_8)

/* 图片声明 */
LV_IMG_DECLARE(panel_img)

/* LCD刷新互斥量 */
extern struct rt_mutex lcd_mutex;

/* 当前运行的任务线程 */
extern task_running_info_t task_running_info;

/* 需要大内存的任务使用的共用体 */
extern task_union_mem_t task_union_memory;

/* 创建对象用于包含后续各个任务的部件 */
extern lv_obj_t* task_container_obj;

/* 创建标签用于显示当前执行的任务 */
extern lv_obj_t* task_label;

/* 蜂鸣器当前的开关状态 */
extern uint8_t task2_beep_flag;

/* 蜂鸣器剩余鸣叫次数 */
extern uint8_t task2_beep_count;

extern lv_obj_t * task03_camera_img;
extern lv_obj_t * task03_processed_img;
extern lv_obj_t * ui_Label_Recog_Value;
extern lv_obj_t * task03_sys_state_label;
extern lv_obj_t * task03_recog_result_label;

extern volatile uint8_t task03_take_photo_flag;
extern volatile uint8_t task03_photo_recognize_flag;

extern volatile uint8_t task03_photo_finish_flag;


/* 返回按键 */
extern lv_obj_t * menu_back_btn;                           //切换回任务列表
extern lv_obj_t * menu_back_label;                         //按钮文本标签
extern void menu_back_handler(lv_event_t * e);
/*********************************************************************
 * FUNCTION
 */

/* 停止正在运行的任务 */
void rt_task_stop(void);

/* 任务主界面 */
long task0(void);

/* 任务1初始化函数 */
long task1(void);

/* 任务2初始化函数 */
long task2(void);

/* 任务3初始化函数 */
long task3(void);

/* 任务4初始化函数 */
long task4(void);

/* 任务5初始化函数 */
long task5(void);

/* 任务6初始化函数 */
long task6(void);

/* 任务03初始化函数 */
long task03(void);
/*********************************************************************
*********************************************************************/
void idtimer_delete(void);
#ifdef __cplusplus
}
#endif

#endif
