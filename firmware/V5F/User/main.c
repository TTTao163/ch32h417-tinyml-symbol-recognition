/********************************** (C) COPYRIGHT *******************************
 * File Name          : main.c
 * Author             : WCH
 * Version            : V1.0.0
 * Date               : 2025/03/01
 * Description        : Main program body.
 *********************************************************************************
 * Copyright (c) 2025 Nanjing Qinheng Microelectronics Co., Ltd.
 * Attention: This software (modified or not) and binary are used for
 * microcontroller manufactured by Nanjing Qinheng Microelectronics.
 *******************************************************************************/
#include "debug.h"
#include "hal_config.h"
#include "rtthread.h"
#include "rthw.h"
#include "lvgl.h"
#include "lv_port_disp_template.h"
#include "lv_port_indev_template.h"
#include "task.h"
#include "qfl.h"

uint8_t task1_pending;
lv_obj_t * menu_back_btn;                           //切换回任务列表
lv_obj_t * menu_back_label;                         //按钮文本标签
void menu_back_handler(lv_event_t * e){
    task0();
    OV5640_FlashLamp_Off();    
    hal_printf("back to menu\r\n");
}

/********************************** Global typedef ********************************************/

/*********************************** Global define ********************************************/

/********************************** Global Variable *******************************************/
/* LVGL线程 */
struct rt_thread lvgl_thread;       //LVGL线程结构体
__attribute__((aligned(4))) uint8_t lvgl_thread_stack[2560];    //LVGL线程的栈，如果空间不够使用可以适当放大，可以通过shell命令查看线程占用情况

/* 按键扫描线程 */
struct rt_thread key_thread;        //按键扫描线程结构体
__attribute__((aligned(4))) uint8_t key_thread_stack[1536];    //按键扫描线程的栈，如果空间不够使用可以适当放大，可以通过shell命令查看线程占用情况


/* LCD刷新互斥量 */
struct rt_mutex lcd_mutex;          //LCD刷新互斥量结构体

/* 当前运行的任务线程 */
task_running_info_t task_running_info = {0};

/* 需要大内存的任务使用的共用体 */
__attribute__((aligned(4)))  task_union_mem_t task_union_memory;

/* LVGL部件 */
lv_obj_t* task_container_obj;       //创建用于包含后续各个任务部件的容器
lv_obj_t* task_label;               //创建用于显示当前执行任务的标签

/*LVGL任务按钮部件*/
lv_obj_t *task_btn1;
lv_obj_t *task_btn2;
lv_obj_t *task_btn3;
lv_obj_t *task_btn4;
lv_obj_t *task_btn5;
lv_obj_t *task_btn6;
uint8_t key_flag = 0;
uint8_t timing_flag = 0;
uint32_t timing_start_time;
    const static uint8_t rgb_mode_buf0[] = {
     0x00, 0x00, 0x00, // 灯1：红
     0x00, 0x00, 0x00, //橙色
     0x00, 0x00, 0x00,//黄
     0x00, 0x00, 0x00,//绿
     0x00, 0x00, 0x00,//蓝
     0x00, 0x00, 0x00,//紫色
    };
/************************************* Function ***********************************************/
/*********************************************************************
 * @fn      rt_task_stop
 *
 * @brief   停止正在运行的任务
 *
 * @return  none
 */
void rt_task_stop(void)
{
    // while(task_running_info.running_thread_num > 0)
    //         {
    //             rt_thread_delete(task_running_info.running_thread[task_running_info.running_thread_num - 1]); //删除当前执行的线程
    //             --task_running_info.running_thread_num;
    //             hal_printf(" Task[%d] thread delete\r\n", task_running_info.running_task);
    //         }

    switch(task_running_info.running_task)
    {
        
        // case kTask1:

        case kTask2:
            while(task_running_info.running_thread_num > 0)
            {
                rt_thread_delete(task_running_info.running_thread[task_running_info.running_thread_num - 1]); //删除当前执行的线程
                --task_running_info.running_thread_num;
                hal_printf(" Task[%d] thread delete\r\n", task_running_info.running_task);
            }

            led_pwm_control(kALL_LED, LED_CONTROL_TIM_PERIOD_1MS, 0);
            beep_pwm_control(BEEP_CONTROL_TIM_PERIOD_1MS, 0);
            TIM_Cmd(TIM3, DISABLE); 
            TIM_Cmd(TIM9, DISABLE); 
            pioc_2812b_short_change(rgb_mode_buf0, sizeof(rgb_mode_buf0), PIOC_IO_CHANNEL);
            break;

        case kTask3:
            while(task_running_info.running_thread_num > 0)
            {
                rt_thread_delete(task_running_info.running_thread[task_running_info.running_thread_num - 1]); //删除当前执行的线程
                --task_running_info.running_thread_num;
                hal_printf(" Task[%d] thread delete\r\n", task_running_info.running_task);
            }
            led_pwm_control(kALL_LED, LED_CONTROL_TIM_PERIOD_1MS, 0);
            TIM_Cmd(TIM3, DISABLE); 
            break;

        case kTask4:
            while(task_running_info.running_thread_num > 0)
            {
                rt_thread_delete(task_running_info.running_thread[task_running_info.running_thread_num - 1]); //删除当前执行的线程
                --task_running_info.running_thread_num;
                hal_printf(" Task[%d] thread delete\r\n", task_running_info.running_task);
            }
            qfl_voice_deinit();
            break;

        case kTask5:
            while(task_running_info.running_thread_num > 0)
            {
                rt_thread_delete(task_running_info.running_thread[task_running_info.running_thread_num - 1]); //删除当前执行的线程
                --task_running_info.running_thread_num;
                hal_printf(" Task[%d] thread delete\r\n", task_running_info.running_task);
            }

            break;

        case kTask6:
            while(task_running_info.running_thread_num > 0)
            {
                rt_thread_delete(task_running_info.running_thread[task_running_info.running_thread_num - 1]); //删除当前执行的线程
                --task_running_info.running_thread_num;
                hal_printf(" Task[%d] thread delete\r\n", task_running_info.running_task);
            }
            
            break;

        case kTask01:   
            while(task_running_info.running_thread_num > 0)
            {
                rt_thread_delete(task_running_info.running_thread[task_running_info.running_thread_num - 1]); //删除当前执行的线程
                --task_running_info.running_thread_num;
                hal_printf(" Task[%d] thread delete\r\n", task_running_info.running_task);
            }
            break;

        case kTask02:
            while(task_running_info.running_thread_num > 0)
            {
                rt_thread_delete(task_running_info.running_thread[task_running_info.running_thread_num - 1]); //删除当前执行的线程
                --task_running_info.running_thread_num;
                hal_printf(" Task[%d] thread delete\r\n", task_running_info.running_task);
            }
            break;

        case kTask03:
            while(task_running_info.running_thread_num > 0)
            {
                rt_thread_delete(task_running_info.running_thread[task_running_info.running_thread_num - 1]); //删除当前执行的线程
                --task_running_info.running_thread_num;
                hal_printf(" Task[%d] thread delete\r\n", task_running_info.running_task);
            }
        
            break;
            
        default:
            break;
    }

    task_running_info.running_task = kNone;
    hal_delay_ms(50); //增加延时让系统进入idle线程来释放上文删除的线程的空间
}

/*********************************************************************
 * @fn      lvgl_thread_entry
 *
 * @brief   LVGL线程的入口函数
 *
 * @return  none
 */
void lvgl_thread_entry(void *parameter)
{
    GT911_init();           //触摸模块初始化
    LCD_Init();             //LCD初始化
    lv_init();              //LVGL库初始化
    lv_port_disp_init();    //LVGL显示初始化
    lv_port_indev_init();   //LVGL输入初始化

    /* 创建标签部件显示主题 */
    lv_obj_t* title_label = lv_label_create(lv_scr_act());
    lv_obj_set_style_text_font(title_label, &chinese_font_8, LV_STATE_DEFAULT);        //设置字体
    lv_label_set_text(title_label, "嵌入式系统应用开发赛项模块一驱动任务开发");             //设置显示内容
    lv_obj_align(title_label, LV_ALIGN_TOP_MID, 0, 0);                                  //设置部件对

    /* 创建容器部件用于包含后续各个任务的部件 */
    task_container_obj = lv_obj_create(lv_scr_act());
    lv_obj_set_size(task_container_obj, 480, 265);                                      //设置部件大小
    lv_obj_align(task_container_obj, LV_ALIGN_BOTTOM_MID, 0, 0);                        //设置部件对齐
    lv_obj_set_scrollbar_mode(task_container_obj, LV_SCROLLBAR_MODE_OFF);  // 关闭滚动条
    lv_obj_clear_flag(task_container_obj, LV_OBJ_FLAG_SCROLLABLE);         // 清除可滚动标志   
        
    task_label = lv_label_create(lv_scr_act());//任务标签
    lv_label_set_text(task_label, "  ");
    lv_obj_align(task_label, LV_ALIGN_TOP_MID, 28, 28);
    lv_obj_set_style_text_font(task_label, &chinese_font_8, LV_STATE_DEFAULT);
        
    task0();    /////////
   

    while(1)
    {
    
        rt_mutex_take(&lcd_mutex, RT_WAITING_FOREVER);  //获取互斥量
        lv_timer_handler();                             //LVGL事件处理函数
        rt_mutex_release(&lcd_mutex);                   //释放互斥量
        hal_delay_ms(10);
    }
}

/*********************************************************************
 * @fn      key_process
 *
 * @brief   按键处理
 *
 * @return  none
 */
void key_process(uint8_t keys)
{
    if(keys & KEY_BUTTON_1)
    {        
        hal_printf(" KEY1 is pressed\r\n");

    }

    else if(keys & KEY_BUTTON_2)
    {
        hal_printf(" KEY2 is pressed\r\n");
      
    }
    else if(keys & KEY_BUTTON_3)
    {
        hal_printf(" KEY3 is pressed\r\n");
             
    }
    else if(keys & KEY_BUTTON_4)
    {
        hal_printf(" KEY4 is pressed\r\n");
        rt_task_stop();
        OV5640_FlashLamp_Off();      //开启摄像头补光灯
        task0();
    }
}

/*********************************************************************
 * @fn      key_thread_entry
 *
 * @brief   按键扫描线程的入口函数
 *
 * @return  none
 */
void key_thread_entry(void *parameter)
{
    uint8_t keys;

    key_gpio_init(); //GPIO获取按键状态初始化

    while(1)
    {
        keys = key_polling();   //按键扫描
        if(keys != 0)
        {
            key_process(keys);      //按键处理
        }
        hal_delay_ms(20);
    }
}

/*********************************************************************
 * @fn      main
 *
 * @brief   Main program.
 *
 * @return  none
 */
int main(void)
{
    rt_err_t status;

    hal_printf("V5F SystemCoreClock: %d\r\n",SystemCoreClock);
    hal_printf("HCLKClock: %d\r\n", HCLKClock);
    hal_printf("ChipID: %08x\r\n", DBGMCU_GetCHIPID());

    /* 数码管驱动初始化，清空显示 */
    ch422_init();

    /* 2812B灯初始化，清空显示 */
    pioc_2812b_init();

    /* 初始化RTC */
    RTC_Init();

    /* 初始化LCD刷新互斥量，用于协调LVGL刷新屏幕的时间点 */
    status = rt_mutex_init( &lcd_mutex,                    //互斥量结构体地址
                            "mlcd",                        //互斥量名字
                            RT_IPC_FLAG_PRIO);             //互斥量标志，该标志不起作用
    if(status == (-RT_ERROR))
    {
        hal_printf(" lcd mutex create fail\r\n");
    }
    else
    {
        hal_printf(" lcd mutex create success\r\n");
    }

    status = rt_thread_init( &lvgl_thread,              //线程结构体地址
                             "lvgl",                    //线程名字
                             lvgl_thread_entry,         //线程入口函数
                             RT_NULL,                      //线程入口函数的参数为空
                             lvgl_thread_stack,         //线程栈的地址
                             sizeof(lvgl_thread_stack), //线程栈的大小
                             5,
                             1000);
    if(status == (-RT_ERROR))
    {
        hal_printf(" lvgl thread create fail\r\n");
    }
    else
    {
        hal_printf(" lvgl thread create success\r\n");
        rt_thread_startup(&lvgl_thread);                    //启动线程
    }

    /* 初始化按键扫描线程 */
    status = rt_thread_init( &key_thread,                  //线程结构体地址ַ
                             "tkey",                       //线程名字
                              key_thread_entry,            //线程入口地址
                              RT_NULL,                        //线程入口函数的参数为空
                              key_thread_stack,            //线程栈的地址
                              sizeof(key_thread_stack),    //线程栈的大小
                              6,
                              20);
    if(status == (-RT_ERROR))
    {
        hal_printf(" key thread create fail\r\n");
    }
    else
    {
        hal_printf(" key thread create success\r\n");
        rt_thread_startup(&key_thread);                    //启动线程
    }

    return 0; //main()是"main"线程中调用的一个函数，返回后，该线程将被删除
}
