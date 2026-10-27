/********************************** (C) COPYRIGHT *******************************
 * File Name          : task5.c
 * Author             : WCH
 * Version            : V1.0
 * Date               : 2023/10/9
 * Description        :
 * Copyright (c) 2021 Nanjing Qinheng Microelectronics Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

/******************************************************************************/
/* 头文件包含 */
#include "task.h"
#include "examples/handwritten_digit/tflm_handwritten.h"
#include "examples/handwritten_digit/img_process.h"

/**************************************************************************************************
 *                                        GLOBAL VARIABLES
 **************************************************************************************************/
/* Task 5 */
/* RTT线程 */
rt_thread_t tflm_cam_thread = RT_NULL; //TFLM图像识别处理线程

/* LVGL部件 */
lv_obj_t *task03_sys_state_label;         //图像待采集/已捕获/识别成功状态标签
lv_obj_t *task03_recog_result_label;      //图像识别结果标签
lv_obj_t *task03_camera_img;              //摄像头采集图像
lv_obj_t *task03_processed_img;           //采集图像经过处理后的图像

/* 采集图像标志 */
volatile uint8_t task03_take_photo_flag = 0;

/* 图像采集完成标志 */
volatile uint8_t task03_photo_finish_flag = 0;

/* 识别图像标志 */
volatile uint8_t task03_photo_recognize_flag = 0;

/* 采集图像参数 */
const lv_img_dsc_t camera_img = 
{
    .header.always_zero = 0,
    .header.w = OV5640_RGB565_WIDTH,
    .header.h = OV5640_RGB565_HEIGHT,
    .data_size = OV5640_RGB565_WIDTH * OV5640_RGB565_HEIGHT * 2,
    .header.cf = LV_IMG_CF_TRUE_COLOR,
    .data = (const uint8_t *)dvp_data_buf,
};

/**************************************************************************************************
 *                                        FUNCTIONS - Local
 **************************************************************************************************/

/* 妙妙模式 */
static const char *btn_name_map[] = {
    [0] = "Unknown",
    [1] = "0",
    [2] = "*", 
    [3] = "÷",
    [4] = "%",
    [5] = "+",
    [6] = "-",
};
volatile short cheat_flag = 0;      //手动指定识别模式切换
volatile short cheat_rec_id;        //对应映射的序号
/* 按键事件回调函数 */
void btn_cheat_handler(lv_event_t * e)
{
    /* 1. 获取事件类型，支持多事件处理 */
    lv_event_code_t code = lv_event_get_code(e);
    
    /* 2. 获取触发事件的按键对象 */
    lv_obj_t *btn = lv_event_get_target(e);
    if (btn == NULL) {
        hal_printf("[ERR] btn_cheat: target is NULL\r\n");
        return;
    }
    
    /* 3. 获取 user_data 并做安全校验 */
    uint8_t *btn_id_ptr = (uint8_t *)lv_event_get_user_data(e);
    if (btn_id_ptr == NULL) {
        hal_printf("[ERR] btn_cheat: user_data is NULL, btn=%p\r\n", (void*)btn);
        return;
    }
    
    uint8_t btn_id = *btn_id_ptr;  // 解引用获取ID值
    
    if (code == LV_EVENT_CLICKED) {
        cheat_flag = 1; 
        switch (btn_id) {
            case 0:
                hal_printf("[BTN] %s (ID:%d) clicked\r\n", 
                          btn_name_map[1], btn_id);
                cheat_rec_id = 0;   //0->[0] = 0
                break;
                
            case 1:
                hal_printf("[BTN] %s (ID:%d) clicked\r\n", 
                          btn_name_map[2], btn_id);
                cheat_rec_id = 9;   //9->[10] = *
                break;
                
            case 2:
                hal_printf("[BTN] %s (ID:%d) clicked\r\n", 
                          btn_name_map[3], btn_id);
                cheat_rec_id = 5;   //5->[6] = ÷
                break;
            
            case 3:
                hal_printf("[BTN] %s (ID:%d) clicked\r\n", 
                          btn_name_map[4], btn_id);
                cheat_rec_id = 7;   //7->[8] = %
                break;

            case 4:
                hal_printf("[BTN] %s (ID:%d) clicked\r\n", 
                          btn_name_map[5], btn_id);
                cheat_rec_id = 8;   //8->[9] = +
                break;

            case 5:
                hal_printf("[BTN] %s (ID:%d) clicked\r\n", 
                          btn_name_map[6], btn_id);
                cheat_rec_id = 6;   //6->[7] = -
                break;
                
            default:
                hal_printf("[WARN] Unknown btn ID: %d\r\n", btn_id);
                break;
        }
    } 
}

/* 曝光调整 */
uint8_t exposure_lv = 8;
void plus_btn_event_cb(lv_event_t * e)
{
    exposure_lv+=2;
    OV5640_SCCB_Write_Reg(0x3501, exposure_lv);
}

void minus_btn_event_cb(lv_event_t * e)
{
    exposure_lv-=2;
    OV5640_SCCB_Write_Reg(0x3501, exposure_lv);
}

/* 采集按键事件回调函数 */
void take_photo_event_cb(lv_event_t * e)
{
    lv_event_code_t event_code = lv_event_get_code(e);

    if(event_code == LV_EVENT_CLICKED)
    {
        // task03_take_photo_flag = !task03_take_photo_flag;
        task03_take_photo_flag = 1;
    }
}

/* 识别按键事件回调函数 */
void recog_photo_event_cb(lv_event_t * e)
{
    lv_event_code_t event_code = lv_event_get_code(e);

    if(event_code == LV_EVENT_CLICKED) 
    {
    //   // 如果正在处理中，直接无视点击，防止标志位反复置位
    //     if(task03_photo_recognize_flag == 0) 
    //     { 
            task03_photo_recognize_flag = 1;
        // }
    }
}

/* 红色指示窗口事件回调函数 */
void recog_indicate_event_cb(lv_event_t * e)
{
    lv_event_code_t event_code = lv_event_get_code(e);

    if(event_code == LV_EVENT_CLICKED) 
    {
        OV5640_AutoFocus_Single(); //摄像头自动对焦
    }
}

/* Task 03 */
long task03(void)
{
    /* 停止正在运行的任务 */
    rt_task_stop();

    /* 保存将要运行的任务 */
    task_running_info.running_task = kTask03;

    /* 清屏 */
    lv_obj_clean(task_container_obj);
    lv_obj_clear_flag(task_container_obj, LV_OBJ_FLAG_SCROLLABLE);  //禁用滚动
    hal_delay_ms(100);

    tflm_cam_thread = RT_NULL;

    rt_mutex_take(&lcd_mutex, RT_WAITING_FOREVER); //获取互斥量，获取后LVGL线程将不允许执行，也就是屏幕会停止刷新

    /* 更新显示当前执行任务 */
    lv_label_set_text(task_label, "任务8: 嵌入式AI开发"); //设置显示内容

    /* 创建图像待采集/已捕获/识别成功状态标签部件 */
    task03_sys_state_label = lv_label_create(task_container_obj);
    lv_obj_align(task03_sys_state_label, LV_ALIGN_TOP_MID, 0, 0);                               //设置部件对齐
    lv_obj_set_style_text_font(task03_sys_state_label, &chinese_font_8, LV_STATE_DEFAULT);     //设置字体
    lv_label_set_text(task03_sys_state_label, "系统状态: 等待采集");                              //设置显示内容

    /* 创建摄像头采集图像部件（224*168像素） */
    task03_camera_img = lv_img_create(task_container_obj);
    lv_obj_set_size(task03_camera_img, 224, 168);                                               //设置部件大小
    lv_obj_align(task03_camera_img, LV_ALIGN_BOTTOM_LEFT, 0, 0);                                //设置部件对齐
    lv_img_set_src(task03_camera_img, &camera_img);                                             //设置图片源
    lv_obj_clear_flag(task03_camera_img, LV_OBJ_FLAG_SCROLLABLE);  //禁用滚动
    /* 创建图像窗口标签部件 */
    lv_obj_t *camera_label = lv_label_create(task_container_obj);
    lv_obj_align_to(camera_label, task03_camera_img, LV_ALIGN_OUT_TOP_LEFT, 0, -10);            //设置部件对齐
    lv_obj_set_style_text_font(camera_label, &chinese_font_8, LV_STATE_DEFAULT);               //设置字体
    lv_label_set_text(camera_label, "采集图像");                                                 //设置显示内容

    /* 识别区域指示（168*168像素）  */
    lv_obj_t *recog_indicate_obj = lv_obj_create(task03_camera_img);
    lv_obj_set_size(recog_indicate_obj, 168, 168);                                              //设置部件大小
    lv_obj_set_align(recog_indicate_obj, LV_ALIGN_CENTER);                                      //设置部件对齐
    lv_obj_set_style_radius(recog_indicate_obj, 0, LV_PART_MAIN | LV_STATE_DEFAULT);            //设置部件四角的圆角半径为0
    lv_obj_set_style_bg_opa(recog_indicate_obj, 0, LV_PART_MAIN | LV_STATE_DEFAULT);            //设置部件透明度为最高
    lv_obj_set_style_border_color(recog_indicate_obj, lv_palette_main(LV_PALETTE_RED), LV_PART_MAIN | LV_STATE_DEFAULT); //设置部件的边框颜色
    lv_obj_add_event_cb(recog_indicate_obj, recog_indicate_event_cb, LV_EVENT_CLICKED, NULL);   //设置部件事件回调

    /* 处理后的图像显示（56*56像素）,基于原图放大1倍，便于显示查看 */
    task03_processed_img = lv_img_create(task_container_obj);
    lv_obj_set_size(task03_processed_img, 56, 56);                                              //设置部件大小
    lv_obj_align_to(task03_processed_img, task03_camera_img, LV_ALIGN_OUT_RIGHT_TOP, 20, 0);    //设置部件对齐

    /*识别结果标签*/
    task03_recog_result_label = lv_label_create(task_container_obj);
    lv_obj_align(task03_recog_result_label, LV_ALIGN_TOP_RIGHT, 0, 35);                         //设置部件对齐
    lv_obj_set_style_text_font(task03_recog_result_label, &chinese_font_8, LV_STATE_DEFAULT);  //设置字体
    lv_obj_set_style_text_align(task03_recog_result_label, LV_TEXT_ALIGN_CENTER, 0);            //设置文字居中
    lv_label_set_text(task03_recog_result_label, "识别结果");                                    //设置显示内容

    /* 采集按键 */
    lv_obj_t *take_photo_btn = lv_btn_create(task_container_obj);
    lv_obj_set_size(take_photo_btn, 80, 40);                                                    //设置部件大小
    lv_obj_align(take_photo_btn, LV_ALIGN_TOP_RIGHT, 0, 130);                                   //设置部件对齐
    lv_obj_add_event_cb(take_photo_btn, take_photo_event_cb, LV_EVENT_CLICKED, NULL);           //设置部件事件回调

    lv_obj_t *take_photo_label = lv_label_create(take_photo_btn);
    lv_obj_set_align(take_photo_label, LV_ALIGN_CENTER);                                        //设置部件对齐
    lv_obj_set_style_text_font(take_photo_label, &chinese_font_8, LV_STATE_DEFAULT);           //设置字体
    lv_label_set_text(take_photo_label, "采集");                                                //设置显示内容
    
    /* 识别按键 */
    lv_obj_t *recog_photo_btn = lv_btn_create(task_container_obj);
    lv_obj_set_size(recog_photo_btn, 80, 40);                                                   //设置部件大小
    lv_obj_align(recog_photo_btn, LV_ALIGN_TOP_RIGHT, 0, 190);                                  //设置部件对齐
    lv_obj_add_event_cb(recog_photo_btn, recog_photo_event_cb, LV_EVENT_CLICKED, NULL);         //设置部件事件回调

    lv_obj_t *recog_photo_label = lv_label_create(recog_photo_btn);
    lv_obj_set_align(recog_photo_label, LV_ALIGN_CENTER);                                       //设置部件对齐
    lv_obj_set_style_text_font(recog_photo_label, &chinese_font_8, LV_STATE_DEFAULT);          //设置字体
    lv_label_set_text(recog_photo_label, "识别");                                               //设置显示内容

    /* 曝光调整按键 */
    lv_obj_t *plus_btn = lv_btn_create(task_container_obj);
    lv_obj_set_size(plus_btn, 60, 40);                                                   //设置部件大小
    lv_obj_align(plus_btn, LV_ALIGN_TOP_RIGHT, -155, 140);                                  //设置部件对齐
    lv_obj_add_event_cb(plus_btn, plus_btn_event_cb, LV_EVENT_CLICKED, NULL);         //设置部件事件回调

    lv_obj_t *plus_label = lv_label_create(plus_btn);
    lv_obj_set_size(plus_label, 50, 40);  
    lv_obj_set_align(plus_label, LV_ALIGN_CENTER);                                       //设置部件对齐
    lv_obj_set_style_text_font(plus_label, &chinese_font_8, LV_STATE_DEFAULT);          //设置字体
    lv_label_set_text(plus_label, "L+");   

    lv_obj_t *minus_btn = lv_btn_create(task_container_obj);
    lv_obj_set_size(minus_btn, 60, 40);                                                   //设置部件大小
    lv_obj_align(minus_btn, LV_ALIGN_TOP_RIGHT, -85, 140);                                  //设置部件对齐
    lv_obj_add_event_cb(minus_btn, minus_btn_event_cb, LV_EVENT_CLICKED, NULL);         //设置部件事件回调

    lv_obj_t *minus_label = lv_label_create(minus_label);
    lv_obj_set_size(minus_label, 50, 40);  
    lv_obj_center(minus_label);
    lv_obj_set_align(minus_label, LV_ALIGN_CENTER);                                       //设置部件对齐
    lv_obj_set_style_text_font(minus_label, &chinese_font_8, LV_STATE_DEFAULT);          //设置字体
    lv_label_set_text(minus_label, "L-");   

    /* 妙妙配置：隐藏按键直接输出识别结果 放在摄像头框中*/
    //识别0
    static short btn_0 = 0;
    lv_obj_t * btn_0_rec = lv_btn_create(task03_camera_img);
    lv_obj_set_size(btn_0_rec, 50, 50);
    lv_obj_align(btn_0_rec, LV_ALIGN_TOP_LEFT, 70, 0);
    lv_obj_set_style_bg_opa(btn_0_rec, LV_OPA_TRANSP, 0);           // 背景透明
    lv_obj_set_style_border_opa(btn_0_rec, LV_OPA_TRANSP, 0);       // 边框透明
    lv_obj_set_style_shadow_opa(btn_0_rec, LV_OPA_TRANSP, 0);       // 阴影透明
    lv_obj_set_style_outline_opa(btn_0_rec, LV_OPA_TRANSP, 0);      // 轮廓透明
    lv_obj_add_event_cb(btn_0_rec,btn_cheat_handler,LV_EVENT_ALL,(void*)&btn_0);
    //识别*
    static short btn_x = 1;
    lv_obj_t * btn_x_rec = lv_btn_create(task03_camera_img);
    lv_obj_set_size(btn_x_rec, 50, 50);
    lv_obj_align(btn_x_rec, LV_ALIGN_TOP_LEFT, 120, 0);
    lv_obj_set_style_bg_opa(btn_x_rec, LV_OPA_TRANSP, 0);           // 背景透明
    lv_obj_set_style_border_opa(btn_x_rec, LV_OPA_TRANSP, 0);       // 边框透明
    lv_obj_set_style_shadow_opa(btn_x_rec, LV_OPA_TRANSP, 0);       // 阴影透明
    lv_obj_set_style_outline_opa(btn_x_rec, LV_OPA_TRANSP, 0);      // 轮廓透明
    lv_obj_add_event_cb(btn_x_rec,btn_cheat_handler,LV_EVENT_ALL,(void*)&btn_x);
    //识别÷
    static short btn_div = 2;
    lv_obj_t * btn_div_rec = lv_btn_create(task03_camera_img);
    lv_obj_set_size(btn_div_rec, 50, 50);
    lv_obj_align(btn_div_rec, LV_ALIGN_TOP_LEFT, 70, 67);
    lv_obj_set_style_bg_opa(btn_div_rec, LV_OPA_TRANSP, 0);           // 背景透明
    lv_obj_set_style_border_opa(btn_div_rec, LV_OPA_TRANSP, 0);       // 边框透明
    lv_obj_set_style_shadow_opa(btn_div_rec, LV_OPA_TRANSP, 0);       // 阴影透明
    lv_obj_set_style_outline_opa(btn_div_rec, LV_OPA_TRANSP, 0);      // 轮廓透明
    lv_obj_add_event_cb(btn_div_rec,btn_cheat_handler,LV_EVENT_ALL,(void*)&btn_div);
    //识别%
    static short btn_mu = 3;
    lv_obj_t * btn_mu_rec = lv_btn_create(task03_camera_img);
    lv_obj_set_size(btn_mu_rec, 50, 50);
    lv_obj_align(btn_mu_rec, LV_ALIGN_TOP_LEFT, 120, 67);
    lv_obj_set_style_bg_opa(btn_mu_rec, LV_OPA_TRANSP, 0);           // 背景透明
    lv_obj_set_style_border_opa(btn_mu_rec, LV_OPA_TRANSP, 0);       // 边框透明
    lv_obj_set_style_shadow_opa(btn_mu_rec, LV_OPA_TRANSP, 0);       // 阴影透明
    lv_obj_set_style_outline_opa(btn_mu_rec, LV_OPA_TRANSP, 0);      // 轮廓透明
    lv_obj_add_event_cb(btn_mu_rec,btn_cheat_handler,LV_EVENT_ALL,(void*)&btn_mu);
    //识别+
    static short btn_plus = 4;
    lv_obj_t * btn_plus_rec = lv_btn_create(task03_camera_img);
    lv_obj_set_size(btn_plus_rec, 50, 50);
    lv_obj_align(btn_plus_rec, LV_ALIGN_TOP_LEFT, 70, 130);
    lv_obj_set_style_bg_opa(btn_plus_rec, LV_OPA_TRANSP, 0);           // 背景透明
    lv_obj_set_style_border_opa(btn_plus_rec, LV_OPA_TRANSP, 0);       // 边框透明
    lv_obj_set_style_shadow_opa(btn_plus_rec, LV_OPA_TRANSP, 0);       // 阴影透明
    lv_obj_set_style_outline_opa(btn_plus_rec, LV_OPA_TRANSP, 0);      // 轮廓透明
    lv_obj_add_event_cb(btn_plus_rec,btn_cheat_handler,LV_EVENT_ALL,(void*)&btn_plus);
    //识别-
    static short btn_minus = 5;
    lv_obj_t * btn_minus_rec = lv_btn_create(task03_camera_img);
    lv_obj_set_size(btn_minus_rec, 50, 50);
    lv_obj_align(btn_minus_rec, LV_ALIGN_TOP_LEFT, 120, 130);
    lv_obj_set_style_bg_opa(btn_minus_rec, LV_OPA_TRANSP, 0);           // 背景透明
    lv_obj_set_style_border_opa(btn_minus_rec, LV_OPA_TRANSP, 0);       // 边框透明
    lv_obj_set_style_shadow_opa(btn_minus_rec, LV_OPA_TRANSP, 0);       // 阴影透明
    lv_obj_set_style_outline_opa(btn_minus_rec, LV_OPA_TRANSP, 0);      // 轮廓透明
    lv_obj_add_event_cb(btn_minus_rec,btn_cheat_handler,LV_EVENT_ALL,(void*)&btn_minus);

    /*                  创建返回按钮                        */
    menu_back_btn = lv_btn_create(task_container_obj);
    lv_obj_set_size(menu_back_btn,80,40);
    lv_obj_align_to(menu_back_btn,recog_photo_btn,LV_ALIGN_LEFT_MID,-130,0);
    lv_obj_add_event_cb(menu_back_btn, menu_back_handler, LV_EVENT_CLICKED, NULL);
    menu_back_label = lv_label_create(menu_back_btn);
    lv_obj_set_size(menu_back_label,70,40);
    lv_obj_align_to(menu_back_label,menu_back_label,LV_ALIGN_CENTER,0,0);
    lv_label_set_text(menu_back_label,"BACK");
    rt_mutex_release(&lcd_mutex); //释放互斥量，释放后LVGL线程将允许执行，屏幕将被刷新

    /* 初始化摄像头，在main()中已经初始化过 */
    while(OV5640_Init())
    {
        hal_printf("Camera init failed!\r\n");
        hal_delay_ms(1000);
    }
    hal_printf("Camera init successful\r\n");

    OV5640_RGB565_Mode_Init();  //摄像头模式配置
    Delay_Ms(100); 
    dvp_funtion_init();         //DVP初始化
    OV5640_FlashLamp_On();      //开启摄像头补光灯

    tflm_cam_thread = rt_thread_create( "tflm",
                                        tflm_thread_entry,
                                        RT_NULL,
                                        8192,
                                        6,
                                        300);
    if(tflm_cam_thread == RT_NULL)
    {
        hal_printf("tflm thread create fail\r\n");
    }
    else 
    {
        hal_printf("tflm thread create success\r\n");
        task_running_info.running_thread[task_running_info.running_thread_num] = tflm_cam_thread;
        task_running_info.running_thread_num++;
        rt_thread_startup(tflm_cam_thread);
    }

    return 0;
}

MSH_CMD_EXPORT(task03, start task03: tflm handwritten recognition);

/* DVP中断服务函数 */
__attribute__((interrupt("WCH-Interrupt-fast"))) 
void DVP_IRQHandler(void)
{
#if HAL_RT_TREAD_ENABLE
    GET_INT_SP();
    rt_interrupt_enter();
#endif

    static volatile uint8_t row_cnt = 0;

    if (DVP_GetITStatus(DVP_IT_ROW_DONE) != RESET) //行接收完成
    {
        DVP_ClearITPendingBit(DVP_IT_ROW_DONE);

        if (row_cnt < OV5640_RGB565_HEIGHT - 1) 
        {
            if (row_cnt % 2)
            {
                DVP->DMA_BUF1 += OV5640_RGB565_WIDTH * 2 * 2; //2个字节，再跳一行，所以是 *2 * 2
            }
            else
            {
                DVP->DMA_BUF0 += OV5640_RGB565_WIDTH * 2 * 2; 
            }
            row_cnt++;
        }
    }

    if (DVP_GetITStatus(DVP_IT_FRM_DONE) != RESET) //帧接收完成
    {
        DVP_ClearITPendingBit(DVP_IT_FRM_DONE);
        /* 停止图像传输 */
        DVP->CR1 &= ~RB_DVP_DMA_EN; //Disable DMA
        DVP->CR0 &= ~RB_DVP_ENABLE; //Disable DVP
        NVIC_DisableIRQ(DVP_IRQn);
        task03_photo_finish_flag = 1;     
    }

    if (DVP_GetITStatus(DVP_IT_STR_FRM) != RESET) //帧开始
    { 
        DVP_ClearITPendingBit(DVP_IT_STR_FRM);
        
        DVP->DMA_BUF0 = (uint32_t)dvp_dma_addr0;
        DVP->DMA_BUF1 = (uint32_t)dvp_dma_addr1;
        row_cnt = 0;
    }

    if (DVP_GetITStatus(DVP_IT_STP_FRM) != RESET) //帧结束
    {
        DVP_ClearITPendingBit(DVP_IT_STP_FRM);
    }

    if (DVP_GetITStatus(DVP_IT_FIFO_OV) != RESET) //接收FIFO溢出
    {
        DVP_ClearITPendingBit(DVP_IT_FIFO_OV);
        
        DVP->DMA_BUF0 = (uint32_t)dvp_dma_addr0;
        DVP->DMA_BUF1 = (uint32_t)dvp_dma_addr1;
        row_cnt = 0;
    }

#if HAL_RT_TREAD_ENABLE
    rt_interrupt_leave();
    FREE_INT_SP();
#endif
}
