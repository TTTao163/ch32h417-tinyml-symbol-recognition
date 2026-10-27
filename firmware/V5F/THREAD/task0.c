/******************************************************************************/
/* 头文件包含 */
#include "task.h"

static const char *btnm_map0[] = {
     "1", "2", "3",  "\n",  // 第一行
    "4", "5", "6",     // 第二行

    ""                           // 结束标记
};

static lv_style_t style_btn0;       // 按钮普通样式
static lv_style_t style_btn_pressed0;  // 按钮按下样式

static uint16_t btn_id00=70;

 void init_styles0() 
 {
    // 普通按钮样式
    lv_style_init(&style_btn0);
    lv_style_set_bg_color(&style_btn0, lv_color_hex(0xE0E0E0));  // 浅灰色背景
    lv_style_set_radius(&style_btn0, 5);  // 圆角5像素

    // 按下状态样式
    lv_style_init(&style_btn_pressed0);
    lv_style_set_bg_color(&style_btn_pressed0, lv_color_hex(0x0000ff));  // 蓝色
    lv_style_set_radius(&style_btn_pressed0, 5);
}

void calc_event_handler0(lv_event_t *e) 
{
    lv_obj_t *obj = lv_event_get_target(e);
   //lv_event_code_t code = lv_event_get_code(e);        //获取回调触发码
    uint16_t btn_id = lv_btnmatrix_get_selected_btn(obj);
    const char *txt = lv_btnmatrix_get_btn_text(obj, btn_id);

    if(btn_id==btn_id00)  //防抖
    {
        btn_id00=70;
        return;
    }
    else {
        btn_id00=btn_id;
    }

    // 根据按钮文本分发处理
    if(strcmp(txt, "1") == 0) 
    {
        task1();
    }
    else if(strcmp(txt, "2") == 0) 
    {
        // task2();
    }
    else if(strcmp(txt, "3") == 0) 
    {
        
        // task3();
    }
    else if(strcmp(txt, "4") == 0) 
    {
        
        // task4();
    }
    else if(strcmp(txt, "5") == 0) 
    {
        
        // task5();
    }
    else if(strcmp(txt, "6") == 0) 
    {
        
        // task6();
    }


}

/* Task_1 */
long task0(void)//回到主菜单
{
    /* 停止正在运行的任务 */
    rt_task_stop();

    /*保存将要运行的任务*/
    task_running_info.running_task = kNone;
   
    /* 清屏 */
    lv_obj_clean(task_container_obj);//清除父容器就黑屏
    hal_delay_ms(100);


    rt_mutex_take(&lcd_mutex, RT_WAITING_FOREVER); //获取互斥量，获取后LVGL线程将不允许执行，也就是屏幕会停止刷新

    /* 创建标签部件显示当前执行任务 */
    lv_obj_t* title_label = lv_label_create(task_container_obj);
    lv_obj_set_style_text_font(title_label, &chinese_font_8, LV_STATE_DEFAULT);    //设置字体
    lv_label_set_text(title_label, "任务");                                        //设置显示内容
    lv_obj_align(title_label, LV_ALIGN_TOP_LEFT, 0, 0);                            //设置部件对齐

    rt_mutex_release(&lcd_mutex); //释放互斥量，释放后LVGL线程将允许执行，屏幕将被刷新
    init_styles0();

     // 创建按钮矩阵
    lv_obj_t *btnm = lv_btnmatrix_create(task_container_obj);
    lv_btnmatrix_set_map(btnm, btnm_map0);
    lv_obj_set_size(btnm, 290, 150);
    lv_obj_align(btnm, LV_ALIGN_BOTTOM_MID, 0, 0);
    lv_obj_add_style(btnm, &style_btn0, LV_PART_ITEMS);
    lv_obj_add_style(btnm, &style_btn_pressed0, LV_PART_ITEMS | LV_STATE_PRESSED);

    lv_obj_add_event_cb(btnm, calc_event_handler0, LV_EVENT_VALUE_CHANGED, NULL);


    return 0;
}