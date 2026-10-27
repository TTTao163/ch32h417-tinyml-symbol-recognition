#ifndef _IMG_PROCESS_H_
#define _IMG_PROCESS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <string.h>


// 快速灰度化宏 (RGB565 -> Gray 0-255)
#define GET_GRAY(p) (( ((p >> 8) & 0xF8) * 30 + ((p >> 3) & 0xFC) * 59 + ((p << 3) & 0xF8) * 11 ) / 100)

void process_mnist_optimized(uint16_t *src_buf, int8_t *model_input);


#ifdef __cplusplus
} /*extern "C"*/
#endif



#endif /* end of img_process.h */
