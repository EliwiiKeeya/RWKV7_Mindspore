#include "kernel_operator.h"

extern "C" __global__ __aicore__ void wkv7_custom(GM_ADDR s, GM_ADDR r, GM_ADDR w, GM_ADDR k, GM_ADDR v, GM_ADDR a, GM_ADDR b, GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}