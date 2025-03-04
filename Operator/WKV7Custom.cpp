/****************************************************************************************************
 * File				: WKV7Custom.cpp
 * Date				: 2025-03-03 23:49:29
 * Author			: Eliwii_Keeya
 * Description		: wkv7自定义算子调用应用程序源文件
 * Last Modified	: 2025-03-03 23:49:29
 ****************************************************************************************************/
#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void wkv7_custom_do(uint32_t blockDim, void* l2ctrl, void *stream, uint8_t *x, uint8_t *y, uint8_t *z);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void add_customwkv7_custom(GM_ADDR s, GM_ADDR r, GM_ADDR w, GM_ADDR k, GM_ADDR v, GM_ADDR a, GM_ADDR kk, GM_ADDR x, GM_ADDR s_ref);
#endif

int32_t main(int32_t argc, char *argv[])
{
    uint32_t blockDim = 8;
    size_t inputByteSize = 8 * 2048 * sizeof(uint16_t);
    size_t outputByteSize = 8 * 2048 * sizeof(uint16_t);

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *s = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *r = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *w = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *k = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *v = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *a = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *kk = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *s_ref = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    
    ReadFile("./input/input_r.bin", inputByteSize, r, inputByteSize);
    ReadFile("./input/input_w.bin", inputByteSize, w, inputByteSize);
    ReadFile("./input/input_k.bin", inputByteSize, k, inputByteSize);
    ReadFile("./input/input_v.bin", inputByteSize, v, inputByteSize);
    ReadFile("./input/input_a.bin", inputByteSize, a, inputByteSize);
    ReadFile("./input/input_kk.bin", inputByteSize, kk, inputByteSize);
    ReadFile("./input/input_x.bin", inputByteSize, x, outputByteSize);
    ReadFile("./input/input_s_ref.bin", inputByteSize, s_ref, outputByteSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(wkv7_custom, blockDim, r, w, k, v, a, kk, x, s_ref); // use this macro for cpu debug

    WriteFile("./output/output_x.bin", x, outputByteSize);
    WriteFile("./output/output_s_ref.bin", s_ref, outputByteSize);

    AscendC::GmFree((void *)r);
    AscendC::GmFree((void *)w);
    AscendC::GmFree((void *)k);
    AscendC::GmFree((void *)v);
    AscendC::GmFree((void *)a);
    AscendC::GmFree((void *)kk);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)s_ref);
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *xHost, *yHost, *zHost;
    uint8_t *xDevice, *yDevice, *zDevice;

    CHECK_ACL(aclrtMallocHost((void **)(&xHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&yHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&zHost), outputByteSize));
    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./input/input_x.bin", inputByteSize, xHost, inputByteSize);
    ReadFile("./input/input_y.bin", inputByteSize, yHost, inputByteSize);

    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    add_custom_do(blockDim, stream, xDevice, yDevice, zDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_z.bin", zHost, outputByteSize);

    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    CHECK_ACL(aclrtFreeHost(zHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
