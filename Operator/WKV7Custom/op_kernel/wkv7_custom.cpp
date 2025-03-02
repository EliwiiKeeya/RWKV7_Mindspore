/****************************************************************************************************
 * File				: wkv7_custom.cpp
 * Date				: 2025-03-01 19:49:45
 * Author			: Eliwii_Keeya
 * Description		: wkv7自定义算子kernel侧源文件
 * Last Modified	: 2025-03-01 19:49:45
 ****************************************************************************************************/
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 7;

class KernelWKV7 {
public:
    __aicore__ inline KernelWKV7() {}

    // Tiling初始化函数, 完成tiling结构体初始化相关操作
    __aicore__ inline void InitTiling(GM_ADDR tiling) {
        GET_TILING_DATA(tiling_data, tiling);
        totalLength = tiling_data.totalLength;
        tileNum = tiling_data.tileNum;
        batch_size = tiling_data.batch_size;
        n_head = tiling_data.n_head;
        head_size = tiling_data.head_size;
    }

    // 初始化函数，完成内存初始化相关操作
    __aicore__ inline void Init(GM_ADDR s, GM_ADDR r, GM_ADDR w, GM_ADDR k, GM_ADDR v, GM_ADDR a, GM_ADDR b, GM_ADDR x, GM_ADDR s_ref, uint32_t totalLength, uint32_t tileNum)
    {
        // 使用获取到的TilingData计算得到singleCoreSize(每个核上总计算数据大小)、tileNum（每个核上分块个数）、singleTileLength（每个分块大小）等变量
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        this->B = batch_size;
        this->H = n_head;
        this->S = head_size;
        
        // 获取当前核的起始索引
        sGm.SetGlobalBuffer((__gm__ DTYPE_S*)s + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        rGm.SetGlobalBuffer((__gm__ DTYPE_R*)r + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        wGm.SetGlobalBuffer((__gm__ DTYPE_W*)w + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        kGm.SetGlobalBuffer((__gm__ DTYPE_K*)k + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        vGm.SetGlobalBuffer((__gm__ DTYPE_V*)v + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        aGm.SetGlobalBuffer((__gm__ DTYPE_A*)a + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        bGm.SetGlobalBuffer((__gm__ DTYPE_B*)b + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        sRefGm.SetGlobalBuffer((__gm__ DTYPE_S*)s_ref + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        // 通过Pipe内存管理对象为输入输出Queue分配内存
        pipe.InitBuffer(inQueueS, BUFFER_NUM, this->tileLength * sizeof(DTYPE_S));
        pipe.InitBuffer(inQueueR, BUFFER_NUM, this->tileLength * sizeof(DTYPE_R));
        pipe.InitBuffer(inQueueW, BUFFER_NUM, this->tileLength * sizeof(DTYPE_W));
        pipe.InitBuffer(inQueueK, BUFFER_NUM, this->tileLength * sizeof(DTYPE_K));
        pipe.InitBuffer(inQueueV, BUFFER_NUM, this->tileLength * sizeof(DTYPE_V));
        pipe.InitBuffer(inQueueA, BUFFER_NUM, this->tileLength * sizeof(DTYPE_A));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, this->tileLength * sizeof(DTYPE_B));
        pipe.InitBuffer(outQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueS, BUFFER_NUM, this->tileLength * sizeof(DTYPE_S));
    }

    // 核心处理函数，实现算子逻辑，调用私有成员函数CopyIn、Split、Compute、Aggregate、CopyOut完成矢量算子的三级流水操作
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Split(i);
            Compute(i);
            Aggregate(i);
            CopyOut(i);
        }
    }


private:
    // 搬入函数，完成CopyIn阶段的处理，被核心Process函数调用
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 从Queue中分配输入Tensor
        AscendC::LocalTensor<DTYPE_S> sLocal = inQueueS.AllocTensor<DTYPE_S>();
        AscendC::LocalTensor<DTYPE_R> rLocal = inQueueR.AllocTensor<DTYPE_R>();
        AscendC::LocalTensor<DTYPE_W> wLocal = inQueueW.AllocTensor<DTYPE_W>();
        AscendC::LocalTensor<DTYPE_K> kLocal = inQueueK.AllocTensor<DTYPE_K>();
        AscendC::LocalTensor<DTYPE_V> vLocal = inQueueV.AllocTensor<DTYPE_V>();
        AscendC::LocalTensor<DTYPE_A> aLocal = inQueueA.AllocTensor<DTYPE_A>();
        AscendC::LocalTensor<DTYPE_B> bLocal = inQueueB.AllocTensor<DTYPE_B>();

         // 将GlobalTensor数据拷贝到LocalTensor
        AscendC::DataCopy(sLocal, sGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(rLocal, rGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(wLocal, wGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(kLocal, kGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(vLocal, vGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(aLocal, aGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(bLocal, bGm[progress * this->tileLength], this->tileLength);
        
        // 将LocalTesor放入VECIN（代表矢量编程中搬入数据的逻辑存放位置）的Queue中
        inQueueS.EnQue(sLocal);
        inQueueR.EnQue(rLocal);
        inQueueW.EnQue(wLocal);
        inQueueK.EnQue(kLocal);
        inQueueV.EnQue(vLocal);
        inQueueA.EnQue(aLocal);
        inQueueB.EnQue(bLocal);
    }
    // 切分函数，完成Split阶段的处理，被核心Process函数调用
    __aicore__ inline void Split(int32_t progress)
    {
        ;
    }
    // 计算函数，完成Compute阶段的处理，被核心Process函数调用
    __aicore__ inline void Compute(int32_t progress)
    {
        ;
    }
    // 整合函数，完成Aggregate阶段的处理，被核心Process函数调用
    __aicore__ inline void Aggregate(int32_t progress)
    {
        ;
    }
    // 搬出函数，完成CopyOut阶段的处理，被核心Process函数调用
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从VecOut的Queue中取出输出Tensor
        AscendC::LocalTensor<DTYPE_X> xLocal = outQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_S> sLocal = outQueueS.DeQue<DTYPE_S>();
        // 将输出Tensor拷贝到GlobalTensor中
        AscendC::DataCopy(xGm[progress * this->tileLength], xLocal, this->tileLength);
        AscendC::DataCopy(sRefGm[progress * this->tileLength], sLocal, this->tileLength);
        // 将不再使用的LocalTensor释放
        outQueueX.FreeTensor(xLocal);
        outQueueX.FreeTensor(sLocal);
    }


private:
    //Pipe内存管理对象
    AscendC::TPipe pipe;
    //输入数据Queue队列管理对象，QuePosition为VECIN
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueS, inQueueR, inQueueW, inQueueK, inQueueV, inQueueA, inQueueB; 
    //输出数据Queue队列管理对象，QuePosition为VECOUT
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueX, outQueueS;
    //管理输入输出Global Memory内存地址的对象
    AscendC::GlobalTensor<DTYPE_S> sGm;
    AscendC::GlobalTensor<DTYPE_R> rGm;
    AscendC::GlobalTensor<DTYPE_W> wGm;
    AscendC::GlobalTensor<DTYPE_K> kGm;
    AscendC::GlobalTensor<DTYPE_V> vGm;
    AscendC::GlobalTensor<DTYPE_A> aGm;
    AscendC::GlobalTensor<DTYPE_B> bGm;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_S> sRefGm;
    
    uint32_t blockLength;   // 每个核上总计算数据大小
    uint32_t tileNum;       // 每个核上总计算数据分块个数
    uint32_t tileLength;    // 每个分块大小
    uint32_t B;             // batch_size
    uint32_t H;             // n_head
    uint32_t S;             // head_size
};


extern "C" __global__ __aicore__ void wkv7_custom(GM_ADDR s, GM_ADDR r, GM_ADDR w, GM_ADDR k, GM_ADDR v, GM_ADDR a, GM_ADDR b, GM_ADDR x, GM_ADDR s_ref, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelWKV7 op;
    op.Init(s, r, w, k, v, a, b, x, s_ref, tiling_data.totalLength, tiling_data.tileNum);
}
