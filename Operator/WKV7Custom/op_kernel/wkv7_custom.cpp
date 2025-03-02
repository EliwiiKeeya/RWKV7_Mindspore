/****************************************************************************************************
 * File				: wkv7_custom.cpp
 * Date				: 2025-03-01 19:49:45
 * Author			: Eliwii_Keeya
 * Description		: wkv7自定义算子kernel侧源文件
 * Last Modified	: 2025-03-01 19:49:45
 ****************************************************************************************************/
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 3;

class KernelWKV7 {
public:
    __aicore__ inline KernelWKV7() {}


private:
    // Pipe内存管理对象
    AscendC::TPipe tPipeObj;

    // TCubeTilling对象
    TCubeTiling tCubeTilingData;

    // Matmul对象
    using M_DTYPE_V = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_V>;
    using M_DTYPE_K = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_K>;
    using M_DTYPE_VK = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_V>;
    Matmul<M_DTYPE_V, M_DTYPE_K,  M_DTYPE_VK> matmulObjVK;

    // Vector 输入数据 Queue队列管理对象，QuePosition为VECIN
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueS, inQueueR, inQueueW, inQueueK, inQueueV, inQueueA, inQueueKK; 

    // Vector 输出数据Queue队列管理对象，QuePosition为VECOUT
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueX, outQueueS;

    //管理输入输出Global Memory内存地址的对象
    AscendC::GlobalTensor<DTYPE_S> sGm;
    AscendC::GlobalTensor<DTYPE_R> rGm;
    AscendC::GlobalTensor<DTYPE_W> wGm;
    AscendC::GlobalTensor<DTYPE_K> kGm;
    AscendC::GlobalTensor<DTYPE_V> vGm;
    AscendC::GlobalTensor<DTYPE_A> aGm;
    AscendC::GlobalTensor<DTYPE_KK> kkGM;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_S> sRefGm;
    AscendC::GlobalTensor<DTYPE_S> workspaceGm;
    
    uint32_t blockLength;   // 每个核上总计算数据大小
    uint32_t tileNum;       // 每个核上总计算数据分块个数
    uint32_t tileLength;    // 每个分块大小
    uint32_t B;             // batch_size
    uint32_t H;             // n_head
    uint32_t S;             // head_size


public:
    // Tiling初始化函数, 完成tilingData结构体初始化相关操作
    __aicore__ inline void InitTiling(GM_ADDR tilingGM) {
        GET_tilingData(tilingData, tilingGM);     // WKV7CustomTilingData tilingData;
        totalLength = tilingData.totalLength;
        tileNum = tilingData.tileNum;
        B = tilingData.batch_size;
        H = tilingData.n_head;
        S = tilingData.head_size;  
    }

    // TCubeTiling初始化函数, 完成tCubeTilingData对象初始化相关操作
    __aicore__ inline void InitTCubeTiling(TCubeTiling *tCubeTilingData, GM_ADDR tilingGM)
    {
        uint32_t *ptr = reinterpret_cast<uint32_t *>(tCubeTilingData);
        auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);
    
        for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
            *ptr = *(tiling32 + i);
        }
        return;
    }

    // 初始化函数，完成内存初始化相关操作
    __aicore__ inline void Init(GM_ADDR s, GM_ADDR r, GM_ADDR w, GM_ADDR k, GM_ADDR v, GM_ADDR a, GM_ADDR kk, GM_ADDR x, GM_ADDR s_ref, GM_ADDR workspaceGM, GM_ADDR tilingGM) //uint32_t totalLength, uint32_t tileNum)
    {
        // 获取tilingData
        InitTiling(tilingGM);
        InitTCubeTiling(tCubeTilingData, tilingGM);

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
        kkGM.SetGlobalBuffer((__gm__ DTYPE_KK*)kk + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        sRefGm.SetGlobalBuffer((__gm__ DTYPE_S*)s_ref + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        workspaceGm.SetGlobalBuffer((__gm__ DTYPE_S*)workspace + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        // 通过Pipe内存管理对象为输入输出Queue分配内存
        tPipeObj.InitBuffer(inQueueS,  BUFFER_NUM, this->tileLength * sizeof(DTYPE_S));
        tPipeObj.InitBuffer(inQueueR,  BUFFER_NUM, this->tileLength * sizeof(DTYPE_R));
        tPipeObj.InitBuffer(inQueueW,  BUFFER_NUM, this->tileLength * sizeof(DTYPE_W));
        tPipeObj.InitBuffer(inQueueK,  BUFFER_NUM, this->tileLength * sizeof(DTYPE_K));
        tPipeObj.InitBuffer(inQueueV,  BUFFER_NUM, this->tileLength * sizeof(DTYPE_V));
        tPipeObj.InitBuffer(inQueueA,  BUFFER_NUM, this->tileLength * sizeof(DTYPE_A));
        tPipeObj.InitBuffer(inQueueKK, BUFFER_NUM, this->tileLength * sizeof(DTYPE_KK));
        
        tPipeObj.InitBuffer(inQueueB,   BUFFER_NUM, this->tileLength * sizeof(DTYPE_A));    // b   = a * kk
        tPipeObj.InitBuffer(inQueueAB,  BUFFER_NUM, this->tileLength * sizeof(DTYPE_A));	// ab  = a @ b
        tPipeObj.InitBuffer(inQueueSW,  BUFFER_NUM, this->tileLength * sizeof(DTYPE_S));    // sw  = s * w
        tPipeObj.InitBuffer(inQueueSAB, BUFFER_NUM, this->tileLength * sizeof(DTYPE_S));    // sab = s @ ab
        tPipeObj.InitBuffer(inQueueVK,  BUFFER_NUM, this->tileLength * sizeof(DTYPE_V));    // vk  = v @ a

        tPipeObj.InitBuffer(outQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        tPipeObj.InitBuffer(outQueueS, BUFFER_NUM, this->tileLength * sizeof(DTYPE_S));
    }

    // 核心处理函数，实现算子逻辑
    __aicore__ inline void Process()
    {
		int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (uint32_t computeRound = 0; computeRound < loopCount; computeRound++)
		{
            CopyInMatmulVK(computeRound); 	// vk = v @ a
			CopyInMultiplyB(computeRound); 	// b = a * kk
			ProcessMatmulVK(computeRound);
			ProcessMultiplyB(computeRound);

			CopyInMultiplySW(computeRound); // sw = s * w
			CopyInMatmulSAB(computeRound); 	// sab = s @ ab
			ProcessMultiplySW(computeRound);
			ProcessMultiplySAB(computeRound);

			CopyInAddS(computeRound);		// s = sw + sab + vk
			ProcessAddS(computeRound);

			CopyInMatmulX(computeRound);	// x = s @ r
			ProcessMatmulX(computeRound);

			CopyOut(computeRound);	// s, w
        }        
    }

private:
	// vk = v @ a
	__aicore__ inline void CopyInMatmulVK(uint32_t progress) {}
	__aicore__ inline void ProcessMatmulVK(uint32_t progress)
	{
		// Matmul 对象初始化
		REGIST_MATMUL_OBJ(&tPipeObj, GetSysWorkSpacePtr(), matmulObjVK, &tCubeTilingData);

		// 设置Matmul的输入（包括左矩阵、右矩阵、bias）
		matmulObjVK.SetWorkspace(workspaceGM)
		matmulObjVK.Init(&tCubeTilingData);
		matmulObjVK.SetTensorV(vGM);
		matmulObjVK.SetTensorK(kGM);
		
		// 调用matmul iterate获取一块[baseM, baseN]的计算结果
		while (matmulObj.template Iterate<true>())
		{
			MatmulCompute();
			CopyOut(computeRound);
			computeRound++;
		}

		matmulObj.End(); // v @ k
	}

	// b = a * kk
	__aicore__ inline void CopyInMultiplyB(uint32_t progress)
	{
		// 从Queue中分配输入Tensor
		AscendC::LocalTensor<DTYPE_A> aLocal = inQueueA.AllocTensor<DTYPE_A>();
		AscendC::LocalTensor<DTYPE_KK> kklocal = inQueueKK.AllocTensor<DTYPE_KK>();

		// 将GlobalTensor数据拷贝到LocalTensor
		AscendC::DataCopy(aLocal, aGm[progress * this->tileLength], this->tileLength);
		AscendC::DataCopy(kklocal, kkGM[progress * this->tileLength], this->tileLength);
		
		// 将LocalTesor放入VECIN（代表矢量编程中搬入数据的逻辑存放位置）的Queue中
		inQueueA.EnQue(aLocal);
		inQueueKK.EnQue(kklocal);
	}

	// TODO: 完成处理函数定义
	__aicore__ inline void ProcessMultiplyB(uint32_t progress) {}
	__aicore__ inline void CopyInMultiplySW(uint32_t progress) {}
	__aicore__ inline void ProcessMultiplySW(uint32_t progress) {}
	__aicore__ inline void CopyInMatmulSAB(uint32_t progress) {}
	__aicore__ inline void ProcessMultiplySAB(uint32_t progress) {}
	__aicore__ inline void CopyInAddS(uint32_t progress) {}
	__aicore__ inline void ProcessAddS(uint32_t progress) {}
	__aicore__ inline void CopyInMatmulX(uint32_t progress) {}
	__aicore__ inline void ProcessMatmulX(uint32_t progress) {}
	
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
};


extern "C" __global__ __aicore__ void wkv7_custom(GM_ADDR s, GM_ADDR r, GM_ADDR w, GM_ADDR k, GM_ADDR v, GM_ADDR a, GM_ADDR kk, GM_ADDR x, GM_ADDR s_ref, GM_ADDR workspace, GM_ADDR tiling)
{
    KernelWKV7 op;
    op.Init(s, r, w, k, v, a, kk, x, s_ref, worksapce, tiling);
    op.Process()
}

#ifndef ASCEND_CPU_DEBUG
// 核函数调用封装
void wkv7_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* s, uint8_t* r, uint8_t* w, uint8_t* k, uint8_t* v, uint8_t* a, uint8_t* kk, uint8_t* x, uint8_t* s_ref)
{
    wkv7_custom<<<blockDim, l2ctrl, stream>>>(s, r, w, k, v, a, kk, x, s_ref);
}
#endif
