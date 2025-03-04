/****************************************************************************************************
 * File				: wkv7_custom.cpp
 * Date				: 2025-03-01 19:49:45
 * Author			: Eliwii_Keeya
 * Description		: wkv7自定义算子kernel侧源文件
 * Last Modified	: 2025-03-01 19:49:45
 ****************************************************************************************************/
#include "kernel_operator.h"
#include <cstdint>

#define GM_ADDR __gm__ uint8_t*

constexpr int32_t TOTAL_LENGTH = 8 * 2048;                            // total length of data
constexpr int32_t USE_CORE_NUM = 8;                                   // num of core used
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;         // length computed of each core
constexpr int32_t TILE_NUM = 8;                                       // split data into 8 tiles for each core
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM; // separate to 2 parts, due to double buffer


class KernelWKV7 {
public:
    __aicore__ inline KernelWKV7() {}


private:
    // Pipe内存管理对象
    AscendC::TPipe tPipeObj;

    // TCubeTilling对象
    // TCubeTiling tCubeTilingData;

    // Matmul对象
    // using M_DTYPE_V = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_V>;
    // using M_DTYPE_K = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_K>;
    // using M_DTYPE_VK = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_V>;
    // Matmul<M_DTYPE_V, M_DTYPE_K,  M_DTYPE_VK> matmulObjVK;

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

    AscendC::GlobalTensor<DTYPE_A> abGm;
    AscendC::GlobalTensor<DTYPE_A> bGm;
    AscendC::GlobalTensor<DTYPE_S> swGm;
    AscendC::GlobalTensor<DTYPE_S> sabGm;
    AscendC::GlobalTensor<DTYPE_V> vkGm;
    
    uint32_t blockLength;   // 每个核上总计算数据大小
    uint32_t tileNum;       // 每个核上总计算数据分块个数
    uint32_t tileLength;    // 每个分块大小
    // uint32_t B;             // batch_size
    // uint32_t H;             // n_head
    // uint32_t S;             // head_size


public:
    // Tiling初始化函数, 完成tilingData结构体初始化相关操作
    // __aicore__ inline void InitTiling(GM_ADDR tilingGM) {
    //     GET_tilingData(tilingData, tilingGM);     // WKV7CustomTilingData tilingData;
    //     totalLength = tilingData.totalLength;
    //     tileNum = tilingData.tileNum;
    //     B = tilingData.batch_size;
    //     H = tilingData.n_head;
    //     S = tilingData.head_size;  
    // }

    // TCubeTiling初始化函数, 完成tCubeTilingData对象初始化相关操作
    // __aicore__ inline void InitTCubeTiling(TCubeTiling *tCubeTilingData, GM_ADDR tilingGM)
    // {
    //     uint32_t *ptr = reinterpret_cast<uint32_t *>(tCubeTilingData);
    //     auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);
    
    //     for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
    //         *ptr = *(tiling32 + i);
    //     }
    //     return;
    // }

    // 初始化函数，完成内存初始化相关操作
    __aicore__ inline void Init(GM_ADDR s, GM_ADDR r, GM_ADDR w, GM_ADDR k, GM_ADDR v, GM_ADDR a, GM_ADDR kk, GM_ADDR x, GM_ADDR s_ref) // , GM_ADDR workspaceGM, GM_ADDR tilingGM) //uint32_t totalLength, uint32_t tileNum)
    {
        // 获取tilingData
        // InitTiling(tilingGM);
        // InitTCubeTiling(tCubeTilingData, tilingGM);

        // 使用获取到的TilingData计算得到singleCoreSize(每个核上总计算数据大小)、tileNum（每个核上分块个数）、singleTileLength（每个分块大小）等变量
        this->blockLength = TOTAL_LENGTH / AscendC::GetBlockNum();
        this->tileNum = TILE_LENGTH;
        this->tileLength = this->blockLength / TILE_NUM / BUFFER_NUM;
        // this->B = batch_size;
        // this->H = n_head;
        // this->S = head_size;
        
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

        abGm.SetGlobalBuffer((__gm__ DTYPE_A*)ab + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        bGm.SetGlobalBuffer((__gm__ DTYPE_A*)b + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        swGm.SetGlobalBuffer((__gm__ DTYPE_S*)sw + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        sabGm.SetGlobalBuffer((__gm__ DTYPE_S*)sab + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        vkGm.SetGlobalBuffer((__gm__ DTYPE_V*)vk + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

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
        tPipeObj.InitBuffer(inQueueVK,  BUFFER_NUM, this->tileLength * sizeof(DTYPE_V));    // vk  = v @ k

        tPipeObj.InitBuffer(outQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        tPipeObj.InitBuffer(outQueueS, BUFFER_NUM, this->tileLength * sizeof(DTYPE_S));
    }

    // 核心处理函数，实现算子逻辑
    __aicore__ inline void Process()
    {
		int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (uint32_t computeRound = 0; computeRound < loopCount; computeRound++)
		{
            CopyIn(computeRound);	// s, r, w, k, v, a, kk
            Compute(computeRound);	// b, ab, sw, sab, vk, x, s_ref

            // CopyInMatmulVK(computeRound); 	// vk = v @ a
			// CopyInMultiplyB(computeRound); 	// b = a * kk
			// ProcessMatmulVK(computeRound);
			// ProcessMultiplyB(computeRound);

			// CopyInMultiplySW(computeRound); // sw = s * w
			// CopyInMatmulSAB(computeRound); 	// sab = s @ ab
			// ProcessMultiplySW(computeRound);
			// ProcessMultiplySAB(computeRound);

			// CopyInAddS(computeRound);		// s = sw + sab + vk
			// ProcessAddS(computeRound);

			// CopyInMatmulX(computeRound);	// x = s @ r
			// ProcessMatmulX(computeRound);

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

    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_S> sLocal = inQueueS.AllocTensor<DTYPE_S>();
        AscendC::LocalTensor<DTYPE_R> rLocal = inQueueR.AllocTensor<DTYPE_R>();
        AscendC::LocalTensor<DTYPE_W> wLocal = inQueueW.AllocTensor<DTYPE_W>();
        AscendC::LocalTensor<DTYPE_K> kLocal = inQueueK.AllocTensor<DTYPE_K>();
        AscendC::LocalTensor<DTYPE_V> vLocal = inQueueV.AllocTensor<DTYPE_V>();
        AscendC::LocalTensor<DTYPE_A> aLocal = inQueueA.AllocTensor<DTYPE_A>();
        AscendC::LocalTensor<DTYPE_KK> kkLocal = inQueueKK.AllocTensor<DTYPE_KK>();

        AscendC::DataCopy(sLocal, sGm[progress * TILE_LENGTH], TILE_LENGTH);
        AscendC::DataCopy(rLocal, rGm[progress * TILE_LENGTH], TILE_LENGTH);
        AscendC::DataCopy(wLocal, wGm[progress * TILE_LENGTH], TILE_LENGTH);
        AscendC::DataCopy(kLocal, kGm[progress * TILE_LENGTH], TILE_LENGTH);
        AscendC::DataCopy(vLocal, vGm[progress * TILE_LENGTH], TILE_LENGTH);
        AscendC::DataCopy(aLocal, aGm[progress * TILE_LENGTH], TILE_LENGTH);
        AscendC::DataCopy(kkLocal, kkGM[progress * TILE_LENGTH], TILE_LENGTH);

        inQueueS.EnQue(sLocal);
        inQueueR.EnQue(rLocal);
        inQueueW.EnQue(wLocal);
        inQueueK.EnQue(kLocal);
        inQueueV.EnQue(vLocal);
        inQueueA.EnQue(aLocal);
        inQueueKK.EnQue(kkLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_S> sLocal = inQueueS.DeQue<DTYPE_S>();
        AscendC::LocalTensor<DTYPE_R> rLocal = inQueueR.DeQue<DTYPE_R>();
        AscendC::LocalTensor<DTYPE_W> wLocal = inQueueW.DeQue<DTYPE_W>();
        AscendC::LocalTensor<DTYPE_K> kLocal = inQueueK.DeQue<DTYPE_K>();
        AscendC::LocalTensor<DTYPE_V> vLocal = inQueueV.DeQue<DTYPE_V>();
        AscendC::LocalTensor<DTYPE_A> aLocal = inQueueA.DeQue<DTYPE_A>();
        AscendC::LocalTensor<DTYPE_KK> kkLocal = inQueueKK.DeQue<DTYPE_KK>();

        AscendC::LocalTensor<DTYPE_A> bLocal = inQueueB.AllocTensor<DTYPE_A>();
        AscendC::LocalTensor<DTYPE_A> abLocal = inQueueAB.AllocTensor<DTYPE_A>();
        AscendC::LocalTensor<DTYPE_S> swLocal = inQueueSW.AllocTensor<DTYPE_S>();
        AscendC::LocalTensor<DTYPE_S> sabLocal = inQueueSAB.AllocTensor<DTYPE_S>();
        AscendC::LocalTensor<DTYPE_V> vkLocal = inQueueVK.AllocTensor<DTYPE_V>();

        AscendC::LocalTensor<DTYPE_X> xLocal = outQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_S> sRefLocal = outQueueS.AllocTensor<DTYPE_S>();

        inQueueB.EnQue(bLocal);
        inQueueAB.EnQue(abLocal);
        inQueueSW.EnQue(swLocal);
        inQueueSAB.EnQue(sabLocal);
        inQueueVK.EnQue(vkLocal);

        AscendC::Add(aLocal, kkLocal, bLocal, TILE_LENGTH);
        AscendC::Add(aLocal, bLocal, abLocal, TILE_LENGTH);
        AscendC::Add(sLocal, wLocal, swLocal, TILE_LENGTH);
        AscendC::Add(sLocal, abLocal, sabLocal, TILE_LENGTH);
        AscendC::Add(vLocal, kLocal, vkLocal, TILE_LENGTH);
        AscendC::Add(swLocal, sabLocal, xLocal, TILE_LENGTH);
        AscendC::Add(xLocal, vkLocal, sRefLocal, TILE_LENGTH);

        outQueueX.EnQue<DTYPE_X>(xLocal);
        outQueueS.EnQue<DTYPE_S>(sRefLocal);

        inQueueS.FreeTensor(sLocal);
        inQueueR.FreeTensor(rLocal);
        inQueueW.FreeTensor(wLocal);
        inQueueK.FreeTensor(kLocal);
        inQueueV.FreeTensor(vLocal);
        inQueueA.FreeTensor(aLocal);

        inQueueKK.FreeTensor(kkLocal);
        inQueueB.FreeTensor(bLocal);
        inQueueAB.FreeTensor(abLocal);
        inQueueSW.FreeTensor(swLocal);
        inQueueSAB.FreeTensor(sabLocal);
        inQueueVK.FreeTensor(vkLocal);
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
};


extern "C" __global__ __aicore__ void wkv7_custom(GM_ADDR s, GM_ADDR r, GM_ADDR w, GM_ADDR k, GM_ADDR v, GM_ADDR a, GM_ADDR kk, GM_ADDR x, GM_ADDR s_ref) //, GM_ADDR workspace, GM_ADDR tiling)
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
