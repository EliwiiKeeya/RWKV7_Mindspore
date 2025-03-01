#ifndef WKV7_CUSTOM_TILING_H
#define WKV7_CUSTOM_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling
{
	BEGIN_TILING_DATA_DEF(TilingData)
		TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 总计算数据量
		TILING_DATA_FIELD_DEF(uint32_t, tileNum);	  // 每个核上总计算数据分块个数
		TILING_DATA_FIELD_DEF(uint32_t, batch_size);
		TILING_DATA_FIELD_DEF(uint32_t, n_head);
		TILING_DATA_FIELD_DEF(uint32_t, head_size);
	END_TILING_DATA_DEF;

	// 注册tiling数据到对应的算子
	REGISTER_TILING_DATA_CLASS(WKV7Custom, TilingData)
} // namesapce optiling
#endif // WKV7_CUSTOM_TILING_H
