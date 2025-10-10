from abc import abstractmethod

import mindspore as ms
from mindspore import Tensor, ops, nn
from mindspore.ops import CustomOpBuilder


class WKVKernelCustom(nn.Cell):
    def __init__(self):
        super().__init__()
        self.my_ops = CustomOpBuilder("my_ops", ['./function_wkv7.cpp'], backend="Ascend").load()

        # 预热
        k = Tensor(ops.zeros((1, 12, 1, 64)), ms.float32)
        v = Tensor(ops.zeros((1, 12, 1, 64)), ms.float32)
        w = Tensor(ops.zeros((1, 12, 1, 64)), ms.float32)
        r = Tensor(ops.zeros((1, 12, 1, 64)), ms.float32)
        a = Tensor(ops.zeros((1, 768)), ms.float32)
        b = Tensor(ops.zeros((1, 768)), ms.float32)
        s = Tensor(ops.zeros((1, 12, 64, 64)), ms.float32)
        self.construct(k, v, w, r, a, b, s)

    def construct(self, k, v, w, r, a, b, hi):
        o, ho = self.my_ops.wkv7(k, v, w, r, a, b, hi)
        return o, ho

    @abstractmethod
    def __call__(self, k, v, w, r, a, b, hi):
        return self.construct(k, v, w, r, a, b, hi)
