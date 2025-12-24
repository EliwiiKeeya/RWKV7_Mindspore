import time

import mindspore as ms
from mindspore import Tensor, ops, nn

from kernel import WKVKernelCustom


class Kernel(nn.Cell):
    def __init__(self):
        super().__init__()
        self.wkv_kernel = WKVKernelCustom()

    def construct(self, k, v, w, r, a, b, s):
        x, s = self.wkv_kernel(k, v, w, r, a, b, s)
        return x, s


def test_kernel_launch_time(times=1000):
    batch_size = 2
    n_head = 12
    head_size = 64
    n_embd = n_head * head_size

    k = Tensor(ops.zeros((batch_size, n_head, 1, head_size)), ms.float32)
    v = Tensor(ops.zeros((batch_size, n_head, 1, head_size)), ms.float32)
    w = Tensor(ops.zeros((batch_size, n_head, 1, head_size)), ms.float32)
    r = Tensor(ops.zeros((batch_size, n_head, 1, head_size)), ms.float32)
    a = Tensor(ops.zeros((batch_size, n_embd)), ms.float32)
    b = Tensor(ops.zeros((batch_size, n_embd)), ms.float32)
    s = Tensor(ops.zeros((batch_size, n_head, head_size, head_size)), ms.float32)

    kernel = Kernel()
    times_list = []
    for i in range(times):
        start = time.time()
        x, s_out = kernel.construct(k, v, w, r, a, b, s)
        end = time.time()
        times_list.append(end - start)
        print(f"Run {i+1}: {end - start:.6f} seconds")

    print(f"Avg launch time: {sum(times_list)/len(times_list):.6f} seconds")

    with open("test_kernel_launch_times.txt", "w") as f:
        f.write(str(times_list) + "\n")
        f.write(
            f"Avg launch time: {sum(times_list)/len(times_list):.6f} seconds\n")


if __name__ == '__main__':
    test_kernel_launch_time(times=1000)
