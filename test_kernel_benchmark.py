import time

import mindspore as ms
from mindspore import Tensor, ops, nn

from kernel import WKVKernelCustom


class Pydantic(nn.Cell):
    def __init__(self):
        super().__init__()

        # 预热
        k = Tensor(ops.zeros((1, 12, 1, 64)), ms.float32)
        v = Tensor(ops.zeros((1, 12, 64, 1)), ms.float32)
        w = Tensor(ops.zeros((1, 12, 1, 64)), ms.float32)
        r = Tensor(ops.zeros((1, 12, 64, 1)), ms.float32)
        a = Tensor(ops.zeros((1, 12, 64, 1)), ms.float32)
        b = Tensor(ops.zeros((1, 12, 1, 64)), ms.float32)
        s = Tensor(ops.zeros((1, 12, 64, 64)), ms.float32)
        self.construct(k, v, w, r, a, b, s)

    def construct(self, k, v, w, r, a, b, s):
        """
        k: (B, H, 1, S)
        v: (B, H, S, 1)
        w: (B, H, 1, S)
        r: (B, H, S, 1)
        a: (B, H, S, 1)
        b: (B, H, 1, S)
        s: (B, H, S, S)
        """
        vk = v @ k
        ab = a @ b
        s = s * ops.exp(w) + s @ ab + vk
        x = s @ r
        return x, s


class Kernel(nn.Cell):
    def __init__(self):
        super().__init__()
        self.wkv_kernel = WKVKernelCustom()

        # 预热
        k = Tensor(ops.zeros((1, 12, 1, 64)), ms.float32)
        v = Tensor(ops.zeros((1, 12, 1, 64)), ms.float32)
        w = Tensor(ops.zeros((1, 12, 1, 64)), ms.float32)
        r = Tensor(ops.zeros((1, 12, 1, 64)), ms.float32)
        a = Tensor(ops.zeros((1, 768)), ms.float32)
        b = Tensor(ops.zeros((1, 768)), ms.float32)
        s = Tensor(ops.zeros((1, 12, 64, 64)), ms.float32)
        self.construct(k, v, w, r, a, b, s)

    def construct(self, k, v, w, r, a, b, s):
        """
        k: (B, H, 1, S)
        v: (B, H, 1, S)
        w: (B, H, 1, S)
        r: (B, H, 1, S)
        a: (B, E)
        b: (B, E)
        s: (B, H, S, S)
        """
        x, s = self.wkv_kernel(k, v, w, r, a, b, s)
        return x, s


def pydantic_benchmark():
    batch_size = 2
    n_head = 12
    head_size = 64
    n_embd = n_head * head_size

    B = batch_size
    H = n_head
    S = head_size
    E = n_embd         # E = H * S

    pydantic = Pydantic()

    k = Tensor(ops.randn((B, H, 1, S)), ms.float32)
    v = Tensor(ops.randn((B, H, S, 1)), ms.float32)
    w = Tensor(ops.randn((B, H, 1, S)), ms.float32)
    r = Tensor(ops.randn((B, H, S, 1)), ms.float32)
    a = Tensor(ops.randn((B, H, S, 1)), ms.float32)
    b = Tensor(ops.randn((B, H, 1, S)), ms.float32)
    s = Tensor(ops.randn((B, H, S, S)), ms.float32)

    out1, state1 = pydantic.construct(k, v, w, r, a, b, s)

    time_start = time.time()
    out1, state1 = pydantic.construct(k, v, w, r, a, b, s)
    time_end = time.time()

    pydantic_time = time_end - time_start
    return pydantic_time


def kernel_benchmark():
    batch_size = 2
    n_head = 12
    head_size = 64
    n_embd = n_head * head_size

    B = batch_size
    H = n_head
    S = head_size
    E = n_embd         # E = H * S

    kernel = Kernel()

    k = Tensor(ops.randn((B, H, 1, S)), ms.float32)
    v = Tensor(ops.randn((B, H, 1, S)), ms.float32)
    w = Tensor(ops.randn((B, H, 1, S)), ms.float32)
    r = Tensor(ops.randn((B, H, 1, S)), ms.float32)
    a = Tensor(ops.randn((B, E)), ms.float32)
    b = Tensor(ops.randn((B, E)), ms.float32)
    s = Tensor(ops.randn((B, H, S, S)), ms.float32)

    time_start = time.time()
    out2, state2 = kernel.construct(k, v, w, r, a, b, s)
    time_end = time.time()

    kernel_time = time_end - time_start
    return kernel_time


def benchmark_random_tensor(times=100):
    pydantic_times = []
    kernel_times = []
    for _ in range(times):
        pydantic_time, kernel_time = pydantic_benchmark(), kernel_benchmark()
        pydantic_times.append(pydantic_time)
        kernel_times.append(kernel_time)
    pydantic_time = sum(pydantic_times) / len(pydantic_times)
    kernel_time = sum(kernel_times) / len(kernel_times)
    print(f"Pydantic time: {pydantic_time}, Kernel time: {kernel_time}")
    print(f"Speedup: {pydantic_time / kernel_time}x")

    return pydantic_time, kernel_time

def benchmark_same_tensor(times=100):
    batch_size = 2
    n_head = 12
    head_size = 64
    n_embd = n_head * head_size

    B = batch_size
    H = n_head
    S = head_size
    E = n_embd         # E = H * S

    # Pydantic data
    k = Tensor(ops.randn((B, H, 1, S)), ms.float32)
    v = Tensor(ops.randn((B, H, S, 1)), ms.float32)
    w = Tensor(ops.randn((B, H, 1, S)), ms.float32)
    r = Tensor(ops.randn((B, H, S, 1)), ms.float32)
    a = Tensor(ops.randn((B, H, S, 1)), ms.float32)
    b = Tensor(ops.randn((B, H, 1, S)), ms.float32)
    s = Tensor(ops.randn((B, H, S, S)), ms.float32)

    # Kernel data
    v_kernel = v.view(B, H, 1, S)
    r_kernel = r.view(B, H, 1, S)
    a_kernel = a.view(B, E)
    b_kernel = b.view(B, E)

    pydantic = Pydantic()
    kernel = Kernel()

    out1, state1 = pydantic.construct(k, v, w, r, a, b, s)
    out2, state2 = kernel.construct(k, v, w, r, a, b, s)

    pydantic_times = []
    kernel_times = []
    for _ in range(times):
        time_start = time.time()
        out1, state1 = pydantic.construct(k, v, w, r, a, b, s)
        time_end = time.time()
        pydantic_times.append(time_end - time_start)

        time_start = time.time()
        out2, state2 = kernel.construct(k, v, w, r, a, b, s)
        time_end = time.time()
        kernel_times.append(time_end - time_start)

    pydantic_time = sum(pydantic_times) / len(pydantic_times)
    kernel_time = sum(kernel_times) / len(kernel_times)
    print(f"Pydantic time: {pydantic_time}, Kernel time: {kernel_time}")
    print(f"Speedup: {pydantic_time / kernel_time}x")

    return pydantic_time, kernel_time

if __name__ == "__main__":
    times = 1000
    print("Benchmarking with random tensors:")
    benchmark_random_tensor(times=times)
    print("Benchmarking with same tensors:")
    benchmark_same_tensor(times=times)

# 结果
# Benchmarking with random tensors:
# Pydantic time: 0.00027157044410705566, Kernel time: 0.00012213659286499024
# Speedup: 2.223497788310253x
# Benchmarking with same tensors:
# Pydantic time: 0.0002385993003845215, Kernel time: 0.00010583853721618652
# Speedup: 2.254370729795301x