import os
import time

import mindspore as ms
from mindspore import Tensor, ops
from mindspore._c_expression import _framework_profiler_step_start, _framework_profiler_step_end


def einsum_benchmark_with_profile():
    head_size = 64
    seq_len = 32

    J = seq_len
    I = head_size

    # einsum 'j, i -> ji'
    x = Tensor(ops.randn((J,)), ms.float32)
    y = Tensor(ops.randn((I,)), ms.float32)

    # 预热
    out = ops.einsum('j,i->ji', x, y)

    time_start = time.time()
    _framework_profiler_step_start()
    out = ops.einsum('j,i->ji', x, y)
    _framework_profiler_step_end()
    time_end = time.time()

    einsum_time = time_end - time_start
    return einsum_time


def benchmark_einsum(times=100):
    einsum_times = []
    for _ in range(times):
        einsum_time = einsum_benchmark_with_profile()
        einsum_times.append(einsum_time)
    avg_einsum_time = sum(einsum_times) / len(einsum_times)
    print(f"Einsum 'j,i->ji' average time: {avg_einsum_time}")
    return avg_einsum_time


def main():
    os.environ['MS_ENABLE_RUNTIME_PROFILER'] = '1'
    times = 100
    print("Benchmarking einsum 'j,i->ji' with random tensors:")
    benchmark_einsum(times=times)

if __name__ == "__main__":
    main()
