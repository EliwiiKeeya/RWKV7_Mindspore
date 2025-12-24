import time
import mindspore
from mindspore import ops
from model_operator import RWKV_RNN
from tokenizer import RWKV_TOKENIZER
from sampler import sample_logits
import matplotlib.pyplot as plt
import numpy as np

def run_benchmark(batch_size, seq_length, trials=100):
    args = {
        'MODEL_NAME': '/home/ma-user/work/model/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096',
        'vocab_size': 65536,
        'batch_size': batch_size,
    }
    model = RWKV_RNN(args)
    tokenizer = RWKV_TOKENIZER()
    initial_string = "The Eiffel tower is in the city of"
    TEMPERATURE = 2.5
    TOP_P = 0.1

    times = []
    for _ in range(trials):
        token = mindspore.Tensor(tokenizer.encode(initial_string), dtype=mindspore.int64).expand([batch_size, -1])
        for t in ops.unstack(token, axis=-1):
            out = model(t)
        token_sampled = sample_logits(out, TEMPERATURE, TOP_P).type_as(token)
        token = ops.cat((token, token_sampled.unsqueeze(1)), 1)

        start_time = time.time()
        for step in range(seq_length):
            out = model(token_sampled)
            token_sampled = sample_logits(out, TEMPERATURE, TOP_P).type_as(token)
            token = ops.cat((token, token_sampled.unsqueeze(1)), 1)
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = np.mean(times)
    tokens_generated = seq_length * batch_size
    speed = tokens_generated / avg_time
    speed_per_batch = speed / batch_size
    return avg_time, speed, speed_per_batch

if __name__ == '__main__':
    import os, csv
    mindspore.set_device("Ascend")
    mindspore.run_check()

    batch_sizes = [1024, 2048, 4096]
    seq_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    trials = 100

    csv_path = 'vertical_benchmark_results.csv'
    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['batch_size', 'seq_length', 'avg_time', 'speed', 'speed_per_batch'])
        if not file_exists:
            writer.writeheader()

        for batch_size in batch_sizes:
            for seq_length in seq_lengths:
                avg_time, speed, speed_per_batch = run_benchmark(batch_size, seq_length, trials)
                row = {
                    'batch_size': batch_size,
                    'seq_length': seq_length,
                    'avg_time': avg_time,
                    'speed': speed,
                    'speed_per_batch': speed_per_batch
                }
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
                print(f"Batch size: {batch_size}, Seq length: {seq_length}, Avg time: {avg_time:.2f}s, Speed: {speed:.2f} tokens/s, Speed per batch: {speed_per_batch:.2f} tokens/s")


    # # 绘制图表
    # fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # for batch_size in batch_sizes:
    #     x = [r['seq_length'] for r in results if r['batch_size'] == batch_size]
    #     y = [r['speed'] for r in results if r['batch_size'] == batch_size]
    #     ax[0].plot(x, y, marker='o', label=f'Batch {batch_size}')
    # ax[0].set_title('Token Generation Speed vs Sequence Length')
    # ax[0].set_xlabel('Sequence Length')
    # ax[0].set_ylabel('Tokens/second')
    # ax[0].legend()

    # for seq_length in seq_lengths:
    #     x = [r['batch_size'] for r in results if r['seq_length'] == seq_length]
    #     y = [r['speed'] for r in results if r['seq_length'] == seq_length]
    #     ax[1].plot(x, y, marker='o', label=f'Seq {seq_length}')
    # ax[1].set_title('Token Generation Speed vs Batch Size')
    # ax[1].set_xlabel('Batch Size')
    # ax[1].set_ylabel('Tokens/second')
    # ax[1].legend()

    # plt.tight_layout()
    # plt.savefig('vertical_benchmark_plot.png')
    # plt.show()
