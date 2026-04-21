import time
import torch
from model import RWKV_RNN
from tokenizer import RWKV_TOKENIZER
from sampler import sample_logits
import numpy as np


def run_benchmark(model, batch_size, seq_length, trials=100):
    args = {
        'MODEL_NAME': model,
        'vocab_size': 65536,
        'batch_size': batch_size,
        'device': 'cuda'
    }
    model = RWKV_RNN(args)
    tokenizer = RWKV_TOKENIZER()
    initial_string = "The Eiffel tower is in the city of"
    TEMPERATURE = 2.5
    TOP_P = 0.1

    times = []
    mem_allocs = []
    for _ in range(trials):
        print(
            f"Trial {_+1}/{trials} for batch size {batch_size} and seq length {seq_length}")
        torch.cuda.reset_peak_memory_stats(args['device'])
        token = torch.tensor(tokenizer.encode(
            initial_string), dtype=torch.int64, device=args['device']).expand([batch_size, -1])
        for t in torch.unbind(token, axis=-1):
            out = model(t)
        token_sampled = sample_logits(out, TEMPERATURE, TOP_P).type_as(token)
        token = torch.cat((token, token_sampled.unsqueeze(1)), 1)
        start_time = time.time()
        for step in range(seq_length):
            out = model(token_sampled)
            token_sampled = sample_logits(
                out, TEMPERATURE, TOP_P).type_as(token)
            token = torch.cat((token, token_sampled.unsqueeze(1)), 1)
        end_time = time.time()
        times.append(end_time - start_time)
        mem_allocs.append(torch.cuda.max_memory_allocated(
            args['device']) / (1024 ** 2))  # MB

    avg_time = np.mean(times)
    avg_mem_alloc = np.mean(mem_allocs)
    tokens_generated = seq_length * batch_size
    speed = tokens_generated / avg_time
    speed_per_batch = speed / batch_size
    return avg_time, speed, speed_per_batch, avg_mem_alloc


if __name__ == '__main__':
    import os
    import csv

    models = [
        "/tmp/code/RWKV7_Pytorch/model/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096",
        "/tmp/code/RWKV7_Pytorch/model/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096",
        "/tmp/code/RWKV7_Pytorch/model/RWKV-x070-World-1.5B-v3-20250127-ctx4096",
        "/tmp/code/RWKV7_Pytorch/model/RWKV-x070-World-2.9B-v3-20250211-ctx4096"
    ]
    batch_sizes = [1024, 2048, 4096]
    seq_lengths = [100]
    trials = 10

    csv_path = 'model_benchmark_results_torch.csv'
    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, 'a', newline='') as f:
        with torch.no_grad():
            writer = csv.DictWriter(f, fieldnames=[
                                    'model', 'batch_size', 'seq_length', 'avg_time', 'speed', 'speed_per_batch', 'avg_mem_alloc'])
            if not file_exists:
                writer.writeheader()

            for model in models:
                for batch_size in batch_sizes:
                    for seq_length in seq_lengths:
                        avg_time, speed, speed_per_batch, avg_mem_alloc = run_benchmark(
                            model, batch_size, seq_length, trials)
                        row = {
                            'model': model,
                            'batch_size': batch_size,
                            'seq_length': seq_length,
                            'avg_time': avg_time,
                            'speed': speed,
                            'speed_per_batch': speed_per_batch,
                            'avg_mem_alloc': avg_mem_alloc
                        }
                        writer.writerow(row)
                        f.flush()
                        os.fsync(f.fileno())
                        print(
                            f"Batch size: {batch_size}, Seq length: {seq_length}, Avg time: {avg_time:.2f}s, Speed: {speed:.2f} tokens/s, Speed per batch: {speed_per_batch:.2f} tokens/s")
