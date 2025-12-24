import time

import mindspore
import numpy as np
from mindspore import ops

from model import RWKV_RNN
from tokenizer import RWKV_TOKENIZER
from sampler import sample_logits


def test_token_speed(trials=5, length_per_trial=50, temperature=2.5, top_p=0.1):
    args = {
        'MODEL_NAME': '/home/ma-user/work/model/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096',
        'vocab_size': 65536,
        'batch_size': 1,
    }
    model = RWKV_RNN(args)
    tokenizer = RWKV_TOKENIZER()
    BATCH_SIZE = args['batch_size']
    initial_string = "The Eiffel tower is in the city of"

    speeds = []
    for trial in range(trials):
        state = mindspore.Tensor(
            ops.zeros([BATCH_SIZE, *model.state_size]), dtype=mindspore.float32)
        token = mindspore.Tensor(tokenizer.encode(
            initial_string), dtype=mindspore.int64).expand([BATCH_SIZE, -1])
        for t in ops.unstack(token, axis=-1):
            out = model(t, state)
        token_sampled = sample_logits(out, temperature, top_p).type_as(token)
        token = ops.cat((token, token_sampled.unsqueeze(1)), 1)

        start_time = time.time()
        for step in range(length_per_trial):
            out = model(token_sampled, state)
            token_sampled = sample_logits(
                out, temperature, top_p).type_as(token)
            token = ops.cat((token, token_sampled.unsqueeze(1)), 1)
        end_time = time.time()

        tokens_generated = length_per_trial * BATCH_SIZE
        speed = tokens_generated / (end_time - start_time)
        speeds.append(speed)
        print(f"Trial {trial+1}: {speed:.2f} tokens/second")

    avg_speed = sum(speeds) / len(speeds)
    print(
        f"\nAverage token generation speed over {trials} trials: {avg_speed:.2f} tokens/second")

    return speeds


if __name__ == '__main__':
    speeds = test_token_speed(
        trials=25, length_per_trial=50, temperature=2.5, top_p=0.1)
    mean = np.mean(speeds)
    std = np.std(speeds)
    cv = std / mean
    print(f"均值: {mean:.2f}, 标准差: {std:.2f}, 变异系数: {cv:.2f}")

    with open("test_plain_speeds.txt", "w") as f:
        f.write(str(speeds) + "\n")
        f.write(f"均值: {mean:.2f}, 标准差: {std:.2f}, 变异系数: {cv:.2f}" + "\n")
