import numpy as np
import json
import torch
import mindspore as ms
import mindspore.nn
from mindspore import ops

from kernel import WKVKernelCustom


class Pydantic(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # 预热
        k = torch.zeros((1, 12, 1, 64), dtype=torch.float32)
        v = torch.zeros((1, 12, 64, 1), dtype=torch.float32)
        w = torch.zeros((1, 12, 1, 64), dtype=torch.float32)
        r = torch.zeros((1, 12, 64, 1), dtype=torch.float32)
        a = torch.zeros((1, 12, 64, 1), dtype=torch.float32)
        b = torch.zeros((1, 12, 1, 64), dtype=torch.float32)
        s = torch.zeros((1, 12, 64, 64), dtype=torch.float32)
        self.forward(k, v, w, r, a, b, s)

    def forward(self, k, v, w, r, a, b, s):
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
        s = s * torch.exp(w) + s @ ab + vk
        x = s @ r
        return x, s


class Kernel(mindspore.nn.Cell):
    def __init__(self):
        super().__init__()
        self.wkv_kernel = WKVKernelCustom()

        # 预热
        k = ops.zeros((1, 12, 1, 64), ms.float32)
        v = ops.zeros((1, 12, 1, 64), ms.float32)
        w = ops.zeros((1, 12, 1, 64), ms.float32)
        r = ops.zeros((1, 12, 1, 64), ms.float32)
        a = ops.zeros((1, 768), ms.float32)
        b = ops.zeros((1, 768), ms.float32)
        s = ops.zeros((1, 12, 64, 64), ms.float32)
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


def cosine_similarity(x1, x2):
    x1_flat = x1.flatten()
    x2_flat = x2.flatten()
    dot_product = np.dot(x1_flat, x2_flat)
    norm1 = np.linalg.norm(x1_flat)
    norm2 = np.linalg.norm(x2_flat)
    return dot_product / (norm1 * norm2 + 1e-8)


def kl_divergence(p, q):
    p_flat = p.flatten()
    q_flat = q.flatten()
    p_flat = np.clip(p_flat, 1e-8, 1)
    q_flat = np.clip(q_flat, 1e-8, 1)
    return np.sum(p_flat * np.log(p_flat / q_flat))


def error_metrics(ref, test):
    diff = ref - test
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rel_err = float(np.linalg.norm(diff) / (np.linalg.norm(ref) + 1e-8))
    return max_abs, mean_abs, rel_err


def make_data(name, shape, mode="normal"):
    if mode == "zeros":
        return np.zeros(shape, dtype=np.float32)
    if mode == "ones":
        return np.ones(shape, dtype=np.float32)

    if name == "k":
        return np.clip(np.random.normal(-0.01, 0.15, size=shape), -4.0, 0.2).astype(np.float32)
    if name == "v":
        return np.clip(np.random.normal(0.0, 0.07, size=shape), -0.2, 1.7).astype(np.float32)
    if name == "w":
        return np.clip(np.random.normal(-0.26, 0.27, size=shape), -0.6, 0.0).astype(np.float32)
    if name == "r":
        return np.clip(np.random.normal(0.0, 0.05, size=shape), -0.35, 0.35).astype(np.float32)
    if name == "a":
        return np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    if name == "b":
        return np.clip(np.random.normal(0.0, 0.05, size=shape), -0.5, 0.6).astype(np.float32)
    if name == "s":
        return np.clip(np.random.normal(0.0, 0.02, size=shape), -1.1, 1.1).astype(np.float32)

    return np.random.normal(0.0, 0.2, size=shape).astype(np.float32)


def benchmark_same_tensor(times=100, mode="normal"):
    batch_size = 2
    n_head = 12
    head_size = 64
    n_embd = n_head * head_size

    B = batch_size
    H = n_head
    S = head_size
    E = n_embd

    k = make_data("k", (B, H, 1, S), mode)
    v = make_data("v", (B, H, S, 1), mode)
    w = make_data("w", (B, H, 1, S), mode)
    r = make_data("r", (B, H, S, 1), mode)
    a = make_data("a", (B, H, S, 1), mode)
    b = make_data("b", (B, H, 1, S), mode)
    s = make_data("s", (B, H, S, S), mode)

    k_pydantic = torch.tensor(k, dtype=torch.float32)
    v_pydantic = torch.tensor(v, dtype=torch.float32)
    w_pydantic = torch.tensor(w, dtype=torch.float32)
    r_pydantic = torch.tensor(r, dtype=torch.float32)
    a_pydantic = torch.tensor(a, dtype=torch.float32)
    b_pydantic = torch.tensor(b, dtype=torch.float32)
    s_pydantic = torch.tensor(s, dtype=torch.float32)

    k_kernel = ms.tensor(k.reshape(B, H, 1, S), ms.float32)
    v_kernel = ms.tensor(v.reshape(B, H, 1, S), ms.float32)
    w_kernel = ms.tensor(w.reshape(B, H, 1, S), ms.float32)
    r_kernel = ms.tensor(r.reshape(B, H, 1, S), ms.float32)
    a_kernel = ms.tensor(a.reshape(B, H, 1, S), ms.float32)
    b_kernel = ms.tensor(b.reshape(B, H, 1, S), ms.float32)
    s_kernel = ms.tensor(s, ms.float32)

    pydantic = Pydantic()
    kernel = Kernel()

    cos_sim_x_list, cos_sim_s_list = [], []
    kl_div_x_list, kl_div_s_list = [], []
    max_abs_x_list, mean_abs_x_list, rel_err_x_list = [], [], []
    max_abs_s_list, mean_abs_s_list, rel_err_s_list = [], [], []

    for _ in range(times):
        
        out1, state1 = pydantic(
            k_pydantic, v_pydantic, w_pydantic, r_pydantic, a_pydantic, b_pydantic, s_pydantic
        )
        out2, state2 = kernel(
            k_kernel, v_kernel, w_kernel, r_kernel, a_kernel, b_kernel, s_kernel
        )

        out1_np = out1.detach().cpu().numpy() if hasattr(
            out1, 'detach') else out1.asnumpy()
        out2_np = out2.asnumpy() if hasattr(
            out2, 'asnumpy') else out2.detach().cpu().numpy()
        state1_np = state1.detach().cpu().numpy() if hasattr(
            state1, 'detach') else state1.asnumpy()
        state2_np = state2.asnumpy() if hasattr(
            state2, 'asnumpy') else state2.detach().cpu().numpy()

        cos_sim_x_list.append(cosine_similarity(out1_np, out2_np))
        cos_sim_s_list.append(cosine_similarity(state1_np, state2_np))
        kl_div_x_list.append(kl_divergence(out1_np, out2_np))
        kl_div_s_list.append(kl_divergence(state1_np, state2_np))

        max_abs, mean_abs, rel_err = error_metrics(out1_np, out2_np)
        max_abs_x_list.append(max_abs)
        mean_abs_x_list.append(mean_abs)
        rel_err_x_list.append(rel_err)

        max_abs, mean_abs, rel_err = error_metrics(state1_np, state2_np)
        max_abs_s_list.append(max_abs)
        mean_abs_s_list.append(mean_abs)
        rel_err_s_list.append(rel_err)

    avg_cos_sim_x = sum(cos_sim_x_list) / len(cos_sim_x_list)
    avg_cos_sim_s = sum(cos_sim_s_list) / len(cos_sim_s_list)
    avg_kl_div_x = sum(kl_div_x_list) / len(kl_div_x_list)
    avg_kl_div_s = sum(kl_div_s_list) / len(kl_div_s_list)
    avg_max_abs_x = sum(max_abs_x_list) / len(max_abs_x_list)
    avg_mean_abs_x = sum(mean_abs_x_list) / len(mean_abs_x_list)
    avg_rel_err_x = sum(rel_err_x_list) / len(rel_err_x_list)
    avg_max_abs_s = sum(max_abs_s_list) / len(max_abs_s_list)
    avg_mean_abs_s = sum(mean_abs_s_list) / len(mean_abs_s_list)
    avg_rel_err_s = sum(rel_err_s_list) / len(rel_err_s_list)

    return {
        "avg_cos_sim_x": avg_cos_sim_x,
        "avg_cos_sim_s": avg_cos_sim_s,
        "avg_kl_div_x": avg_kl_div_x,
        "avg_kl_div_s": avg_kl_div_s,
        "avg_max_abs_err_x": avg_max_abs_x,
        "avg_mean_abs_err_x": avg_mean_abs_x,
        "avg_rel_err_x": avg_rel_err_x,
        "avg_max_abs_err_s": avg_max_abs_s,
        "avg_mean_abs_err_s": avg_mean_abs_s,
        "avg_rel_err_s": avg_rel_err_s,
    }


if __name__ == "__main__":
    times = 1000
    modes = ["normal", "zeros", "ones"]
    results = {}
    for mode in modes:
        print(f"Benchmarking with mode: {mode}")
        metrics = benchmark_same_tensor(times=times, mode=mode)
        results[mode] = metrics
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}")
        print("-" * 50)

    with open("test_kernel_precision.json", "w") as f:
        json.dump(results, f, indent=4)
