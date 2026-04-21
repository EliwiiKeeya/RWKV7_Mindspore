import numpy as np
import json
import mindspore as ms
import mindspore.nn
from mindspore import ops


class MatrixMethod(mindspore.nn.Cell):
    def __init__(self):
        super().__init__()

        # 预热
        k = ops.zeros((1, 12, 1, 64), ms.float32)
        v = ops.zeros((1, 12, 64, 1), ms.float32)
        w = ops.zeros((1, 12, 1, 64), ms.float32)
        r = ops.zeros((1, 12, 64, 1), ms.float32)
        a = ops.zeros((1, 12, 64, 1), ms.float32)
        b = ops.zeros((1, 12, 1, 64), ms.float32)
        s = ops.zeros((1, 12, 64, 64), ms.float32)
        self.construct(k, v, w, r, a, b, s)

    def construct(self, k, v, w, r, a, b, s):
        """
        k: (B, H, T, S)
        v: (B, H, S, T)
        w: (B, H, T, S)
        r: (B, H, S, T)
        a: (B, H, S, T)
        b: (B, H, T, S)
        s: (B, H, S, S)
        """
        vk = v @ k
        ab = a @ b
        s = s * ops.exp(w) + s @ ab + vk
        x = s @ r
        return x, s


class NaiveMethod(mindspore.nn.Cell):
    def __init__(self):
        super().__init__()

        # 预热
        k = ops.zeros((1, 12, 1, 64), ms.float32)
        v = ops.zeros((1, 12, 1, 64), ms.float32)
        w = ops.zeros((1, 12, 1, 64), ms.float32)
        r = ops.zeros((1, 12, 1, 64), ms.float32)
        a = ops.zeros((1, 12, 1, 64), ms.float32)
        b = ops.zeros((1, 12, 1, 64), ms.float32)
        s = ops.zeros((1, 12, 64, 64), ms.float32)
        self.construct(k, v, w, r, a, b, s)

    def construct(self, k, v, w, r, a, b, s):
        """
        k: (B, H, T, S)
        v: (B, H, T, S)
        w: (B, H, T, S)
        r: (B, H, T, S)
        a: (B, H, T, S)
        b: (B, H, T, S)
        s: (B, H, S, S)
        """
        B, H, T, S = k.shape
        x = ops.zeros((B, H, T, S), ms.float32)
        s = s.copy()
        
        for t in range(T):
            for batch in range(B):
                for h in range(H):                    
                    s[batch, h, :, :] = s[batch, h, :, :] * ops.exp(w[batch, h, t, :]) \
                        + (s[batch, h, :, :] @ a[batch, h, t, :]).unsqueeze(-1) * b[batch, h, t, None, :] \
                        + v[batch, h, t, :, None] * k[batch, h, t, None, :]
                    x[batch, h, t, :] = s[batch, h, :, :] @ r[batch, h, t, :]
        
        # for t in range(T):
        #     for bi in range(B):
        #         for hi in range(H):
        #             r_t = r[bi, hi, t]
        #             k_t = k[bi, hi, t]
        #             v_t = v[bi, hi, t]
        #             a_t = a[bi, hi, t]
        #             b_t = b[bi, hi, t]
        #             w_t = ops.exp(w[bi, hi, t])

        #             sa = ops.sum((a_t[None, :] * s[bi, hi]), dim=1)

        #             s[bi, hi] = (s[bi, hi] * w_t[None, :] + 
        #                         k_t[None, :] * v_t[:, None] + 
        #                         sa[:, None] * b_t[None, :])

        #             y = ops.sum((s[bi, hi] * r_t[None, :]), dim=1)
        #             s[bi, hi, t] = y
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

    k_matrix = ms.tensor(k, dtype=ms.float32)
    v_matrix = ms.tensor(v, dtype=ms.float32)
    w_matrix = ms.tensor(w, dtype=ms.float32)
    r_matrix = ms.tensor(r, dtype=ms.float32)
    a_matrix = ms.tensor(a, dtype=ms.float32)
    b_matrix = ms.tensor(b, dtype=ms.float32)
    s_matrix = ms.tensor(s, dtype=ms.float32)

    k_naive = ms.tensor(k.reshape(B, H, 1, S), ms.float32)
    v_naive = ms.tensor(v.reshape(B, H, 1, S), ms.float32)
    w_naive = ms.tensor(w.reshape(B, H, 1, S), ms.float32)
    r_naive = ms.tensor(r.reshape(B, H, 1, S), ms.float32)
    a_naive = ms.tensor(a.reshape(B, H, 1, S), ms.float32)
    b_naive = ms.tensor(b.reshape(B, H, 1, S), ms.float32)
    s_naive = ms.tensor(s, ms.float32)

    matrix_method = MatrixMethod()
    naive_method = NaiveMethod()

    cos_sim_x_list, cos_sim_s_list = [], []
    kl_div_x_list, kl_div_s_list = [], []
    max_abs_x_list, mean_abs_x_list, rel_err_x_list = [], [], []
    max_abs_s_list, mean_abs_s_list, rel_err_s_list = [], [], []

    for _ in range(times):
        # print(f"trial {_+1}:")

        out1, state1 = matrix_method.construct(
            k_matrix, v_matrix, w_matrix, r_matrix, a_matrix, b_matrix, s_matrix
        )
        out2, state2 = naive_method.construct(
            k_naive, v_naive, w_naive, r_naive, a_naive, b_naive, s_naive
        )
        
        # print(out1.sum(), out2.sum(), out1.sum() - out2.sum())
        # print(state1.sum(), state2.sum(), state1.sum() - state2.sum())        

        out1_np = out1.asnumpy()
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
    times = 10
    modes = ["normal", "zeros", "ones"]
    results = {}
    for mode in modes:
        print(f"Benchmarking with mode: {mode}")
        metrics = benchmark_same_tensor(times=times, mode=mode)
        results[mode] = metrics
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}")
        print("-" * 50)

    with open("test_naive_precision.json", "w") as f:
        json.dump(results, f, indent=4)
