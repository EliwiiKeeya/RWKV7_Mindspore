# -*- encoding: utf-8 -*-
# @File         : sampler.py
# @Date         : 2024/10/04 16:26:42
# @Author       : Eliwii_Keeya
# @Modified from: yuunnn-w, et al., 2024 -- RWKV_Pytorch

import mindspore
from mindnlp.core import ops


def sample_logits(out: mindspore.Tensor, temperature: float = 1.0, top_p: float = 0.8) -> mindspore.Tensor:
    """
    Sample from the logits tensor produced by the model.

    Args:
        out (mindspore.Tensor): Logits tensor from the model, shape [* , vocab_size].
        temperature (float): Temperature parameter for controlling the diversity of sampling. Default is 1.0.
        top_p (float): Top-p truncation parameter for stabilizing and controlling the sampling probability distribution. Default is 0.8.

    Returns:
        mindspore.Tensor: Sampled indices, shape [*].
    """
    # Apply temperature scaling
    scaled_logits = out / temperature

    # Convert logits to probabilities
    probabilities = ops.softmax(scaled_logits, dim=-1)

    # Sort the probabilities to identify the top-p candidates
    sorted_probs, sorted_indices = ops.sort(probabilities, descending=True)

    # Compute the cumulative distribution of probabilities
    cumulative_probs = ops.cumsum(sorted_probs, dim=-1)

    # Remove tokens with a cumulative probability above the threshold (top_p)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].copy()
    sorted_indices_to_remove[..., 0] = 0

    # Create a mask for the indices to remove
    indices_to_remove = sorted_indices_to_remove.scatter(axis=-1, index=sorted_indices, src=sorted_indices_to_remove)

    # Use the mask to zero out probabilities that should be removed
    probabilities[indices_to_remove] = 0.0

    # Resample if probabilities are all zero (unlikely but just in case)
    if ops.all(probabilities == 0):
        probabilities = ops.ones_like(probabilities)
        probabilities /= probabilities.sum()

    # Sample from the modified distribution
    sampled_indices = ops.multinomial(probabilities, 1)

    return sampled_indices.squeeze(-1)
