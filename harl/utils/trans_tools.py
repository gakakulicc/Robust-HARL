"""Tools for HARL."""
import numpy as np


def _t2n(value):
    """Convert torch.Tensor to numpy.ndarray."""
    return value.detach().cpu().numpy()


def _flatten(T, N, value):
    """Flatten the first two dimensions of a tensor."""
    return value.reshape(T * N, *value.shape[2:])

def softmax(x):
    x = x.copy()
    x -= np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def _sa_cast(value):
    """This function is used for buffer data operation.
    Specifically, it transposes a tensor from (episode_length, n_rollout_threads, *dim) to (n_rollout_threads, episode_length, *dim).
    Then it combines the first two dimensions into one dimension.
    """
    return value.transpose(1, 0, 2).reshape(-1, *value.shape[2:])


def _ma_cast(value):
    """This function is used for buffer data operation.
    Specifically, it transposes a tensor from (episode_length, n_rollout_threads, num_agents, *dim) to (n_rollout_threads, num_agents, episode_length, *dim).
    Then it combines the first three dimensions into one dimension.
    """
    return value.transpose(1, 2, 0, 3).reshape(-1, *value.shape[3:])
