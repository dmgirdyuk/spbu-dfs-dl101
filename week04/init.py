import os
import random
from os.path import join as pjoin
from typing import Callable, Literal

import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn.functional import relu, sigmoid, tanh


WeightInitName = Literal[
    "random", "xavier_uni", "xavier_norm", "kaiming_uni", "kaiming_norm"
]


def main() -> None:
    seed_everything(42)
    os.makedirs("plots")

    for act_func in [sigmoid, tanh, relu]:
        weight_init_experiment(init_type="random", act_func=act_func, std=0.01)
        weight_init_experiment(init_type="random", act_func=act_func, std=0.05)
        weight_init_experiment(init_type="xavier_uni", act_func=act_func)
        weight_init_experiment(init_type="xavier_norm", act_func=act_func)
        weight_init_experiment(init_type="kaiming_uni", act_func=act_func)
        weight_init_experiment(init_type="kaiming_norm", act_func=act_func)


def seed_everything(seed: int = 314159, torch_deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(torch_deterministic)


def weight_init_experiment(
    batch_size: int = 8,
    h_dim: int = 4096,
    layers_num: int = 6,
    std: float = 0.01,
    act_func: Callable[[Tensor], Tensor] = relu,
    init_type: WeightInitName = "random",
) -> None:
    if init_type == "random":
        weight_init = _random_weight_init
        tag = f"random_{act_func.__name__}_std_{std:.03f}"
    elif init_type == "xavier_uni":
        weight_init = _xavier_uni_weight_init
        tag = f"xavier_uni_{act_func.__name__}"
    elif init_type == "xavier_norm":
        weight_init = _xavier_norm_weight_init
        tag = f"xavier_norm_{act_func.__name__}"
    elif init_type == "kaiming_uni":
        weight_init = _kaiming_uni_weight_init
        tag = f"kaiming_uni_{act_func.__name__}"
    elif init_type == "kaiming_norm":
        weight_init = _kaiming_norm_weight_init
        tag = f"kaiming_norm_{act_func.__name__}"
    else:
        raise NotImplementedError

    dims = [h_dim] * layers_num
    hidden_states = []
    x = torch.randn(batch_size, dims[0])
    for h_in, h_out in zip(dims[:-1], dims[1:]):
        weights = weight_init(h_in=h_in, h_out=h_out, std=std)
        x = act_func(x @ weights)
        hidden_states.append(x.reshape(-1))

    plot_histograms(hidden_states, tag=tag)


def _random_weight_init(
    h_in: int, h_out: int, std: float = 0.01
) -> Tensor:
    return torch.randn(h_in, h_out) * std


def _xavier_uni_weight_init(
    h_in: int, h_out: int, **_
) -> Tensor:
    a = (6 / (h_in + h_out)) ** 0.5
    return (torch.rand(h_in, h_out) * 2 - 1) * a


def _xavier_norm_weight_init(
    h_in: int, h_out: int, **_
) -> Tensor:
    return torch.randn(h_in, h_out) * (2 / (h_in + h_out)) ** 0.5


def _kaiming_uni_weight_init(
    h_in: int, h_out: int, **_
) -> Tensor:
    a = (3 / h_in) ** 0.5
    return (torch.randn(h_in, h_out) * 2 - 1) * a


def _kaiming_norm_weight_init(
    h_in: int, h_out: int, **_
) -> Tensor:
    return torch.randn(h_in, h_out) * (2 / h_in) ** 0.5


def plot_histograms(hidden_states: list[Tensor], tag: str) -> None:
    layers_num = len(hidden_states)
    fig, axes = plt.subplots(1, layers_num)
    for layer_idx, hidden in enumerate(hidden_states):
        mean = hidden.mean().item()
        std = hidden.std().item()

        axes[layer_idx].hist(hidden, bins=30)
        axes[layer_idx].set_title(
            f"Layer {layer_idx + 1}\nmean {mean:.03f}, std {std:.03f}"
        )
        axes[layer_idx].set_xlim(
            xmin=-1.0, xmax=max(1.0, hidden.max().item())
        )

    fig.set_size_inches(layers_num * 4, 4)
    plt.savefig(pjoin("plots", f"{tag}.png"), dpi=300)


if __name__ == "__main__":
    main()
