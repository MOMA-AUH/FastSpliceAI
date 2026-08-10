"""PyTorch port of the SpliceAI model.

Architecture: https://doi.org/10.1016/j.cell.2018.12.015
"""

import itertools
from collections.abc import Sequence
from importlib.resources import as_file, files
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from spliceai import name

__all__ = ["EnsembleSpliceAIModel", "CONTEXT", "HALF_CONTEXT"]

CONTEXT = 10000
HALF_CONTEXT = CONTEXT // 2
_CHANNELS = 32
_BATCH_NORM_EPSILON = 0.001
_SKIP_CONNECTION_SPECS = (
    {"kernel_size": 11, "dilation": 1},
    {"kernel_size": 11, "dilation": 4},
    {"kernel_size": 21, "dilation": 10},
    {"kernel_size": 41, "dilation": 25},
)
_RESIDUAL_BLOCKS_PER_GROUP = 4


class ResidualBlock(nn.Sequential):
    def __init__(self, *args: nn.Module):
        super().__init__(*args)

    def forward(self, x):
        return x + super().forward(x)


class AccumulativeSkipConnection(nn.Module):
    def __init__(self, module, skip):
        super().__init__()
        self.module = module
        self.skip = skip

    def forward(self, x, x_skip):
        x = self.module(x)
        if x_skip is None:
            return x, self.skip(x)
        return x, x_skip + self.skip(x)


class SpliceAIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.ModuleList()
        self.stem.append(
            AccumulativeSkipConnection(
                module=nn.Conv1d(4, _CHANNELS, kernel_size=1),
                skip=nn.Conv1d(_CHANNELS, _CHANNELS, kernel_size=1),
            )
        )
        for kw in _SKIP_CONNECTION_SPECS:
            residual_blocks = [
                ResidualBlock(
                    nn.BatchNorm1d(_CHANNELS, eps=_BATCH_NORM_EPSILON),
                    nn.ReLU(),
                    nn.Conv1d(
                        _CHANNELS,
                        _CHANNELS,
                        padding="same",
                        **kw,
                    ),
                    nn.BatchNorm1d(_CHANNELS, eps=_BATCH_NORM_EPSILON),
                    nn.ReLU(),
                    nn.Conv1d(
                        _CHANNELS,
                        _CHANNELS,
                        padding="same",
                        **kw,
                    ),
                )
                for _ in range(_RESIDUAL_BLOCKS_PER_GROUP)
            ]
            self.stem.append(
                AccumulativeSkipConnection(
                    module=nn.Sequential(*residual_blocks),
                    skip=nn.Conv1d(_CHANNELS, _CHANNELS, 1),
                )
            )
        self.output_conv = nn.Conv1d(_CHANNELS, 3, 1)

    @classmethod
    def from_keras_h5(cls, path):
        def read_keras_array(weights, layer_name, variable_name):
            dataset_name = f"model_weights/{layer_name}/{layer_name}/{variable_name}:0"
            try:
                return np.asarray(weights[dataset_name])
            except KeyError as error:
                raise ValueError(f"Keras weight file is missing {dataset_name}") from error


        def copy_tensor(target, value, dataset_name):
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Keras weight {dataset_name} has shape {value.shape}; "
                    f"expected {tuple(target.shape)}"
                )
            with torch.no_grad():
                target.copy_(torch.from_numpy(np.ascontiguousarray(value)))

        model = cls()
        conv_names = (f"conv1d_{i}" for i in itertools.count(1))
        batch_norm_names = (
            f"batch_normalization_{i}" for i in itertools.count(1)
        )

        try:
            with h5py.File(path, "r") as weights:
                for layer in model.modules():
                    if isinstance(layer, nn.Conv1d):
                        layer_name = next(conv_names)
                        copy_tensor(
                            layer.weight,
                            read_keras_array(
                                weights, layer_name, "kernel"
                            ).transpose(2, 1, 0),
                            f"{layer_name}/kernel",
                        )
                        copy_tensor(
                            layer.bias,
                            read_keras_array(weights, layer_name, "bias"),
                            f"{layer_name}/bias",
                        )
                    elif isinstance(layer, nn.BatchNorm1d):
                        layer_name = next(batch_norm_names)
                        variables = {
                            "gamma": layer.weight,
                            "beta": layer.bias,
                            "moving_mean": layer.running_mean,
                            "moving_variance": layer.running_var,
                        }
                        for variable_name, target in variables.items():
                            copy_tensor(
                                target,
                                read_keras_array(
                                    weights, layer_name, variable_name
                                ),
                                f"{layer_name}/{variable_name}",
                            )
        except OSError as error:
            raise ValueError(
                f"Unable to read Keras weights from {path}: {error}"
            ) from error
        model.requires_grad_(False)
        model.eval()
        return model

    def forward(self, x):
        x_skip = None
        for m in self.stem:
            x, x_skip = m(x, x_skip)
        x_skip = x_skip[:, :, HALF_CONTEXT : -HALF_CONTEXT]
        return F.softmax(self.output_conv(x_skip), dim=1)


class EnsembleSpliceAIModel(nn.Module):
    """Frozen PyTorch ensemble backed by the five bundled Keras weight files."""

    def __init__(self, model_paths: Sequence[str | Path] | None = None):
        super().__init__()
        if model_paths is None:
            model_paths = tuple(
                files(name).joinpath(f"models/spliceai{index}.h5")
                for index in range(1, 6)
            )
        else:
            model_paths = tuple(model_paths)
        if not model_paths:
            raise ValueError("model_paths must contain at least one Keras weight file")

        members = []
        for model_path in model_paths:
            if isinstance(model_path, (str, Path)):
                members.append(SpliceAIModel.from_keras_h5(model_path))
            else:
                with as_file(model_path) as resolved_path:
                    members.append(SpliceAIModel.from_keras_h5(resolved_path))
        self.members = nn.ModuleList(members)
        self.requires_grad_(False)
        self.eval()

    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.ndim != 3 or inputs.shape[-1] != 4:
            raise ValueError("inputs must have shape (batch, length, 4)")
        if inputs.shape[1] <= CONTEXT:
            raise ValueError("input length must be greater than 10000")

        channels_first = inputs.transpose(1, 2)
        predictions = torch.stack(
            [member(channels_first) for member in self.members]
        ).mean(dim=0)
        return predictions.transpose(1, 2)

    def infer(self, inputs):
        """Return NumPy predictions for channels-last NumPy inputs."""
        parameter = next(self.parameters())
        tensor = torch.as_tensor(np.asarray(inputs), dtype=parameter.dtype)
        tensor = tensor.to(device=parameter.device)
        with torch.inference_mode():
            predictions = self(tensor)
        return predictions.detach().cpu().numpy()
