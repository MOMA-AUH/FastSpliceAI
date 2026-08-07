from collections.abc import Sequence
from importlib.resources import as_file, files
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from spliceai import logger, name

__all__ = [
    "EnsembleModel",
    "configure_model_device",
    "configure_model_threads",
]

_CHANNELS = 32
_CONTEXT = 10000
_KERAS_BATCH_NORM_EPSILON = 0.001
_BLOCK_SPECS = (
    *((11, 1),) * 4,
    *((11, 4),) * 4,
    *((21, 10),) * 4,
    *((41, 25),) * 4,
)
_BLOCK_CONVOLUTION_NAMES = (
    ("conv1d_3", "conv1d_4"),
    ("conv1d_5", "conv1d_6"),
    ("conv1d_7", "conv1d_8"),
    ("conv1d_9", "conv1d_10"),
    ("conv1d_12", "conv1d_13"),
    ("conv1d_14", "conv1d_15"),
    ("conv1d_16", "conv1d_17"),
    ("conv1d_18", "conv1d_19"),
    ("conv1d_21", "conv1d_22"),
    ("conv1d_23", "conv1d_24"),
    ("conv1d_25", "conv1d_26"),
    ("conv1d_27", "conv1d_28"),
    ("conv1d_30", "conv1d_31"),
    ("conv1d_32", "conv1d_33"),
    ("conv1d_34", "conv1d_35"),
    ("conv1d_36", "conv1d_37"),
)
_SKIP_CONVOLUTION_NAMES = (
    "conv1d_11",
    "conv1d_20",
    "conv1d_29",
    "conv1d_38",
)

class _FrozenBatchNorm1d(nn.Module):
    """Batch normalization that always uses stored Keras inference statistics."""

    def __init__(self, channels):
        super().__init__()
        self.register_buffer("weight", torch.ones(channels))
        self.register_buffer("bias", torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, inputs):
        return F.batch_norm(
            inputs,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=False,
            momentum=0.0,
            eps=_KERAS_BATCH_NORM_EPSILON,
        )


class _ResidualBlock(nn.Module):
    def __init__(self, kernel_size, dilation):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.batch_norm_1 = _FrozenBatchNorm1d(_CHANNELS)
        self.conv_1 = nn.Conv1d(
            _CHANNELS,
            _CHANNELS,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.batch_norm_2 = _FrozenBatchNorm1d(_CHANNELS)
        self.conv_2 = nn.Conv1d(
            _CHANNELS,
            _CHANNELS,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, inputs):
        outputs = self.conv_1(F.relu(self.batch_norm_1(inputs)))
        outputs = self.conv_2(F.relu(self.batch_norm_2(outputs)))
        return inputs + outputs


class _SpliceAIModel(nn.Module):
    """One frozen SpliceAI member with weights imported from a Keras HDF5 file."""

    def __init__(self):
        super().__init__()
        self.initial_conv = nn.Conv1d(4, _CHANNELS, 1)
        self.initial_skip = nn.Conv1d(_CHANNELS, _CHANNELS, 1)
        self.residual_blocks = nn.ModuleList(
            _ResidualBlock(kernel_size, dilation)
            for kernel_size, dilation in _BLOCK_SPECS
        )
        self.skip_convs = nn.ModuleList(
            nn.Conv1d(_CHANNELS, _CHANNELS, 1) for _ in _SKIP_CONVOLUTION_NAMES
        )
        self.output_conv = nn.Conv1d(_CHANNELS, 3, 1)

    @classmethod
    def from_keras_h5(cls, path):
        model = cls()
        model._load_keras_weights(path)
        model.requires_grad_(False)
        model.eval()
        return model

    def _load_keras_weights(self, path):
        try:
            with h5py.File(path, "r") as weights:
                _copy_conv1d(weights, "conv1d_1", self.initial_conv)
                _copy_conv1d(weights, "conv1d_2", self.initial_skip)

                for block_index, (block, convolution_names) in enumerate(
                    zip(self.residual_blocks, _BLOCK_CONVOLUTION_NAMES)
                ):
                    batch_norm_index = 2 * block_index + 1
                    _copy_batch_norm(
                        weights,
                        f"batch_normalization_{batch_norm_index}",
                        block.batch_norm_1,
                    )
                    _copy_conv1d(weights, convolution_names[0], block.conv_1)
                    _copy_batch_norm(
                        weights,
                        f"batch_normalization_{batch_norm_index + 1}",
                        block.batch_norm_2,
                    )
                    _copy_conv1d(weights, convolution_names[1], block.conv_2)

                for layer_name, layer in zip(_SKIP_CONVOLUTION_NAMES, self.skip_convs):
                    _copy_conv1d(weights, layer_name, layer)
                _copy_conv1d(weights, "conv1d_39", self.output_conv)
        except OSError as error:
            raise ValueError(
                f"Unable to read Keras weights from {path}: {error}"
            ) from error

    def forward(self, inputs):
        outputs = self.initial_conv(inputs)
        skip = self.initial_skip(outputs)
        for block_index, block in enumerate(self.residual_blocks):
            outputs = block(outputs)
            if (block_index + 1) % 4 == 0:
                skip_index = block_index // 4
                skip = skip + self.skip_convs[skip_index](outputs)

        skip = skip[:, :, _CONTEXT // 2 : -_CONTEXT // 2]
        return F.softmax(self.output_conv(skip), dim=1)


def _read_keras_array(weights, layer_name, variable_name):
    dataset_name = f"model_weights/{layer_name}/{layer_name}/{variable_name}:0"
    try:
        return np.asarray(weights[dataset_name])
    except KeyError as error:
        raise ValueError(f"Keras weight file is missing {dataset_name}") from error


def _copy_tensor(target, value, dataset_name):
    if tuple(value.shape) != tuple(target.shape):
        raise ValueError(
            f"Keras weight {dataset_name} has shape {value.shape}; "
            f"expected {tuple(target.shape)}"
        )
    with torch.no_grad():
        target.copy_(torch.from_numpy(np.ascontiguousarray(value)))


def _copy_conv1d(weights, layer_name, layer):
    kernel_name = f"{layer_name}/kernel"
    kernel = _read_keras_array(weights, layer_name, "kernel")
    _copy_tensor(layer.weight, kernel.transpose(2, 1, 0), kernel_name)
    bias_name = f"{layer_name}/bias"
    _copy_tensor(
        layer.bias,
        _read_keras_array(weights, layer_name, "bias"),
        bias_name,
    )


def _copy_batch_norm(weights, layer_name, layer):
    variables = {
        "gamma": layer.weight,
        "beta": layer.bias,
        "moving_mean": layer.running_mean,
        "moving_variance": layer.running_var,
    }
    for variable_name, target in variables.items():
        _copy_tensor(
            target,
            _read_keras_array(weights, layer_name, variable_name),
            f"{layer_name}/{variable_name}",
        )


class EnsembleModel(nn.Module):
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
                members.append(_SpliceAIModel.from_keras_h5(model_path))
            else:
                with as_file(model_path) as resolved_path:
                    members.append(_SpliceAIModel.from_keras_h5(resolved_path))
        self.members = nn.ModuleList(members)
        self.requires_grad_(False)
        self.eval()

    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.ndim != 3 or inputs.shape[-1] != 4:
            raise ValueError("inputs must have shape (batch, length, 4)")
        if inputs.shape[1] <= _CONTEXT:
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


def configure_model_threads(threads):
    """Set the PyTorch intra-op thread count when explicitly requested."""
    if threads is not None:
        torch.set_num_threads(threads)


def configure_model_device(device):
    """Construct a model on the requested device, with auto-mode fallback."""
    if device == "auto":
        if torch.cuda.is_available():
            try:
                return EnsembleModel().to("cuda")
            except (AssertionError, RuntimeError) as error:
                logger.warning(
                    "Unable to initialize model on CUDA; falling back to CPU: %s",
                    error,
                )
        try:
            return EnsembleModel()
        except (AssertionError, RuntimeError) as error:
            raise ValueError(f"Unable to initialize model on cpu: {error}") from error

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    try:
        return EnsembleModel().to(device)
    except (AssertionError, RuntimeError) as error:
        raise ValueError(f"Unable to initialize model on {device}: {error}") from error
