import enum
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
import torch._dynamo
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

torch._dynamo.config.suppress_errors = True

class FP8Config(QuantizationConfig):
    """Config class for FP8."""

    @classmethod
    def get_name(cls) -> str:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # TODO: PyTorch 2.3.0+ is required to run FP8 on
        # SM 89 (e.g. Ada) GPUs. Specifically, this PR has to
        # be included: https://github.com/pytorch/pytorch/pull/118881
        return 90

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FP8Config":
        return cls()

    def get_linear_method(self) -> "Fp8LinearMethod":
        return Fp8LinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []


class Fp8LinearState(Enum):

    UNINITIALIZED = enum.auto()
    READY = enum.auto()


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    We now support common FP16/BF16 model checkpoints ONLY. The weight
    scaling factor will be initialized during the first forward pass.

    Note that we currently only support per-tensor quantization due to
    torch._scaled_mm support.
       
    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: FP8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_size_per_partition: int,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(torch.empty(output_size_per_partition,
                                       input_size_per_partition,
                                       dtype=torch.float8_e4m3fn),
                           requires_grad=False)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)

        # Will be initialized for FP16/BF16 checkpoints
        # during the first forward pass.
        w_scale = Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("weight_scaling_factor", w_scale)

        # We always initialize the state to UNINITIALIZED because
        # we cannot know whether weights going to be loaded are in FP8
        # or not. If they are in FP8, the first forward pass simply
        # sets the state to READY without re-quantization.
        layer.fp8_linear_state = Fp8LinearState.UNINITIALIZED

    def proc_before_loading(self, layer: Module, param: Parameter,
                            loaded_weight: torch.Tensor) -> torch.Tensor:
        if loaded_weight.dtype != torch.float8_e4m3fn:
            loaded_weight, weight_scale = per_tensor_quantize(loaded_weight)
            # If the loaded weight is not in FP8, we override the
            # weight and scaling factor with the quantized weight.
            layer.weight_scaling_factor.data.copy_(weight_scale)
        return loaded_weight

    def proc_after_loading(self, layer: Module) -> None:
        # Note that torch._scaled_mm requires column-major in
        # the second input (weight), so we transpose the quantized
        # weight here.
        layer.weight.data = layer.weight.data.t()

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        # if layer.fp8_linear_state == Fp8LinearState.UNINITIALIZED:
        #     # Per-tensor scaling on the fly for FP16/BF16 weights
        #     # during the first forward pass if the loaded weights
        #     # are not in FP8.
        #     if layer.weight.dtype != torch.float8_e4m3fn:
        #         qweight, weight_scale = per_tensor_quantize(layer.weight.data)
        #         # Note that torch._scaled_mm requires column-major in
        #         # the second input, so we transpose the quantized weight here.
        #         layer.weight.data = qweight.t()
        #         layer.weight_scaling_factor.data = weight_scale
        #     layer.fp8_linear_state = Fp8LinearState.READY

        qinput, x_scale = per_tensor_quantize(x)
        output, _ = torch._scaled_mm(
            qinput,
            layer.weight,
            out_dtype=x.dtype,
            scale_a=x_scale,
            scale_b=layer.weight_scaling_factor,
            bias=bias,
        )
        return output


#@torch.compile
def per_tensor_quantize(
        tensor: torch.Tensor,
        qdtype=torch.float8_e4m3fn) -> tuple[torch.Tensor, float]:
    """Quantize a tensor using per-tensor static scaling factor.

    Args:
        tensor: The input tensor.
        qdtype: The quantized data type.
    """
    finfo = torch.finfo(qdtype)
    # Calculate the scale as dtype max divided by absmax
    # min_val, max_val = tensor.aminmax()
    # amax = max(-min_val, max_val)
    amax = tensor.abs().max()
    scale = finfo.max / amax.clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    qweight = (tensor * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(qdtype)
    scale = scale.float().reciprocal()
    return qweight, scale
