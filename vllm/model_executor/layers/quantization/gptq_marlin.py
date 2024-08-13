from typing import Any, Dict, List, Optional, Union
import enum
from enum import Enum
import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    set_weight_attrs,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.fused_moe_marlin import fused_moe_marlin
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear,
    check_gptq_marlin_supported,
    marlin_is_k_full,
    marlin_make_empty_g_idx,
    marlin_make_workspace,
    marlin_permute_scales,
    marlin_moe_permute_scales,
    marlin_repeat_scales_on_all_ranks,
    marlin_sort_g_idx,
    replace_tensor,
    verify_gptq_marlin_supported,
    verify_marlin_supports_shape,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

logger = init_logger(__name__)


class GPTQMarlinState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


class GPTQMarlinConfig(QuantizationConfig):
    """Config class for GPTQ Marlin"""

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        is_sym: bool,
        lm_head_quantized: bool,
    ) -> None:
        if desc_act and group_size == -1:
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            desc_act = False

        self.weight_bits = weight_bits
        self.pack_factor = 32 // self.weight_bits  # packed into int32
        self.group_size = group_size
        self.desc_act = desc_act
        self.is_sym = is_sym
        self.lm_head_quantized = lm_head_quantized

        # Verify supported on platform.
        verify_gptq_marlin_supported(num_bits=self.weight_bits,
                                     group_size=self.group_size,
                                     is_sym=self.is_sym)

    def __repr__(self) -> str:
        return (f"GPTQMarlinConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"desc_act={self.desc_act}, "
                f"lm_head_quantized={self.lm_head_quantized})")

    @classmethod
    def get_name(cls) -> str:
        return "gptq_marlin"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQMarlinConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        is_sym = cls.get_from_keys(config, ["sym"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"],
                                                 default=False)
        return cls(weight_bits, group_size, desc_act, is_sym,
                   lm_head_quantized)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        can_convert = cls.is_gptq_marlin_compatible(hf_quant_cfg)

        is_valid_user_quant = (user_quant is None or user_quant == "marlin"
                               or user_quant == "gptq_marlin")

        if can_convert and is_valid_user_quant:
            msg = ("The model is convertible to {} during runtime."
                   " Using {} kernel.".format(cls.get_name(), cls.get_name()))
            logger.info(msg)
            return cls.get_name()

        if can_convert and user_quant == "gptq":
            logger.info("Detected that the model can run with gptq_marlin"
                        ", however you specified quantization=gptq explicitly,"
                        " so forcing gptq. Use quantization=gptq_marlin for"
                        " faster inference")
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[Union["GPTQMarlinLinearMethod", "GPTQMarlinMoEMethod"]]:
        if isinstance(layer, LinearBase) or (isinstance(layer, ParallelLMHead)
                                             and self.lm_head_quantized):
            return GPTQMarlinLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return GPTQMarlinMoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def is_gptq_marlin_compatible(cls, quant_config: Dict[str, Any]):
        # Extract data from quant config.
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits", None)
        group_size = quant_config.get("group_size", None)
        sym = quant_config.get("sym", None)
        desc_act = quant_config.get("desc_act", None)

        if quant_method != "gptq":
            return False

        # If we cannot find the info needed in the config, cannot convert.
        if num_bits is None or group_size is None or sym is None or desc_act is None:
            return False

        return check_gptq_marlin_supported(
            num_bits=num_bits,
            group_size=group_size,
            is_sym=sym,
            min_capability=cls.get_min_capability(),
        )


class GPTQMarlinLinearMethod(LinearMethodBase):
    """Linear method for GPTQ Marlin.

    Args:
        quant_config: The GPTQ Marlin quantization config.
    """

    def __init__(self, quant_config: GPTQMarlinConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del output_size
        output_size_per_partition = sum(output_partition_sizes)
        is_row_parallel = input_size != input_size_per_partition

        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        verify_marlin_supports_shape(
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            input_size=input_size,
            group_size=group_size,
        )

        # Determine sharding
        if marlin_repeat_scales_on_all_ranks(self.quant_config.desc_act,
                                             self.quant_config.group_size,
                                             is_row_parallel):
            # By setting scale_dim == None, weight_loader will
            # repeat the scales on each GPU in TP>1 case.
            scales_and_zp_input_dim = None
            scales_and_zp_size = input_size // group_size
        else:
            # By setting scale_dim == 0, weight_loader will
            # shard the scales in TP>1 case.
            scales_and_zp_input_dim = 0
            scales_and_zp_size = input_size_per_partition // group_size

        # Quantized weights
        qweight = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                **extra_weight_attrs,
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 0,
                "pack_factor": self.quant_config.pack_factor,
            },
        )

        # Activation order
        g_idx = Parameter(
            torch.empty(
                input_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        # Ignore warning from fused linear layers such as QKVParallelLinear.
        set_weight_attrs(
            g_idx,
            {
                **extra_weight_attrs, "input_dim": 0,
                "ignore_warning": True
            },
        )

        # Scales
        scales = Parameter(
            torch.empty(
                scales_and_zp_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                **extra_weight_attrs,
                "input_dim": scales_and_zp_input_dim,
                "output_dim": 1,
            },
        )

        # Quantized zero-points
        qzeros = Parameter(
            torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
                device="meta",
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros,
            {
                **extra_weight_attrs,
                "input_dim": scales_and_zp_input_dim,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            },
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.input_size = input_size
        layer.is_k_full = marlin_is_k_full(self.quant_config.desc_act,
                                           is_row_parallel)

    # Checkpoints are serialized in AutoGPTQ format, which is different from the
    # marlin format. This function is called after the weights are loaded.
    # Here, we handle the repacking, including the activation reordering case.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.qweight.device

        # Allocate marlin workspace
        layer.workspace = marlin_make_workspace(
            layer.output_size_per_partition, device)

        # Handle sorting for activation reordering if needed.
        if self.quant_config.desc_act:
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(layer.g_idx)
            layer.g_idx_sort_indices = g_idx_sort_indices
            replace_tensor(layer, "g_idx", g_idx)
        else:
            layer.g_idx = marlin_make_empty_g_idx(device)
            layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        # No zero-point
        layer.zp = marlin_make_empty_g_idx(device)

        # Repack weights from autogptq format to marlin format.
        marlin_qweight = ops.gptq_marlin_repack(
            layer.qweight,
            perm=layer.g_idx_sort_indices,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            num_bits=self.quant_config.weight_bits,
        )
        replace_tensor(layer, "qweight", marlin_qweight)

        # Permute scales from autogptq format to marlin format.
        marlin_scales = marlin_permute_scales(
            layer.scales,
            size_k=(layer.input_size if self.quant_config.desc_act else
                    layer.input_size_per_partition),
            size_n=layer.output_size_per_partition,
            group_size=self.quant_config.group_size,
        )
        replace_tensor(layer, "scales", marlin_scales)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return apply_gptq_marlin_linear(
            input=x,
            weight=layer.qweight,
            weight_scale=layer.scales,
            weight_zp=layer.zp,
            g_idx=layer.g_idx,
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=layer.workspace,
            num_bits=self.quant_config.weight_bits,
            output_size_per_partition=layer.output_size_per_partition,
            input_size_per_partition=layer.input_size_per_partition,
            is_k_full=layer.is_k_full,
            bias=bias,
        )


class GPTQMarlinMoEMethod(FusedMoEMethodBase):
    """MoE Marlin method with quantization."""

    def __init__(self, quant_config: GPTQMarlinConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Currently assuming is_k_full is always True
        # (input size per partition is the same as full input size)
        # Supports only sym for now (no zp)
        if self.quant_config.group_size != -1:
            scales_size13 = hidden_size // self.quant_config.group_size
            scales_size2 = intermediate_size // self.quant_config.group_size
        else:
            scales_size13 = 1
            scales_size2 = 1
        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.quant_config.pack_factor,
                2 * intermediate_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size // self.quant_config.pack_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        # up_proj scales
        w13_scales = torch.nn.Parameter(
            torch.empty(num_experts,
                        scales_size13,
                        2 * intermediate_size,
                        dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)
        # down_proj scales
        w2_scales = torch.nn.Parameter(
            torch.empty(num_experts,
                        scales_size2,
                        hidden_size,
                        dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)
        # up_proj scales
        w13_qzeros = torch.nn.Parameter(
            torch.empty(num_experts,
                        scales_size13,
                        2 * intermediate_size // self.quant_config.pack_factor,
                        dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)
        # down_proj scales
        w2_qzeros = torch.nn.Parameter(
            torch.empty(num_experts,
                        scales_size2,
                        hidden_size // self.quant_config.pack_factor,
                        dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)
        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)
        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)
        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices",
                                 w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)
        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices",
                                 w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)
        layer.marlin_state = GPTQMarlinState.REPACK

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.marlin_state = GPTQMarlinState.READY

        # Process act_order
        if self.quant_config.desc_act:
            # Get sorting based on g_idx
            num_experts = layer.w13_g_idx.shape[0]
            w13_g_idx_sort_indices = torch.empty_like(layer.w13_g_idx)
            w2_g_idx_sort_indices = torch.empty_like(layer.w2_g_idx)
            w13_sorted_g_idx = torch.empty_like(layer.w13_g_idx)
            w2_sorted_g_idx = torch.empty_like(layer.w2_g_idx)
            for e in range(num_experts):
                w13_g_idx_sort_indices[e] = torch.argsort(
                    layer.w13_g_idx[e]).to(torch.int32)
                w2_g_idx_sort_indices[e] = torch.argsort(layer.w2_g_idx[e]).to(
                    torch.int32)
                w13_sorted_g_idx[e] = layer.w13_g_idx[e][
                    w13_g_idx_sort_indices[e]]
                w2_sorted_g_idx[e] = layer.w2_g_idx[e][
                    w2_g_idx_sort_indices[e]]
            replace_tensor(layer, "w13_g_idx", w13_sorted_g_idx)
            replace_tensor(layer, "w2_g_idx", w2_sorted_g_idx)
            replace_tensor(layer, "w13_g_idx_sort_indices",
                           w13_g_idx_sort_indices)
            replace_tensor(layer, "w2_g_idx_sort_indices",
                           w2_g_idx_sort_indices)
        else:
            # Reset g_idx related tensors
            num_experts = layer.w13_g_idx.shape[0]
            device = layer.w13_g_idx.device
            layer.w13_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32,
                            device=device),
                requires_grad=False,
            )
            layer.w2_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32,
                            device=device),
                requires_grad=False,
            )
            layer.w13_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32,
                            device=device),
                requires_grad=False,
            )
            layer.w2_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32,
                            device=device),
                requires_grad=False,
            )
        # Repack weights
        # marlin_w13_qweight = ops.gptq_marlin_moe_repack(
        #     layer.w13_qweight,
        #     layer.w13_g_idx_sort_indices,
        #     layer.w13_qweight.shape[1] * self.quant_config.pack_factor,
        #     layer.w13_qweight.shape[2],
        #     self.quant_config.weight_bits,
        # )
        # replace_tensor(layer, "w13_qweight", marlin_w13_qweight)
        marlin_w2_qweight = ops.gptq_marlin_moe_repack(
            layer.w2_qweight,
            layer.w2_g_idx_sort_indices,
            layer.w2_qweight.shape[1] * self.quant_config.pack_factor,
            layer.w2_qweight.shape[2],
            self.quant_config.weight_bits,
        )
        replace_tensor(layer, "w2_qweight", marlin_w2_qweight)
        # Repack scales
        marlin_w13_scales = marlin_moe_permute_scales(
            s=layer.w13_scales,
            size_k=(layer.intermediate_size if self.quant_config.desc_act else
                    layer.intermediate_size_per_partition),
            size_n=layer.w13_scales.shape[2],
            group_size=self.quant_config.group_size,
        )
        replace_tensor(layer, "w13_scales", marlin_w13_scales)
        logger.error(f"{layer.w2_scales.size()}, {layer.intermediate_size_per_partition}, {self.quant_config.group_size}")
        marlin_w2_scales = marlin_moe_permute_scales(
            s=layer.w2_scales,
            size_k=layer.w2_scales.shape[1] * self.quant_config.pack_factor,
            size_n=layer.w2_scales.shape[2],
            group_size=self.quant_config.group_size,
        )
        replace_tensor(layer, "w2_scales", marlin_w2_scales)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
    ) -> torch.Tensor:
        return fused_moe_marlin(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            router_logits,
            layer.w13_g_idx,
            layer.w2_g_idx,
            layer.w13_g_idx_sort_indices,
            layer.w2_g_idx_sort_indices,
            top_k,
            renormalize=renormalize,
            w1_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
        )
