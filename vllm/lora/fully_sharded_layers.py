# pylint: disable=unused-argument
from typing import TYPE_CHECKING, Optional, List

import torch

from vllm.lora.punica import bgmv, dispatch_bgmv_low_level
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank)
from vllm.lora.layers import (ColumnParallelLinearWithLoRA,
                              MergedColumnParallelLinearWithLoRA,
                              QKVParallelLinearWithLora,
                              RowParallelLinearWithLoRA)

if TYPE_CHECKING:
    pass


class ColumnParallelLinearWithShardedLoRA(ColumnParallelLinearWithLoRA):
    """
    Differs from ColumnParallelLinearWithLoRA by slicing LoRA A also.

    Based on S-LoRA, slicing happens along the rank dim.
    """

    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = self.lora_a_stacked.shape[2]
        start_idx = tp_rank * shard_size
        lora_a = lora_a[:, start_idx:start_idx + shard_size]
        return lora_a

    def apply_weights(self, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias)

        x = x.view(-1, x.shape[-1])
        output, out_orig_shape = output.view(-1,
                                             output.shape[-1]), output.shape
        buffer = torch.zeros((x.shape[0], self.lora_a_stacked.shape[2]),
                             dtype=torch.float32,
                             device=x.device)

        bgmv(buffer, x, self.lora_a_stacked,
             self.indices[:self.indices_len[0]], 0, 1.0)
        buffer = tensor_model_parallel_all_gather(buffer)
        bgmv(output, buffer, self.lora_b_stacked,
             self.indices[:self.indices_len[0]], 0, 1.0)
        # now have column partitioned output

        output = output.view(*out_orig_shape)
        return output


def _mcp_apply_weights(x, bias, layer):
    """
    MergedColumnParallelLinearWithShardedLoRA and 
    QKVParallelLinearWithShardedLora share the same 
    LoRa weight application method.
    
    The main difference is the step by shard_size for lora_b which can
    vary for QKVParallelLinearWithShardedLora but is constant for 
    MergedColumnParallelLinearWithShardedLoRA.
    """
    n = len(
        layer.lora_a_stacked)  # expecting 2 for column parallel and 3 for qkv
    output = layer.base_layer.linear_method.apply_weights(
        layer.base_layer.linear_weights, x, bias)

    x = x.view(-1, x.shape[-1])
    output, out_orig_shape = output.view(-1, output.shape[-1]), output.shape
    buffers = torch.zeros((n, x.shape[0], layer.lora_a_stacked[0].shape[2]),
                          dtype=torch.float32,
                          device=x.device)
    for idx in range(n):
        bgmv(buffers[idx], x, layer.lora_a_stacked[idx],
             layer.indices[:layer.indices_len[0]], 0, 1.0)

    buffers = tensor_model_parallel_all_gather(buffers)
    left_offset = 0
    for idx in range(n):
        shard_size = layer.lora_b_stacked[idx].shape[2]
        dispatch_bgmv_low_level(output, buffers[idx],
                                layer.lora_b_stacked[idx],
                                layer.indices[:layer.indices_len[0]], 0, 1.0,
                                left_offset, shard_size)
        left_offset += shard_size

    output = output.view(*out_orig_shape)
    # now have column partitioned and packed output
    return output


class MergedColumnParallelLinearWithShardedLoRA(
        MergedColumnParallelLinearWithLoRA):
    """
    Differs from MergedColumnParallelLinearWithLoRA by slicing the 
    LoRA A's also.

    Based on S-LoRA, slicing happens along the rank dim.
    """

    def slice_lora_a(self, lora_a: List[torch.Tensor]) -> List[torch.Tensor]:
        output_shard_size = self.lora_a_stacked[0].shape[2]
        output_start_idx = self.tp_rank * output_shard_size
        lora_a = [
            lora_a[i][:, output_start_idx:output_start_idx + output_shard_size]
            for i in range(2)
        ]
        return lora_a

    def apply_weights(self, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        return _mcp_apply_weights(x, bias, self)


class QKVParallelLinearWithShardedLora(QKVParallelLinearWithLora):
    """
    Differs from QKVParallelLinearWithLora by slicing the 
    LoRA A's also.

    Based on S-LoRA, slicing happens along the rank dim.
    """

    def slice_lora_a(self, lora_a: List[torch.Tensor]) -> List[torch.Tensor]:
        shard_size = [self.lora_a_stacked[i].shape[2] for i in range(3)]
        start_idx = [self.tp_rank * shard_size[i] for i in range(3)]
        lora_a = [
            lora_a[i][:, start_idx[i]:start_idx[i] +
                      shard_size[i]] if lora_a[i] is not None else None
            for i in range(3)
        ]
        return lora_a

    def apply_weights(self, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        return _mcp_apply_weights(x, bias, self)


class RowParallelLinearWithShardedLoRA(RowParallelLinearWithLoRA):
    """
    Differs from RowParallelLinearWithLoRA by slicing the 
    LoRA B's also.

    Based on S-LoRA, slicing happens along the output dim.
    This yields a combined partial sum from the row parallel base 
    layer and column partitioned output from the LoRA.
    """

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        shard_size = self.lora_b_stacked.shape[2]
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size
        lora_b = lora_b[:, start_idx:end_idx]
        return lora_b

    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x)

        x = x.view(-1, x.shape[-1])
        output, out_orig_shape = output.view(-1,
                                             output.shape[-1]), output.shape
        buffer = torch.zeros((x.shape[0], self.lora_a_stacked.shape[2]),
                             dtype=torch.float32,
                             device=x.device)
        bgmv(buffer, x, self.lora_a_stacked,
             self.indices[:self.indices_len[0]], 0, 1.0)
        buffer = tensor_model_parallel_all_reduce(buffer)

        # following S-LoRA, allows the fusing of all_gather and all_reduce
        # by adding the column partitioned lora output to a slice of output
        # tensor. All that remains is a standard all_reduce. User should
        # be aware though that the output is not the same as a normal
        # row_parallel, it should be reduced before being used
        shard_size = self.lora_b_stacked.shape[2]
        start_idx = self.tp_rank * shard_size
        dispatch_bgmv_low_level(output, buffer, self.lora_b_stacked,
                                self.indices[:self.indices_len[0]], 0, 1.0,
                                start_idx, shard_size)

        output = output.view(*out_orig_shape)
        return output
