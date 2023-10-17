import pytest
import torch
import torch.nn as nn
import math
from vllm.model_executor.models.baichuan import NormHead as vllm_NormHead

import torch.distributed as dist
import os
from vllm.model_executor.parallel_utils.parallel_state import initialize_model_parallel
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "8000")
dist.init_process_group(rank=0, world_size=1)
initialize_model_parallel()
torch.cuda.init()


class NormHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((vocab_size, hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.first_flag = True

    def forward(self, hidden_states):
        if self.training:
            norm_weight = nn.functional.normalize(self.weight)
            self.first_flag = True
        elif self.first_flag:
            self.first_flag = False
            self.weight = nn.Parameter(nn.functional.normalize(self.weight))
            norm_weight = self.weight
        else:
            norm_weight = self.weight
        return nn.functional.linear(hidden_states, norm_weight)


@pytest.mark.parametrize("hidden_size", [512, 1024])
@pytest.mark.parametrize("vocab_size", [256, 1256])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16, torch.float])
@torch.inference_mode()
def test_column_blora(
    hidden_size: int,
    vocab_size: int,
    dtype: torch.dtype,
):
    hf_normhead = NormHead(hidden_size, vocab_size, bias=False)
    hf_normhead.eval().cuda()
    vllm_normhead = vllm_NormHead(hidden_size, vocab_size, bias=False)

    # align weights
    hf_normhead.weight.data = hf_normhead.weight.to(dtype)
    vllm_normhead.weight.copy_(hf_normhead.weight)
    assert torch.allclose(vllm_normhead.weight, hf_normhead.weight, atol=1e-8)

    #prepare inputs
    x = torch.randn(1, hidden_size, device="cuda")

    hf_first_output = hf_normhead.forward(x)
    vllm_first_output, _ = vllm_normhead.forward(x)
    assert torch.allclose(hf_first_output, vllm_first_output, atol=1e-8)


    hf_second_output = hf_normhead.forward(x)
    vllm_second_output, _ = vllm_normhead.forward(x)
    assert torch.allclose(hf_second_output, vllm_second_output, atol=1e-8)
