"""Utilities for selecting and loading neuron models."""
import contextlib
import importlib
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from vllm.config import DeviceConfig, ModelConfig, LoadConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.ascend_sampler import AscendSampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput


# Models supported by MindIE.
_MindIE_SUPPORTED_MODELS: Dict[str, Tuple[str, str, str]] = {
    # TODO
    "LlamaForCausalLM": ("transformers_neuronx.llama.model",
                         "LlamaForSampling", "LlamaForCausalLM"),
    "MistralForCausalLM": ("transformers_neuronx.mistral.model",
                           "MistralForSampling", "MistralForCausalLM")
}


class MindIECasualLM(nn.Module):

    def __init__(
        self,
        model_config,
        linear_method=None,
        lora_config=None,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.rank = model_config['rank']
        self.local_rank = model_config['local_rank']
        self.npu_id = self.local_rank
        self.world_size = model_config['world_size']
        self.atb_model = None
        self.sampler = None

    def forward(
        self,
        input_ids,
        positions,
        kv_caches,
        attn_metadata,
        intermediate_tensors,
    ) -> torch.Tensor:
        is_prompt = attn_metadata.num_prefill_tokens > 0
        # TODO

        return logits

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        return hidden_states

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        # TODO
        self.sampler = AscendSampler()
        pass

    def create_dummy_kv_cache(self, attn_metadata, input_ids):
        # TODO
        pass


def _get_model_architecture(config) -> str:
    # TODO: 是否可以忽略
    architectures = getattr(config, "architectures", [])
    pass


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    # TODO
    pass


def get_mindie_model(model_config: ModelConfig,
                     device_config: DeviceConfig,
                     load_config: LoadConfig,
                     mindie_model_config, **kwarg) -> nn.Module:
    # TODO
    model = MindIECasualLM(model_config.hf_config)
    return model.eval()


def model_supports_in_mindie(model_config):
    # TODO
    pass