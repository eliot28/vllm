# coding=utf-8

"""Inference-only Jurassic model."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import conv_transpose3d, nn
import os
from vllm.model_executor.mamba_metadata import MambaCacheParams

from vllm.transformers_utils.configs.jamba import JambaConfig
from transformers.activations import ACT2FN
from vllm.config import LoRAConfig
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead, DEFAULT_VOCAB_PADDING_SIZE)
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

KVCache = Tuple[torch.Tensor, torch.Tensor]

# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
class JambaMambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: JambaConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_expand * config.hidden_size
        self.time_step_rank = config.mamba_dt_rank
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=self.conv_kernel_size - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.apply_inner_layernorms = config.mamba_inner_layernorms

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=self.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.use_bias)

        if self.apply_inner_layernorms:
            self.dt_layernorm = RMSNorm(self.time_step_rank, eps=config.rms_norm_eps)
            self.B_layernorm = RMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
            self.C_layernorm = RMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm.forward(dt.contiguous())
        if self.B_layernorm is not None:
            B = self.B_layernorm.forward(B.contiguous())
        if self.C_layernorm is not None:
            C = self.C_layernorm.forward(C.contiguous())
        return dt, B, C

    def mamba_forward(self, hidden_states: torch.Tensor, cache_params: MambaCacheParams = None):
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        hidden_states, gate = projected_states.chunk(2, dim=1)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if cache_params is not None and not cache_params.is_prompt:
            hidden_states = causal_conv1d_update(
                hidden_states.squeeze(-1),
                cache_params.conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
            )
            hidden_states = hidden_states.unsqueeze(-1)
        else:
            if cache_params is not None:
                conv_states = nn.functional.pad(
                    hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_state.copy_(conv_states)
            hidden_states = causal_conv1d_fn(
                hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
            )

        # 3. State Space Model sequence transformation
        # 3.a. input varying initialization of time_step, B and C
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        time_step, B, C = self._apply_layernorms(time_step, B, C)

        # Here we need to apply dt_proj without the bias, as the bias is added in the selective scan kernel.
        # This is a hack to apply dt_proj while still using the forward pass of `torch.nn.Linear`, which is needed
        # in order to make quantization work. Quantization code replaces `torch.nn.Linear` layers with quantized
        # linear layers, and requires to call the forward pass directly.
        # The original code here was: ```discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)```
        dt_proj_bias = self.dt_proj.bias
        self.dt_proj.bias = None
        discrete_time_step = self.dt_proj(time_step).transpose(1, 2)
        self.dt_proj.bias = dt_proj_bias

        A = -torch.exp(self.A_log.float())
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None
        if cache_params is not None and not cache_params.is_prompt:
            scan_outputs = selective_state_update(
                cache_params.ssm_state,
                hidden_states[..., 0],
                discrete_time_step[..., 0],
                A,
                B[:, 0],
                C[:, 0],
                self.D,
                gate[..., 0],
                time_proj_bias,
                dt_softplus=True,
            ).unsqueeze(-1)
        else:
            scan_outputs, ssm_state = selective_scan_fn(
                hidden_states,
                discrete_time_step,
                A,
                B.transpose(1, 2),
                C.transpose(1, 2),
                self.D.float(),
                gate,
                time_proj_bias,
                delta_softplus=True,
                return_last_state=True,
            )
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_state.copy_(ssm_state)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
        return contextualized_states

    def forward(self, hidden_states: torch.Tensor, input_metadata: InputMetadata, conv_state: torch.Tensor, ssm_state: torch.Tensor):
        cache = MambaCacheParams(
            input_metadata.is_prompt, 
            conv_state=conv_state[self.layer_idx],
            ssm_state=ssm_state[self.layer_idx]
        )
        hidden_states = self.mamba_forward(hidden_states, cache_params=cache)

        return hidden_states




class JambaMoE(nn.Module):
    """A tensor-parallel MoE implementation for Mixtral that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
            self,
            num_experts: int,
            top_k: int,
            hidden_size: int,
            intermediate_size: int,
            params_dtype: Optional[torch.dtype] = None,
            tp_size: Optional[int] = None,
    ):
        super().__init__()
        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.num_total_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size // self.tp_size

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        if self.num_total_experts > 1:
            #   init expert router iff this layer has multiple experts
            self.router = ReplicatedLinear(self.hidden_size,
                                           self.num_total_experts,
                                           bias=False,
                                           params_dtype=self.params_dtype,
                                           linear_method=None)

        self.ws = nn.Parameter(
            torch.empty(self.num_total_experts,
                        2 * self.intermediate_size,
                        self.hidden_size,
                        device="cuda",
                        dtype=self.params_dtype))
        self.w2s = nn.Parameter(
            torch.empty(self.num_total_experts,
                        self.hidden_size,
                        self.intermediate_size,
                        device="cuda",
                        dtype=self.params_dtype))

        set_weight_attrs(self.ws, {
            "weight_loader": self.weight_loader,
        })
        set_weight_attrs(self.w2s, {
            "weight_loader": self.weight_loader,
        })

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      weight_name: str, expert_id: int):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        if weight_name.endswith("gate_proj.weight"):
            param_data[expert_id, 0:shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith("up_proj.weight"):
            param_data[expert_id, shard_size:2 * shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith("down_proj.weight"):
            param_data[expert_id, :, :] = loaded_weight[:, shard]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (batch * sequence_length, n_experts)
        if self.num_total_experts > 1:
            router_logits, _ = self.router(hidden_states)
        else:
            router_logits = torch.ones([hidden_states.shape[0], 1], device=hidden_states.device,
                                       dtype=hidden_states.dtype)

        final_hidden_states = fused_moe(hidden_states,
                                        self.ws,
                                        self.w2s,
                                        router_logits,
                                        self.top_k,
                                        renormalize=False,  # Mixtral normalize the expert probs to 1. We don't!
                                        inplace=True)

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states.view(batch_size, sequence_length,
                                        hidden_size)


class JambaMambaDecoderLayer(nn.Module):
    def __init__(self, config: JambaConfig, actual_num_experts: int, actual_num_experts_per_tok: int ,layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.mamba = JambaMambaMixer(config, layer_idx)
        self.moe = JambaMoE(
            num_experts=actual_num_experts,
            top_k=actual_num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.pre_moe_layernorm = RMSNorm(config.hidden_size,
                                         eps=config.rms_norm_eps)

    def forward(self,
                hidden_states: torch.Tensor,
                input_metadata: InputMetadata,
                residual: Optional[torch.Tensor],
                conv_state: torch.Tensor,
                ssm_state: torch.Tensor,
                **kwargs):

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.mamba(
            hidden_states,
            input_metadata,
            conv_state,
            ssm_state
        )
        # Fully Connected
        hidden_states, residual = self.pre_moe_layernorm(
            hidden_states, residual)
        hidden_states = self.moe(hidden_states)
        return hidden_states, residual


class JambaAttentionDecoderLayer(nn.Module):
    def __init__(
            self, config: JambaConfig, actual_num_experts: int, actual_num_experts_per_tok: int ,layer_idx: int, linear_method: Optional[LinearMethodBase] = None
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.use_positional_embeddings = False
        self.sliding_window = config.sliding_window

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            linear_method=linear_method,
        )

        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            sliding_window=self.sliding_window,
        )


        self.moe = JambaMoE(
            num_experts=actual_num_experts,
            top_k=actual_num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.pre_moe_layernorm = RMSNorm(config.hidden_size,
                                         eps=config.rms_norm_eps)

    def self_attention(self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: KVCache,
            input_metadata: InputMetadata,
            **kwargs) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        #   TODO - add embedding flag
        if self.use_positional_embeddings:
            q, k = self.rotary_emb(positions, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: KVCache,
            input_metadata: InputMetadata,
            residual: Optional[torch.Tensor],
            **kwargs):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attention(
                positions=positions,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                input_metadata=input_metadata,
        )
        # Fully Connected
        hidden_states, residual = self.pre_moe_layernorm(
            hidden_states, residual)
        hidden_states = self.moe(hidden_states)
        return hidden_states, residual


class JambaModel(nn.Module):
    def __init__(
            self,
            config: JambaConfig,
            linear_method: Optional[LinearMethodBase] = None,
            lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        #   init each model layer, decide if it's mamba/attention and has experts and pass it down

        module_list = []
        for i in range(config.num_hidden_layers):
            is_attn = True if (i - self.config.attn_layer_offset) % self.config.attn_layer_period == 0 else False
            is_expert = True if (i - self.config.expert_layer_offset) % self.config.expert_layer_period == 0 else False

            actual_num_experts = config.num_experts if is_expert else 1
            actual_num_experts_per_tok = config.num_experts_per_tok if is_expert else 1

            if is_attn:
                module_list.append(JambaAttentionDecoderLayer(config,
                                                              actual_num_experts=actual_num_experts,
                                                              actual_num_experts_per_tok=actual_num_experts_per_tok,
                                                              layer_idx=i,
                                                              linear_method=linear_method
                                                              ))
            else:
                module_list.append(JambaMambaDecoderLayer(config,
                                                          actual_num_experts=actual_num_experts,
                                                          actual_num_experts_per_tok=actual_num_experts_per_tok,
                                                          layer_idx=i
                                                          ))

        self.layers = nn.ModuleList(module_list)
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[KVCache],
            input_metadata: InputMetadata,
            conv_state: torch.Tensor,
            ssm_state: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]

            hidden_states, residual = layer(positions=positions,
                                            hidden_states=hidden_states,
                                            kv_cache=kv_caches[i],
                                            input_metadata=input_metadata,
                                            residual=residual,
                                            conv_state=conv_state,
                                            ssm_state=ssm_state
                                            )
        hidden_states, _ = self.final_layernorm(hidden_states, residual)
        return hidden_states


class JambaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "embed_tokens",
        "lm_head",
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(
            self,
            config: JambaConfig,
            linear_method: Optional[LinearMethodBase] = None,
            lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = JambaModel(config,
                                linear_method,
                                lora_config=lora_config)
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
        )
        self.sampler = Sampler(self.unpadded_vocab_size, config.vocab_size)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[KVCache],
            input_metadata: InputMetadata,
            conv_state: torch.Tensor,
            ssm_state: torch.Tensor
        ):
        hidden_states = self.model(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            conv_state,
            ssm_state
        )
        return hidden_states

    def sample(
            self,
            hidden_states: Optional[torch.Tensor],
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        expert_params_mapping = [
            # (param_name, weight_name, expert_id)
            ("ws" if weight_name in ["gate_proj", "up_proj"] else "w2s",
             f"experts.{expert_id}.{weight_name}.weight", expert_id)
            for expert_id in range(self.config.num_experts)
            for weight_name in ["down_proj", "up_proj", "gate_proj"]
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path,
                cache_dir,
                load_format,
                revision,
                fall_back_to_pt=True):  # erez - might need to change later to False
            if "rotary_emb.inv_freq" in name:
                continue

            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for param_name, weight_name, expert_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  weight_name,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
