from dataclasses import dataclass
from typing import Optional, List
from xformers.ops.fmha.attn_bias import AttentionBias

import torch


# SANG-TODO Refactor prompt_lens -> seqlens
@dataclass
class InputMetadata:
    """Metadata for input sequences. Used in PagedAttention.

    Both prefill and decoding inputs are mixed up in this class,
    which can be useful to optimize performance.

    If you want metadata specific to prefill or decode,
    Use .prefill_input_metadata() or .decode_input_metadata() API.
    Some of metadata that's not needed for prefill or decode
    could be None.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    # (num_tokens,). The indices of the token slots that input tokens will be
    # stored into. E.g., if `slot_mapping` is [35, 2, 17] and the block size
    # is 16, the three tokens are stored in the 3rd slot in block 2, 2nd slot
    # in block 0, and 1st slot in block 1, respectively.
    slot_mapping: torch.Tensor
    # The number of chunked prefill sequences in the batch.
    num_chunked_prefill: int
    # (batch_size,). The prompt length per sequence. None if it is a decoding.
    prompt_lens: Optional[List]
    # prompt_lens stored as a tensor.
    prompt_lens_tensor: Optional[torch.Tensor]
    # The number of prompt tokens. Doesn't include padding.
    num_prompt_tokens: int
    # The number of generation tokens. Doesn't include padding.
    num_generation_tokens: int
    """
    Definition of context_len, subquery_len, and seqlen.
    |---------- N-1 iteration --------|
    |---------------- N iteration ---------------------|
    |- tokenA -|......................|-- newTokens ---|
    |---------- context_len ----------|
    |-------------------- seqlen ----------------------|
                                      |- subquery_len -|
    
    """

    # Maximum sequence length in the batch.
    max_subquery_len: Optional[int]
    # Maximum context length in the batch.
    max_context_len: Optional[int]
    # Maximum sequence length in the batch.
    max_seq_len: Optional[int]
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    subquery_start_loc: Optional[torch.Tensor]
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]
    # (batch_size,). The length of context (tokens stored in KV cache) per
    # sequence. It doesn't include the length of new tokens.
    context_lens: Optional[torch.Tensor]
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # The first dimension is padded if it is cuda-graph captured.
    block_tables: Optional[torch.Tensor]
    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    use_cuda_graph: bool
    kv_cache_dtype: str

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[AttentionBias]] = None
        # Cuda graph is only used for decoding now.
        if self.use_cuda_graph:
            assert self.num_prompt_tokens == 0

    def prefill_input_metadata(self) -> "InputMetadata":
        """Create a new InputMetadata that only contains
        metadata needed for prefill requests.
        """
        return InputMetadata(
            self.slot_mapping[:self.num_prompt_tokens],
            self.prompt_lens[:self.num_prompts],
            self.num_chunked_prefill,
            self.num_prompt_tokens,
            0,
            # start_loc only contains prompts.
            self.start_loc,
            self.max_seq_len,
            None,
            self.context_lens[:self.num_prompts],
            self.block_tables[:self.num_prompts],
            False,
            self.kv_cache_dtype,
        )

    def decode_input_metadata(self) -> "InputMetadata":
        """Create a new InputMetadata that only contains
        metadata needed for decoding requests.
        """
        return InputMetadata(
            self.slot_mapping[self.num_prompt_tokens:],
            None,
            0,
            0,
            self.num_generation_tokens,
            None,
            None,
            self.max_context_len,
            self.context_lens[self.num_prompts:],
            self.block_tables[self.num_prompts:],
            self.use_cuda_graph,
            self.kv_cache_dtype,
        )
