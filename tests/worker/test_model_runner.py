import pytest
import torch

from vllm.config import ModelConfig, SchedulerConfig
from vllm.distributed.parallel_state import init_distributed_environment
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplingParams, SequenceData, SequenceGroupMetadata
from vllm.utils import get_open_port
from vllm.worker.model_runner import ModelRunner, _get_graph_batch_size


@pytest.mark.parametrize("batch_size", list(range(1, 257)))
def test_prepare_prompt(batch_size):
    scheduler_config = SchedulerConfig(100000,
                                       100000,
                                       100000,
                                       enable_chunked_prefill=False)
    model_runner = ModelRunner(model_config=None,
                               parallel_config=None,
                               scheduler_config=scheduler_config,
                               device_config=None,
                               load_config=None,
                               lora_config=None)
    model_runner.set_block_size(16)

    seqlens = []
    seq_group_metadata_list = []
    block_tables = {0: [1]}
    for i in range(batch_size):
        # make sure all tokens fit into one block
        seqlen = i % (model_runner.block_size - 1) + 1
        seqlens.append(seqlen)
        seq_data = SequenceData(list(range(seqlen)))
        seq_group_metadata = SequenceGroupMetadata(
            request_id=f"test_{i}",
            is_prompt=True,
            seq_data={0: seq_data},
            sampling_params=SamplingParams(temperature=0),
            block_tables=block_tables,
        )
        assert seq_group_metadata.token_chunk_size == seq_data.get_len()
        seq_group_metadata_list.append(seq_group_metadata)

    expected_selected_token_indices = []
    selected_token_start_idx = 0
    for seqlen in seqlens:
        expected_selected_token_indices.append(selected_token_start_idx +
                                               seqlen - 1)
        selected_token_start_idx += seqlen
    (input_tokens, input_positions, attn_metadata, return_seqlens, _, _, _, _,
     _, slot_mapping) = (model_runner._prepare_prompt(seq_group_metadata_list))
    assert return_seqlens == seqlens
    assert len(slot_mapping) == len(input_tokens)

    # Verify input metadata is correct for prompts.
    device = model_runner.device
    assert attn_metadata.is_prompt is True
    assert torch.allclose(
        attn_metadata.seqlens_tensor,
        torch.tensor(seqlens, device=device, dtype=torch.int))
    assert attn_metadata.seqlens == seqlens
    assert attn_metadata.max_seqlen == max(seqlens)

    # Test subquery start locs.
    start_idx = 0
    start_loc = [start_idx]
    for seqlen in seqlens:
        start_idx += seqlen
        start_loc.append(start_idx)
    assert torch.allclose(
        attn_metadata.subquery_start_loc,
        torch.tensor(start_loc, dtype=torch.int32, device=device))

    # Test seq start locs. Note that for normal prefill it is
    # equivalent to subquery_start_loc.
    start_idx = 0
    seq_start_loc = [start_idx]
    for seqlen in seqlens:
        start_idx += seqlen
        seq_start_loc.append(start_idx)

    assert torch.allclose(
        attn_metadata.seq_start_loc,
        torch.tensor(start_loc, dtype=torch.int32, device=device))
    assert torch.allclose(
        attn_metadata.context_lens_tensor,
        torch.zeros(attn_metadata.context_lens_tensor.shape[0],
                    dtype=torch.int,
                    device=device))

    expected = torch.tensor([[] for _ in range(len(seq_group_metadata_list))],
                            dtype=torch.int32,
                            device=model_runner.device)
    assert torch.allclose(attn_metadata.block_tables, expected)
    # Cuda graph should not be used for prerill.
    assert attn_metadata.use_cuda_graph is False

    assert len(input_tokens) == sum(seqlens)
    assert len(input_positions) == sum(seqlens)
    torch.testing.assert_close(input_tokens, input_positions)

    sampling_metadata = SamplingMetadata.prepare(
        seq_group_metadata_list,
        seqlens,
        query_lens=seqlens,
        device=model_runner.device,
        pin_memory=model_runner.pin_memory)
    assert len(input_tokens) == sum(seqlens)
    assert len(input_positions) == sum(seqlens)
    actual = sampling_metadata.selected_token_indices
    expected = torch.tensor(expected_selected_token_indices,
                            device=actual.device,
                            dtype=actual.dtype)
    torch.testing.assert_close(actual, expected)
    assert input_tokens == input_positions

    actual = sampling_metadata.selected_token_indices
    expected = torch.tensor(expected_selected_token_indices,
                            device=actual.device,
                            dtype=actual.dtype)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("batch_size", list(range(1, 257)))
def test_prepare_decode_cuda_graph(batch_size):
    model_config = ModelConfig(
        "facebook/opt-125m",
        "facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
        revision=None,
        enforce_eager=False,
    )
    scheduler_config = SchedulerConfig(100000,
                                       100000,
                                       100000,
                                       enable_chunked_prefill=False)
    model_runner = ModelRunner(model_config=model_config,
                               parallel_config=None,
                               scheduler_config=scheduler_config,
                               device_config=None,
                               load_config=None,
                               lora_config=None)
    model_runner.set_block_size(16)

    seqlens = []
    seq_group_metadata_list = []
    for i in range(batch_size):
        # make sure all tokens fit into one block
        seqlen = i % (model_runner.block_size - 1) + 1
        seqlens.append(seqlen)
        seq_data = list(range(seqlen))
        seq_data = SequenceData(seq_data)
        seq_group_metadata = SequenceGroupMetadata(
            request_id=f"test_{i}",
            is_prompt=False,
            seq_data={0: seq_data},
            sampling_params=SamplingParams(temperature=0),
            block_tables={0: [1]},
        )
        assert seq_group_metadata.token_chunk_size == 1
        seq_group_metadata_list.append(seq_group_metadata)

    input_tokens, input_positions, attn_metadata, _, _, _, slot_mapping = (
        model_runner._prepare_decode(seq_group_metadata_list))
    assert len(slot_mapping) == len(input_tokens)

    expected_bs = _get_graph_batch_size(len(seq_group_metadata_list))
    # Verify input metadata is correct for prompts.
    device = model_runner.device
    assert attn_metadata.is_prompt is False
    assert attn_metadata.seqlens is None
    assert attn_metadata.subquery_start_loc is None
    assert attn_metadata.seq_start_loc is None
    assert attn_metadata.max_seqlen == max(seqlens)
    assert torch.allclose(
        attn_metadata.seqlens_tensor[:len(seqlens)],
        torch.tensor(seqlens, dtype=torch.int, device=device))

    # block table's first index corresponds to each batch, meaning in
    # decoding it is each token.
    assert attn_metadata.block_tables.shape[0] == len(input_tokens)
    # Block table's second dim correspondsd to each token's block number.
    # It is padded up to
    assert attn_metadata.block_tables.shape[1] == (
        model_runner.get_max_block_per_batch())
    # Cuda graph should not be used for prerill.
    assert attn_metadata.use_cuda_graph is True

    assert len(input_tokens) == expected_bs
    assert len(input_positions) == expected_bs
    assert input_tokens == input_positions

    # Verify Sampling
    expected_selected_token_indices = []
    selected_token_start_idx = 0
    for seqlen in seqlens:
        expected_selected_token_indices.append(selected_token_start_idx)
        selected_token_start_idx += 1
    sampling_metadata = SamplingMetadata.prepare(
        seq_group_metadata_list,
        seqlens,
        query_lens=seqlens,
        device=model_runner.device,
        pin_memory=model_runner.pin_memory)
    actual = sampling_metadata.selected_token_indices
    expected = torch.tensor(expected_selected_token_indices,
                            device=actual.device,
                            dtype=actual.dtype)
    torch.testing.assert_close(actual, expected)


def test_empty_seq_group():
    """Verify prepare prompt and decode returns empty output."""
    model_config = ModelConfig(
        "facebook/opt-125m",
        "facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
        revision=None,
        enforce_eager=False,
    )
    model_runner = ModelRunner(model_config=model_config,
                               parallel_config=None,
                               scheduler_config=None,
                               device_config=None,
                               load_config=None,
                               lora_config=None)
    model_runner.set_block_size(16)
    seq_group_metadata_list = []
    input_tokens, input_positions, attn_metadata, _, _, _, slot_mapping = (
        model_runner._prepare_decode(seq_group_metadata_list))
    assert len(input_tokens) == 0
    assert len(input_positions) == 0
    assert attn_metadata is None
    assert len(slot_mapping) == 0

    (input_tokens, input_positions, attn_metadata, return_seqlens, _, _, _, _,
     _, slot_mapping) = (model_runner._prepare_prompt(seq_group_metadata_list))
    assert len(input_tokens) == 0
    assert len(input_positions) == 0
    assert attn_metadata is None
    assert len(slot_mapping) == 0
    assert len(return_seqlens) == 0


@pytest.fixture
def distributed_init():
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        local_rank=0)


@pytest.mark.parametrize("batch_size", list(range(2, 128)))
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_hybrid_batches(batch_size, enforce_eager, distributed_init):

    model_config = ModelConfig(
        "facebook/opt-125m",
        "facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
        revision=None,
        enforce_eager=enforce_eager,
    )
    scheduler_config = SchedulerConfig(100000,
                                       100000,
                                       100000,
                                       enable_chunked_prefill=True)
    model_runner = ModelRunner(model_config=model_config,
                               parallel_config=None,
                               scheduler_config=scheduler_config,
                               device_config=None,
                               load_config=None,
                               lora_config=None,
                               is_driver_worker=True)
    model_runner.set_block_size(16)

    # Add prefill requests.
    seqlens = []
    seq_group_metadata_list = []
    prefill_metadata_list = []
    decode_metadata_list = []
    block_tables = {0: [1]}
    prefill_batch_size = batch_size // 2
    decode_batch_size = batch_size - prefill_batch_size
    for i in range(prefill_batch_size):
        # make sure all tokens fit into one block
        seqlen = i % (model_runner.block_size - 1) + 1
        seqlens.append(seqlen)
        seq_data = SequenceData(list(range(seqlen)))
        seq_group_metadata = SequenceGroupMetadata(
            request_id=f"test_{i}",
            is_prompt=True,
            seq_data={0: seq_data},
            sampling_params=SamplingParams(temperature=0),
            block_tables=block_tables,
        )
        assert seq_group_metadata.token_chunk_size == seq_data.get_len()
        seq_group_metadata_list.append(seq_group_metadata)
        prefill_metadata_list.append(seq_group_metadata)

    # Add decode requests
    for i in range(prefill_batch_size, batch_size):
        # make sure all tokens fit into one block
        seqlen = i % (model_runner.block_size - 1) + 1
        prompt_toks = list(range(seqlen))
        seq_data = SequenceData(prompt_toks)
        seq_group_metadata = SequenceGroupMetadata(
            request_id=f"test_{i}",
            is_prompt=False,
            seq_data={0: seq_data},
            sampling_params=SamplingParams(temperature=0),
            block_tables={0: [1]},
        )
        assert seq_group_metadata.token_chunk_size == 1
        seq_group_metadata_list.append(seq_group_metadata)
        decode_metadata_list.append(seq_group_metadata)

    (input_tokens, input_positions, attn_metadata, _, _, _,
     _) = model_runner.prepare_input_tensors(seq_group_metadata_list)

    prefill_meta_actual = attn_metadata.prefill_metadata
    decode_meta_actual = attn_metadata.decode_metadata

    assert len(attn_metadata.slot_mapping) == len(input_tokens)
    assert len(input_positions) == len(input_tokens)
    assert attn_metadata.kv_cache_dtype == "auto"
    assert attn_metadata.num_prefills == prefill_batch_size
    if enforce_eager:
        assert attn_metadata.num_decode_tokens == decode_batch_size
    else:
        assert attn_metadata.num_decode_tokens == _get_graph_batch_size(
            decode_batch_size)
    assert attn_metadata.num_prefill_tokens == sum(seqlens)

    # Verify attn metadata is consistent. We don't need to test individual
    # values here because they are tested above.
    prefill_meta = model_runner._prepare_prompt(
        prefill_metadata_list).attn_metadata
    decode_meta = model_runner._prepare_decode(
        decode_metadata_list).attn_metadata

    for attr_expected, attr_actual in zip(vars(prefill_meta),
                                          vars(prefill_meta_actual)):
        assert attr_expected[1] == attr_actual[1]
    for attr_expected, attr_actual in zip(vars(decode_meta),
                                          vars(decode_meta_actual)):
        assert attr_expected[1] == attr_actual[1]
