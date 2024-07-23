import pytest

from .conftest import run_equality_correctness_test


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model": "JackFram/llama-68m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,

        # speculative model
        "speculative_model": "JackFram/llama-160m",

        # num speculative tokens
        "num_speculative_tokens": 3,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("temperature", [0.1, 1.0])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        10,
    ])
@pytest.mark.parametrize("seed", [1])
def test_seeded_consistency(baseline_llm_generator, batch_size: int,
                            temperature: float, output_len: int):
    """Verify outputs are consistent across multiple runs with same seed
    """
    run_equality_correctness_test(baseline_llm_generator,
                                  baseline_llm_generator,
                                  batch_size,
                                  max_output_len=output_len,
                                  temperature=temperature,
                                  seeded=True,
                                  force_output_len=True)
