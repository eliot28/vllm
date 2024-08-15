from unittest.mock import patch

import pytest

from tests.quantization.utils import is_quant_method_supported

from ..utils import compare_two_settings


def test_cpu_offload():
    compare_two_settings("meta-llama/Llama-2-7b-hf", [],
                         ["--cpu-offload-gb", "4"])


@pytest.mark.skipif(not is_quant_method_supported("fp8"),
                    reason="fp8 is not supported on this GPU type.")
def test_cpu_offload_fp8():
    # Test quantization of an unquantized checkpoint
    compare_two_settings("meta-llama/Meta-Llama-3-8B-Instruct",
                         ["--quantization", "fp8"],
                         ["--quantization", "fp8", "--cpu-offload-gb", "2"])
    # Test loading a quantized checkpoint
    compare_two_settings("neuralmagic/Meta-Llama-3-8B-Instruct-FP8", [],
                         ["--cpu-offload-gb", "2"])


@pytest.mark.skipif(not is_quant_method_supported("gptq_marlin"),
                    reason="gptq_marlin is not supported on this GPU type.")
def test_cpu_offload_gptq():
    # Test GPTQ Marlin
    compare_two_settings("Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4", [],
                         ["--cpu-offload-gb", "1"])
    # Test GPTQ
    # The model output logits has small variance between runs, which do not play
    # well with the flashinfer sampler.
    with patch.dict("os.environ", {"VLLM_DISABLE_FLASHINFER_SAMPLER": "1"}):
        compare_two_settings(
            "Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4", ["--quantization", "gptq"],
            ["--quantization", "gptq", "--cpu-offload-gb", "1"])


@pytest.mark.skipif(not is_quant_method_supported("awq_marlin"),
                    reason="awq_marlin is not supported on this GPU type.")
def test_cpu_offload_awq():
    # Test AWQ Marlin
    compare_two_settings("Qwen/Qwen2-1.5B-Instruct-AWQ", [],
                         ["--cpu-offload-gb", "1"])
    # Test AWQ
    compare_two_settings("Qwen/Qwen2-1.5B-Instruct-AWQ",
                         ["--quantization", "awq"],
                         ["--quantization", "awq", "--cpu-offload-gb", "1"])


@pytest.mark.skipif(not is_quant_method_supported("gptq_marlin"),
                    reason="gptq_marlin is not supported on this GPU type.")
def test_cpu_offload_compressed_tensors():
    # Test wNa16
    compare_two_settings("nm-testing/tinyllama-oneshot-w4a16-channel-v2", [],
                         ["--cpu-offload-gb", "1"])
    # Test w4a16_marlin24
    compare_two_settings("nm-testing/llama7b-one-shot-2_4-w4a16-marlin24-t",
                         [], ["--cpu-offload-gb", "1"])
    # Test w8a8
    compare_two_settings(
        "nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change", [],
        ["--cpu-offload-gb", "1"])
