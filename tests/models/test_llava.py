from typing import List, Tuple

import pytest
from transformers import AutoTokenizer

from vllm.config import VisionLanguageConfig

from ..conftest import IMAGE_ASSETS

pytestmark = pytest.mark.vlm

# The image token is placed before "user" on purpose so that the test can pass
HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "<image>\nUSER: What's the content of the image?\nASSISTANT:",
    "cherry_blossom":
    "<image>\nUSER: What is the season?\nASSISTANT:",
})


def iter_llava_configs(model_name: str):
    image_hw_to_feature_size = {
        (336, 336): 576,
    }

    for (h, w), f in image_hw_to_feature_size.items():
        input_shape = (1, 3, h, w)
        yield (model_name,
               VisionLanguageConfig(image_input_type=None,
                                    image_feature_size=f,
                                    image_token_id=32000,
                                    image_input_shape=input_shape))


model_and_vl_config = [
    *iter_llava_configs("llava-hf/llava-1.5-7b-hf"),
]


def vllm_to_hf_output(vllm_output: Tuple[List[int], str],
                      vlm_config: VisionLanguageConfig, model_id: str):
    """Sanitize vllm output to be comparable with hf output.
    The function reduces `input_ids` from 1, 32000, 32000, ..., 32000,
    x1, x2, x3 ... to 1, 32000, x1, x2, x3 ...
    It also reduces `output_str` from "<image><image>bla" to "bla".
    """
    output_ids, output_str = vllm_output
    image_token_id = vlm_config.image_token_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    image_token_str = tokenizer.decode(image_token_id)

    hf_output_ids = [
        token_id for idx, token_id in enumerate(output_ids)
        if token_id != image_token_id or output_ids[idx - 1] != image_token_id
    ]
    hf_output_str = output_str \
        .replace(image_token_str * vlm_config.image_feature_size, "")

    return hf_output_ids, hf_output_str


# TODO: Add test for `tensor_parallel_size` [ref: PR #3883]
@pytest.mark.parametrize("model_and_config", model_and_vl_config)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
def test_models(hf_runner, vllm_runner, image_assets, model_and_config,
                dtype: str, max_tokens: int) -> None:
    """Inference result should be the same between hf and vllm.

    All the image fixtures for the test is under tests/images.
    For huggingface runner, we provide the PIL images as input.
    For vllm runner, we provide MultiModalData objects and corresponding
    vision language config as input.
    Note, the text input is also adjusted to abide by vllm contract.
    The text output is sanitized to be able to compare with hf.
    """
    model_id, vlm_config = model_and_config
    hf_images = [asset.for_hf() for asset in image_assets]
    vllm_images = [asset.for_vllm() for asset in image_assets]

    with hf_runner(model_id, dtype=dtype, is_vision_model=True) as hf_model:
        hf_outputs = hf_model.generate_greedy(HF_IMAGE_PROMPTS,
                                              max_tokens,
                                              images=hf_images)

    vllm_image_prompts = [
        p.replace("<image>", "<image>" * vlm_config.image_feature_size)
        for p in HF_IMAGE_PROMPTS
    ]

    with vllm_runner(model_id,
                     dtype=dtype,
                     enforce_eager=True,
                     **vlm_config.as_cli_args_dict()) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(vllm_image_prompts,
                                                  max_tokens,
                                                  images=vllm_images)

    for i in range(len(HF_IMAGE_PROMPTS)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_to_hf_output(
            vllm_outputs[i], vlm_config, model_id)
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")
