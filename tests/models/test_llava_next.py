from typing import List, Optional, Tuple, Type, overload

import pytest
from transformers import AutoConfig, AutoTokenizer

from vllm.multimodal.utils import rescale_image_size
from vllm.sequence import SampleLogprobs

from ..conftest import (IMAGE_ASSETS, HfRunner, VllmRunner, _ImageAssets,
                        PromptImageInput)
from .utils import check_logprobs_close

pytestmark = pytest.mark.vlm

_LIMIT_IMAGE_PER_PROMPT = 4

_PREFACE = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's "
    "questions.")

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    f"{_PREFACE} USER: <image>\nWhat's the content of the image? ASSISTANT:",
    "cherry_blossom":
    f"{_PREFACE} USER: <image>\nWhat is the season? ASSISTANT:",
})

models = ["llava-hf/llava-v1.6-mistral-7b-hf"]


def vllm_to_hf_output(vllm_output: Tuple[List[int], str,
                                         Optional[SampleLogprobs]],
                      model: str):
    """Sanitize vllm output to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    config = AutoConfig.from_pretrained(model)
    image_token_id = config.image_token_index

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = [
        token_id for idx, token_id in enumerate(output_ids)
        if token_id != image_token_id or output_ids[idx - 1] != image_token_id
    ]

    assert output_str[0] == " "
    hf_output_str = output_str[1:]
    if hf_output_ids[-1] == eos_token_id:
        hf_output_str = hf_output_str + tokenizer.decode(eos_token_id)

    return hf_output_ids, hf_output_str, out_logprobs


@overload
def run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    image_assets: _ImageAssets,
    model: str,
    *,
    size_factors: List[float],
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    ...


@overload
def run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    image_assets: _ImageAssets,
    model: str,
    *,
    sizes: List[Tuple[int, int]],
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    ...


def run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    image_assets: _ImageAssets,
    model: str,
    *,
    size_factors: Optional[List[float]] = None,
    sizes: Optional[List[Tuple[int, int]]] = None,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    images = [asset.pil_image for asset in image_assets]

    if size_factors is not None:
        inputs_per_image = [(
            [prompt for _ in size_factors],
            [rescale_image_size(image, factor) for factor in size_factors],
        ) for image, prompt in zip(images, HF_IMAGE_PROMPTS)]
    elif sizes is not None:
        inputs_per_image = [(
            [prompt for _ in sizes],
            [image.resize(size) for size in sizes],
        ) for image, prompt in zip(images, HF_IMAGE_PROMPTS)]
    else:
        raise ValueError("You must provide either `size_factors` or `sizes`")

    _run_test(hf_runner,
              vllm_runner,
              inputs_per_image,
              model,
              dtype=dtype,
              max_tokens=max_tokens,
              num_logprobs=num_logprobs,
              tensor_parallel_size=tensor_parallel_size,
              distributed_executor_backend=distributed_executor_backend)


def _run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    inputs: List[Tuple[List[str], PromptImageInput]],
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    # max_model_len should be greater than image_feature_size
    with vllm_runner(model,
                     dtype=dtype,
                     max_model_len=32768,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True,
                     limit_mm_per_prompt={"image": _LIMIT_IMAGE_PER_PROMPT
                                          }) as vllm_model:
        vllm_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images)
            for prompts, images in inputs
        ]

    with hf_runner(model, dtype=dtype, is_vision_model=True) as hf_model:
        hf_outputs_per_image = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=images)
            for prompts, images in inputs
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_image,
                                        vllm_outputs_per_image):
        # TODO: Check whether using original CLIPVisionModel can improve
        # consistency against HF
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=[
                vllm_to_hf_output(vllm_output, model)
                for vllm_output in vllm_outputs
            ],
            name_0="hf",
            name_1="vllm",
        )


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "size_factors",
    [
        # No image
        [],
        # Single-scale
        [1.0],
        # Single-scale, batched
        [1.0, 1.0, 1.0],
        # Multi-scale
        [0.25, 0.5, 1.0],
    ],
)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(hf_runner, vllm_runner, image_assets, model, size_factors,
                dtype, max_tokens, num_logprobs) -> None:
    """Inference result should be the same between hf and vllm.

    All the image fixtures for the test is under tests/images.
    For huggingface runner, we provide the PIL images as input.
    For vllm runner, we provide MultiModalDataDict objects
    and corresponding MultiModalConfig as input.
    Note, the text input is also adjusted to abide by vllm contract.
    The text output is sanitized to be able to compare with hf.
    """
    run_test(
        hf_runner,
        vllm_runner,
        image_assets,
        model,
        size_factors=size_factors,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "sizes",
    [[(1669, 2560), (2560, 1669), (183, 488), (488, 183)]],
)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models_fixed_sizes(hf_runner, vllm_runner, image_assets, model, sizes,
                            dtype, max_tokens, num_logprobs) -> None:
    run_test(
        hf_runner,
        vllm_runner,
        image_assets,
        model,
        sizes=sizes,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models_multiple_image_inputs(hf_runner, vllm_runner, image_assets,
                                      model, dtype, max_tokens,
                                      num_logprobs) -> None:
    img_0 = image_assets[0].pil_image
    img_1 = image_assets[1].pil_image

    inputs = [(
        [
            f"{_PREFACE} USER: <image><image>\nDescribe the 2 images. ASSISTANT:",
            f"{_PREFACE} USER: <image><image><image><image>\nDescribe the 4 images. ASSISTANT:"
        ],
        [
            [img_0, img_1],
            # Images with different sizes and aspect-ratios
            [
                img_0,
                rescale_image_size(img_0, 0.25),
                img_0.resize((183, 488)),
                img_0.resize((488, 183))
            ],
        ])]

    _run_test(
        hf_runner,
        vllm_runner,
        inputs,
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )
