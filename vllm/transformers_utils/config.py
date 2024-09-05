import contextlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from huggingface_hub import hf_hub_download
from transformers import GenerationConfig, PretrainedConfig
from transformers.models.auto.image_processing_auto import (
    get_image_processor_config)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)

from vllm.envs import VLLM_USE_MODELSCOPE
from vllm.logger import init_logger
from vllm.transformers_utils.configs import (ChatGLMConfig, DbrxConfig,
                                             EAGLEConfig, ExaoneConfig,
                                             InternVLChatConfig, JAISConfig,
                                             MedusaConfig, MLPSpeculatorConfig,
                                             MPTConfig, NemotronConfig,
                                             RWConfig, UltravoxConfig)
from vllm.transformers_utils.utils import check_gguf_file

if VLLM_USE_MODELSCOPE:
    from modelscope import AutoConfig
else:
    from transformers import AutoConfig

logger = init_logger(__name__)

_CONFIG_REGISTRY: Dict[str, Type[PretrainedConfig]] = {
    "chatglm": ChatGLMConfig,
    "dbrx": DbrxConfig,
    "mpt": MPTConfig,
    "RefinedWeb": RWConfig,  # For tiiuae/falcon-40b(-instruct)
    "RefinedWebModel": RWConfig,  # For tiiuae/falcon-7b(-instruct)
    "jais": JAISConfig,
    "mlp_speculator": MLPSpeculatorConfig,
    "medusa": MedusaConfig,
    "eagle": EAGLEConfig,
    "exaone": ExaoneConfig,
    "internvl_chat": InternVLChatConfig,
    "nemotron": NemotronConfig,
    "ultravox": UltravoxConfig,
}

for name, cls in _CONFIG_REGISTRY.items():
    with contextlib.suppress(ValueError):
        AutoConfig.register(name, cls)


def get_config(
    model: Union[str, Path],
    trust_remote_code: bool,
    revision: Optional[str] = None,
    code_revision: Optional[str] = None,
    rope_scaling: Optional[dict] = None,
    rope_theta: Optional[float] = None,
    load_from_params: bool = False,
    **kwargs,
) -> PretrainedConfig:
    # Separate model folder from file path for GGUF models

    is_gguf = check_gguf_file(model)
    if is_gguf:
        kwargs["gguf_file"] = Path(model).name
        model = Path(model).parent

    try:
        if load_from_params:
            config = load_params_config(model, revision)
        else:
            config = AutoConfig.from_pretrained(
                model,
                trust_remote_code=trust_remote_code,
                revision=revision,
                code_revision=code_revision,
                **kwargs)
    except ValueError as e:
        if (not trust_remote_code and
                "requires you to execute the configuration file" in str(e)):
            err_msg = (
                "Failed to load the model config. If the model is a custom "
                "model not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e
    if config.model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[config.model_type]
        config = config_class.from_pretrained(model,
                                              revision=revision,
                                              code_revision=code_revision)

    # Special architecture mapping check for GGUF models
    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(
                f"Can't get gguf config for {config.model_type}.")
        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config.update({"architectures": [model_type]})

    for key, value in [("rope_scaling", rope_scaling),
                       ("rope_theta", rope_theta)]:
        if value is not None:
            logger.info("Updating %s from %r to %r", key,
                        getattr(config, key, None), value)
            config.update({key: value})

    return config


def load_params_config(model, revision) -> PretrainedConfig:
    # This function loads a params.json config which
    # should be used when loading models in mistral format

    config_file_name = "params.json"

    config_path = Path(model) / config_file_name

    if not config_path.is_file():
        config_path = Path(
            hf_hub_download(model, config_file_name, revision=revision))

    with open(config_path, 'r') as file:
        config_dict = json.load(file)

    config_mapping = {
        "dim": "hidden_size",
        "norm_eps": "rms_norm_eps",
        "n_kv_heads": "num_key_value_heads",
        "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads",
        "hidden_dim": "intermediate_size",
    }

    def recurse_elems(elem: Any):
        if isinstance(elem, dict):
            config_dict = {}
            for key, value in elem.items():
                key = config_mapping.get(key, key)
                config_dict[key] = recurse_elems(value)
            return PretrainedConfig(**config_dict)
        else:
            return elem

    config_dict["model_type"] = config_dict.get("model_type", "transformer")
    config_dict["hidden_act"] = config_dict.get("activation", "silu")
    config_dict["tie_word_embeddings"] = config_dict.get(
        "tie_embeddings", False)

    if config_dict["model_type"] == "transformer":
        if "moe" in config_dict:
            config_dict["architectures"] = ["MixtralForCausalLM"]
        else:
            config_dict["architectures"] = ["MistralForCausalLM"]

    return recurse_elems(config_dict)


def get_hf_image_processor_config(
    model: Union[str, Path],
    revision: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    # ModelScope does not provide an interface for image_processor
    if VLLM_USE_MODELSCOPE:
        return dict()
    # Separate model folder from file path for GGUF models
    if check_gguf_file(model):
        model = Path(model).parent
    return get_image_processor_config(model, revision=revision, **kwargs)


def get_hf_text_config(config: PretrainedConfig):
    """Get the "sub" config relevant to llm for multi modal models.
        No op for pure text models.
    """
    if hasattr(config, "text_config"):
        # The code operates under the assumption that text_config should have
        # `num_attention_heads` (among others). Assert here to fail early
        # if transformers config doesn't align with this assumption.
        assert hasattr(config.text_config, "num_attention_heads")
        return config.text_config
    else:
        return config


def try_get_generation_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
) -> Optional[GenerationConfig]:
    try:
        return GenerationConfig.from_pretrained(
            model,
            revision=revision,
        )
    except OSError:  # Not found
        try:
            config = get_config(
                model,
                trust_remote_code=trust_remote_code,
                revision=revision,
            )
            return GenerationConfig.from_model_config(config)
        except OSError:  # Not found
            return None
