from transformers import AutoConfig, PretrainedConfig

from vllm.transformers_utils.configs import *  # pylint: disable=wildcard-import

_CONFIG_REGISTRY = {
    "mpt": MPTConfig,
    "baichuan": BaiChuanConfig,
    "aquila": AquilaConfig,
    "qwen": QWenConfig,
    "RefinedWeb": RWConfig,  # For tiiuae/falcon-40b(-instruct)
    "RefinedWebModel": RWConfig,  # For tiiuae/falcon-7b(-instruct)
    "skywork": SkyWorkConfig,
    "skywork2.6B": SkyWork_2_6BConfig,
}


def get_config(model: str, trust_remote_code: bool) -> PretrainedConfig:
    try:
        if model.find("skywork") != -1:
            config = SkyWorkConfig()
            config.save_pretrained("skywork")
        elif model.find("skywork2.6B") != -1:
            config = SkyWork_2_6BConfig()
            config.save_pretrained("skywork2.6B")
        else:
            config = AutoConfig.from_pretrained(
                model, trust_remote_code=trust_remote_code)
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
        config = config_class.from_pretrained(model)
    return config



