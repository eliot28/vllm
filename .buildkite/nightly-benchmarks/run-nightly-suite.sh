#!/bin/bash

set -ex
set -o pipefail

check_gpus() {
    # check the number of GPUs and GPU type.
    declare -g gpu_count=$(nvidia-smi --list-gpus | wc -l)
    if [[ $gpu_count -gt 0 ]]; then
        echo "GPU found."
    else
        echo "Need at least 1 GPU to run benchmarking."
        exit 1
    fi
    declare -g gpu_type=$(echo $(nvidia-smi --query-gpu=name --format=csv,noheader) | awk '{print $2}')
    echo "GPU type is $gpu_type"
}

check_hf_token() {
    # check if HF_TOKEN is available and valid
    if [[ -z "$HF_TOKEN" ]]; then
        echo "Error: HF_TOKEN is not set."
        exit 1
    elif [[ ! "$HF_TOKEN" =~ ^hf_ ]]; then
        echo "Error: HF_TOKEN does not start with 'hf_'."
        exit 1
    else
        echo "HF_TOKEN is set and valid."
    fi
}

main() {

    check_gpus
    check_hf_token

    (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
    cd /
    git clone https://github.com/KuntaiDu/vllm.git
    cd vllm
    git checkout kuntai-benchmark-dev
    cd benchmarks
    wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

    # run lmdeploy
    if which lmdeploy >/dev/null; then
        echo "lmdeploy is available, redirect to run-lmdeploy-nightly.sh"
        bash ../.buildkite/nightly-benchmarks/scripts/run-lmdeploy-nightly.sh
    fi

}

main "$@"