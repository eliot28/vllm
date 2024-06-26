#!/bin/bash

set -ex
set -o pipefail


main() {

    (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
    (which jq) || (apt-get update && apt-get -y install jq)
    
    df -h

    if [ ! -f /workspace/buildkite-agent ]; then
        echo "buildkite-agent binary not found. Skip uploading the results."
        return 0
    else
        /workspace/buildkite-agent annotate --style "info" --context "benchmark-results" < $VLLM_SOURCE_CODE_LOC/.buildkite/nightly-benchmarks/nightly-descriptions.md
    fi
    
}

main "$@"