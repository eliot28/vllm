.. _nginxloadbalancer:

Nginx Loadbalancer
========================

This document shows how to launch multiple vLLM serving containers and use Nginx to act as a load balancer between the servers. 

.. _nginxloadbalancer_nginx_build:

Build Nginx Container
------------

This guide assumes that you have just cloned the vLLM project and you're currently in the vllm root directory.

.. code-block:: console

    export vllm_root=`pwd`

Create a file named ``Dockerfile.nginx``:

.. code-block:: console

    # Copyright (C) 2024 Intel Corporation
    # SPDX-License-Identifier: Apache-2.0

    FROM nginx:latest
    RUN rm /etc/nginx/conf.d/default.conf
    EXPOSE 80
    CMD ["nginx", "-g", "daemon off;"]

Build the container:

.. code-block:: console

    docker build . -f Dockerfile.nginx --tag nginx-lb

Create Simple Nginx Config file
------------

Create a file named ``nginx_conf/nginx.conf``. Note that you can add as many servers as you'd like. In the below example we'll start with two. To add more, add another ``server vllmN:8000 max_fails=3 fail_timeout=10000s;`` entry to ``upstream backend``.

.. code-block:: console

    upstream backend {
        least_conn;
        server vllm0:8000 max_fails=3 fail_timeout=10000s;
        server vllm1:8000 max_fails=3 fail_timeout=10000s;
    }     
    server {
        listen 80;
        location / {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }

Build vLLM Container
------------

Notes:
* Adjust the model name that you want to use in your vLLM servers if you don't want to use ``Llama-2-7b-hf``. 

.. code-block:: console

    cd $vllm_root
    model=meta-llama/Llama-2-7b-hf
    sed -i "s|ENTRYPOINT \[\"python3\", \"-m\", \"vllm.entrypoints.openai.api_server\"\]|ENTRYPOINT [\"python3\", \"-m\", \"vllm.entrypoints.openai.api_server\", \"--model\", \"$model\"]|" Dockerfile.cpu
    docker build -f Dockerfile.cpu . --tag vllm --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy

Create Docker Network
------------

.. code-block:: console

    docker network create vllm_nginx

Launch vLLM Containers
------------

Notes:
* If you have your HuggingFace models cached somewhere else, update ``hf_cache_dir`` below. 
* If you don't have an existing HuggingFace cache you will want to start ``vllm0`` and wait for the model to complete downloading and the server to be ready. This will ensure that ``vllm1`` can leverage the model you just downloaded and it won't have to be downloaded again.
* The below example assumes a machine where socket 0 has cores 0-47 and socket 1 has cores 48-95. Adjust as needed for your application.

.. code-block:: console

    mkdir -p ~/.cache/huggingface/hub/
    hf_cache_dir=~/.cache/huggingface/
    SVR_0_CORES=0-47
    SVR_1_CORES=48-96
    docker run -itd --ipc host --network vllm_nginx --cap-add=SYS_ADMIN --shm-size=10.24gb -e VLLM_CPU_KVCACHE_SPACE=40 -e VLLM_CPU_OMP_THREADS_BIND=$SVR_0_CORES -e http_proxy=$http_proxy -e https_proxy=$https_proxy -v $hf_cache_dir:/root/.cache/huggingface/ -p 8081:8000 --name vllm0 vllm
    docker run -itd --ipc host --network vllm_nginx --cap-add=SYS_ADMIN --shm-size=10.24gb -e VLLM_CPU_KVCACHE_SPACE=40 -e VLLM_CPU_OMP_THREADS_BIND=$SVR_1_CORES -e http_proxy=$http_proxy -e https_proxy=$https_proxy -v $hf_cache_dir:/root/.cache/huggingface/ -p 8082:8000 --name vllm1 vllm 

Launch Nginx
------------

.. code-block:: console

    docker run -itd -p 8000:80 --network vllm_nginx -v ./nginx_conf/:/etc/nginx/conf.d/ --name nginx-lb nginx-lb:latest 

