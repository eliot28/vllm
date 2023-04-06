import argparse
import logging
import os
import pickle
import time
from typing import List

from tqdm import tqdm
from transformers import AutoConfig

from benchmark.trace import generate_text_completion_requests
from cacheflow.master.simple_frontend import SimpleFrontend
from cacheflow.master.server import (Server, add_server_arguments,
                                     initialize_ray_cluster)
from cacheflow.sampling_params import SamplingParams
from cacheflow.utils import get_gpu_memory, get_cpu_memory


logger = logging.getLogger(__name__)


def main(args: argparse.Namespace):
    assert args.pipeline_parallel_size == 1, (
        'Pipeline parallelism is not supported yet.')

    (num_nodes, num_devices_per_node, distributed_init_method,
    all_stage_devices) = (
        initialize_ray_cluster(
            address='local',
            pipeline_parallel_size=args.pipeline_parallel_size,
            tensor_parallel_size=args.tensor_parallel_size))

    # Create a server.
    server = Server(
        model=args.model,
        model_path=args.model_path,
        pipeline_parallel_size=args.pipeline_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        block_size=args.block_size,
        dtype=args.dtype,
        seed=args.seed,
        swap_space=args.swap_space,
        max_num_batched_tokens=args.max_num_batched_tokens,
        num_nodes=num_nodes,
        num_devices_per_node=num_devices_per_node,
        distributed_init_method=distributed_init_method,
        all_stage_devices=all_stage_devices,
        gpu_memory=get_gpu_memory(),
        cpu_memory=get_cpu_memory(),
        collect_stats=True,
    )

    # Create a frontend.
    frontend = SimpleFrontend(
        model_name=args.model,
        block_size=args.block_size,
    )
    # Generate requests.
    requests = generate_text_completion_requests(
        args.dataset,
        args.request_rate,
        args.duration,
        args.seed,
        args.n1,
        args.n2,
        args.n4,
        args.n8,
        args.n2_beam,
        args.n4_beam,
        args.n8_beam,
    )

    # Warm up.
    logger.info('Warming up.')
    num_warmup_requests = 8
    warmup_input_len = 8
    warmup_output_len = 32
    warmup_sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=0.99,
        max_num_steps=warmup_output_len,
        use_beam_search=False,
        stop_token_ids=set(),
        num_logprobs=0,
        context_window_size=None,
    )
    for _ in range(num_warmup_requests):
        frontend._add_query([0] * warmup_input_len, warmup_sampling_params)
    server.add_sequence_groups(frontend.get_inputs())
    while True:
        server.step()
        if not server.has_unfinished_requests():
            break

    # Start benchmarking.
    logger.info('Start benchmarking.')
    # Initialize tqdm.
    pbar = tqdm(total=len(requests), desc='Finished requests')

    finished = []
    start_time = time.time()
    while True:
        now = time.time()
        while requests:
            if requests[0][0] <= now - start_time:
                request_time, input_tokens, sampling_params = requests.pop(0)
                frontend._add_query(
                    input_tokens, sampling_params, arrival_time=now + request_time)
            else:
                break
        server.add_sequence_groups(frontend.get_inputs())
        updated_seq_groups = server.step()

        now = time.time()
        for seq_group in updated_seq_groups:
            if not seq_group.is_finished():
                continue
            arrival_time = seq_group.arrival_time
            finish_time = now
            for seq in seq_group.get_seqs():
                seq_len = seq.get_len()
                output_len = seq_len - seq.prompt_len
                finished.append({
                    'group_id': seq_group.group_id,
                    'seq_id': seq.seq_id,
                    'arrival_time': arrival_time, 
                    'finish_time': finish_time,
                    'prompt_len': seq.prompt_len,
                    'output_len': output_len,
                })
            pbar.update(1)

        if not (requests or server.has_unfinished_requests()):
            break
    pbar.close()
    logger.info('Finish benchmarking. Saving stats.')
    server.scheduler.save_stats(args.output_dir)
    with open(os.path.join(args.output_dir, 'sequences.pkl'), 'wb') as f:
        pickle.dump(finished, f)
    logger.info('Done.')


def get_model_name(model: str) -> str:
    OPT_MODELS = [
        'opt-125m',
        'opt-350m',
        'opt-1.3b',
        'opt-2.7b',
        'opt-6.7b',
        'opt-13b',
        'opt-30b',
        'opt-60b',
    ]
    for opt_model in OPT_MODELS:
        if opt_model in model:
            return opt_model

    config = AutoConfig.from_pretrained(model)
    assert config.model_type == 'llama'
    hidden_size = config.hidden_size
    if hidden_size == 4096:
        return 'llama-7b'
    elif hidden_size == 5120:
        return 'llama-13b'
    elif hidden_size == 6656:
        return 'llama-30b'
    elif hidden_size == 8192:
        return 'llama-65b'
    else:
        raise ValueError(f'Unknown model: {model}')


def get_dataset_name(dataset: str) -> str:
    if 'sharegpt' in dataset.lower():
        return 'sharegpt'
    elif 'alpaca' in dataset.lower():
        return 'alpaca'
    else:
        raise ValueError(f'Unknown dataset: {dataset}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CacheFlow simple server.')
    parser = add_server_arguments(parser) 
    parser.add_argument('--output-dir', type=str, help='path to output directory', default=None)

    parser.add_argument('--dataset', type=str, help='path to dataset', required=True)
    parser.add_argument('--request-rate', type=float, help='reqs/sec', required=True)
    parser.add_argument('--duration', type=int, help='duration in seconds', required=True)

    parser.add_argument('--n1', type=float, help='ratio of requests with n=1', default=0.0)
    parser.add_argument('--n2', type=float, help='ratio of requests with n=2', default=0.0)
    parser.add_argument('--n4', type=float, help='ratio of requests with n=4', default=0.0)
    parser.add_argument('--n8', type=float, help='ratio of requests with n=8', default=0.0)
    parser.add_argument('--n2-beam', type=float, help='ratio of requests with n=2 & beam search', default=0.0)
    parser.add_argument('--n4-beam', type=float, help='ratio of requests with n=4 & beam search', default=0.0)
    parser.add_argument('--n8-beam', type=float, help='ratio of requests with n=8 & beam search', default=0.0)
    args = parser.parse_args()
    if args.n1 + args.n2 + args.n4 + args.n8 + args.n2_beam + args.n4_beam + args.n8_beam != 1.0:
        raise ValueError('The ratios of requests must sum to 1.')

    model_name = get_model_name(args.model)
    dataset_name = get_dataset_name(args.dataset)
    if 'opt' in model_name:
        if 'opt' not in args.dataset.lower():
            raise ValueError(f'OPT models can only be used with OPT datasets.')
    elif 'llama' in model_name:
        if 'llama' not in args.dataset.lower():
            raise ValueError(f'Llama models can only be used with Llama datasets.')

    dataset_name = 'sharegpt' if 'sharegpt' in args.dataset else 'alpaca'
    if args.output_dir is None:
        args.output_dir = os.path.join(
            'outputs',
            dataset_name,
            f'{model_name}-tp{args.tensor_parallel_size}',
            f'sample-n1-{args.n1}-n2-{args.n2}-n4-{args.n4}-n8-{args.n8}-n2b-{args.n2_beam}-n4b-{args.n4_beam}-n8b-{args.n8_beam}',
            f'req-rate-{args.request_rate}',
            f'duration-{args.duration}',
            f'seed{args.seed}',
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
        ],
    )
    logger.info(args)

    main(args)
