import argparse
import copy
import itertools
import math
import pickle as pkl
import time
from typing import Callable, Iterable, List, Tuple, Optional, Dict
from functools import partial
from dataclasses import dataclass

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from weight_shapes import WEIGHT_SHAPES

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    GPTQ_MARLIN_MAX_PARALLEL, GPTQ_MARLIN_MIN_THREAD_N, marlin_permute_scales, 
    marlin_zero_points)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    MarlinWorkspace)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test_qqq import (
    marlin_qqq_quantize)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    gptq_pack, pack_rows, quantize_weights)
from vllm.scalar_type import ScalarType, scalar_types
from vllm.utils import FlexibleArgumentParser

DEFAULT_MODELS = ["meta-llama/Llama-3-8b", "meta-llama/Llama-2-70b-hf"]
DEFAULT_BATCH_SIZES = [1, 16, 32, 64, 128, 256, 512, 1024]
DEFAULT_TP_SIZES = [1]

@dataclass
class BenchmarkTensors:
    a: torch.tensor
    w_ref: torch.tensor
    w_q: torch.tensor
    group_size: int
    wtype: ScalarType
    w_s: torch.tensor
    w_zp: Optional[torch.tensor]

def quantize_and_pack_contiguous(w: torch.tensor, 
                             wtype: ScalarType, 
                             stype: torch.dtype,
                             zero_points: bool,
                             group_size: int) -> Dict[str, torch.tensor]:
    if stype is not None:
        w = w.to(stype)
        
    w_ref, w_q, w_s, w_zp = quantize_weights(
        w,
        wtype,
        group_size=group_size,
        zero_points=zero_points)
    w_q = pack_rows(w_q, wtype.size_bits, *w_q.shape)
    
    return { "w_ref": w_ref, "w_q": w_q, "w_s": w_s, "w_zp": w_zp }

def torch_matmul_create_bench_fn(bt: BenchmarkTensors) -> Callable:
    return lambda: torch.matmul(bt.a, bt.w_ref)

def marlin_create_bench_fn(bt: BenchmarkTensors) -> Callable:
    device = bt.a.device
    
    workspace = MarlinWorkspace(
        bt.w_ref.shape[1], GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MAX_PARALLEL)
    
    if bt.w_zp is None:
        w_zp = torch.empty(0, dtype=torch.int, device=device)
    else:
        w_zp = marlin_zero_points(bt.w_zp, *bt.w_ref.shape, bt.wtype.size_bits)
    w_s = marlin_permute_scales(bt.w_s, *bt.w_ref.shape, bt.group_size)

    sort_indices = torch.empty(0, dtype=torch.int, device=device)
    g_idx = torch.empty(0, dtype=torch.int, device=device)

    w_q = ops.gptq_marlin_repack(bt.w_q, sort_indices, *bt.w_ref.shape, 
                                 bt.wtype.size_bits)

    if bt.a.dtype.is_floating_point:
        fn = partial(ops.gptq_marlin_gemm,
                     a=bt.a,
                     b_q_weight=w_q,
                     b_scales=w_s,
                     b_zeros=w_zp,
                     g_idx=g_idx,
                     perm=sort_indices,
                     workspace=workspace.scratch,
                     b_q_type=bt.wtype,
                     size_m=bt.a.shape[0],
                     size_n=bt.w_ref.shape[1],
                     size_k=bt.w_ref.shape[0],
                     is_k_full=True)
    else:
        assert bt.a.dtype == torch.int8
        s_ch = torch.ones(bt.w_ref.shape[1], dtype=torch.float32, device=device)
        s_tok = torch.ones(bt.a.shape[0], dtype=torch.float32, device=device)
        fn = partial(ops.marlin_qqq_gemm,
                     a=bt.a,
                     b_q_weight=bt.w_q,
                     s_group=w_s,
                     s_tok=s_tok,
                     s_ch=s_ch,
                     workspace=workspace.scratch,
                     b_q_type=bt.wtype,
                     size_m=bt.a.shape[0],
                     size_n=bt.w_ref.shape[1],
                     size_k=bt.w_ref.shape[0])
    
    return fn

def machete_create_bench_fn(bt: BenchmarkTensors, schedule=None) -> Callable:
    w_q = bt.w_q.t().contiguous().t()  # make col major
    w_q = ops.machete_prepack_B(w_q, bt.a.dtype, bt.wtype)
    
    w_zp = bt.w_zp
    if w_zp is not None:
        w_zp = -1 * bt.w_s * (w_zp.to(bt.w_s.dtype))

    fn = partial(ops.machete_gemm,
                 a=bt.a, b_q=w_q, b_type=bt.wtype, b_scales=bt.w_s, 
                 b_group_size=bt.group_size, schedule=schedule)
    
    return fn


def make_bench_tensors(
    atype: torch.dtype, 
    wtype: ScalarType, 
    stype: torch.dtype,
    group_size: int, 
    m: int, 
    n: int,
    k: int
) -> List[BenchmarkTensors]:
    assert wtype.is_integer(), "TODO: support floating point weights"

    # we want to make sure that weights don't fit into L2 cache between runs so
    #  we construct enough weights to exceed L2 cache, which is 50mb on a H100
    #  so we target total weight size > 2*50mb
    num_weights = math.ceil(2 * 50 * 1024**2 * 8 / (k * n * wtype.size_bits))

    a = torch.randn((m, k), device="cuda", dtype=atype) * 5
    weights = [
        torch.randn((k, n), device="cuda", dtype=atype)
        for _ in range(num_weights)
    ]

    return [
        BenchmarkTensors(a=a, group_size=group_size, wtype=wtype,
                         **quantize_and_pack_contiguous(
            w, wtype,stype, zero_points=False, group_size=group_size))
        for w in weights
    ]


# impl


# bench
def bench_fn(label: str, sub_label: str, description: str,
             fn: Callable) -> TMeasurement:

    min_run_time = 1
    return TBenchmark.Timer(
        stmt="fn()",
        globals={
            "fn": fn
        },
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)


def loop_bench_fns(fns: List[Callable], label: str, sub_label: str):
    for fn in fns:
        fn()

def bench_fns(label: str, sub_label: str, description: str, fns: List[Callable]):
    min_run_time = 1
    return TBenchmark.Timer(
        stmt="""
        for fn in fns:
            fn()
        """,
        globals={
            "fns": fns
        },
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)


def bench(atype: torch.dtype,
          wtype: ScalarType,
          stype: torch.dtype,
          group_size: int,
          m: int,
          k: int,
          n: int,
          label: str,
          sub_label: str,
          benchmark_marlinv1: bool = True,
          sweep_schedules: bool = True) -> Iterable[TMeasurement]:
    benchmark_tensors = make_bench_tensors(atype, wtype, stype, group_size, m, n, k)
    sub_label += f", L={len(benchmark_tensors)}"


    timers = []
    # pytorch impl
    timers.append(bench_fns(
        label, sub_label, "torch.matmul", [
            torch_matmul_create_bench_fn(bt) for bt in benchmark_tensors]))
    
    if benchmark_marlinv1:
        timers.append(bench_fns(
            label, sub_label, "marlin", [
                marlin_create_bench_fn(bt) for bt in benchmark_tensors]))

    # machete
    timers.append(bench_fns(
        label, sub_label, "machete", [
            machete_create_bench_fn(bt) for bt in benchmark_tensors]))

    if sweep_schedules:
        print("Finding best schedule for machete")
        best = None
        best_schedule = None
        schedules = ops.machete_supported_schedules(
            atype, wtype, stype)
        for schedule in reversed(schedules):
            res = bench_fns(label, sub_label, "machete", [
                machete_create_bench_fn(bt, schedule=schedule) 
                for bt in benchmark_tensors])

            print(f"  {res.median:5.5} ", schedule)
            if not best or res.median < best.median:
                best = res
                best_schedule = schedule
        print("Best schedule:", best_schedule)
        timers.append(best)

    return timers


# runner
def print_timers(timers: Iterable[TMeasurement]):
    compare = TBenchmark.Compare(timers)
    compare.print()


def run(dtype: torch.dtype, sweep_schedules: bool,
        MKNs: Iterable[Tuple[int, int, int]]) -> Iterable[TMeasurement]:

    results = []
    for m, k, n in MKNs:
        timers = bench(dtype,
                       scalar_types.uint4b8,
                       torch.float16,
                       128,
                       m,
                       k,
                       n,
                       f"{dtype}-gemm",
                       f"MKN=({m}x{k}x{n})",
                       sweep_schedules=sweep_schedules)
        print_timers(timers)
        results.extend(timers)

    return results


# output makers
def make_output(
    data: Iterable[TMeasurement],
    MKNs: Iterable[Tuple[int, int, int]],
    base_description: str,
    timestamp=None,
):

    print(f"== All Results {base_description} ====")
    print_timers(data)

    # pickle all the results
    timestamp = int(time.time()) if timestamp is None else timestamp
    with open(f"{base_description}-{timestamp}.pkl", "wb") as f:
        pkl.dump(data, f)


# argparse runners


def run_square_bench(args):
    dim_sizes = list(
        range(args.dim_start, args.dim_end + 1, args.dim_increment))
    MKNs = list(zip(dim_sizes, dim_sizes, dim_sizes))
    data = run(args.dtype, args.sweep_schedules, MKNs)

    make_output(data, MKNs, f"square_bench-{args.dtype}")


def run_range_bench(args):
    dim_sizes = list(range(args.dim_start, args.dim_end, args.dim_increment))
    n = len(dim_sizes)
    Ms = [args.m_constant] * n if args.m_constant is not None else dim_sizes
    Ks = [args.k_constant] * n if args.k_constant is not None else dim_sizes
    Ns = [args.n_constant] * n if args.n_constant is not None else dim_sizes
    MKNs = list(zip(Ms, Ks, Ns))
    data = run(args.dtype, args.sweep_schedules, MKNs)

    make_output(data, MKNs, f"range_bench-{args.dtype}")


def run_model_bench(args):

    print("Benchmarking models:")
    for i, model in enumerate(args.models):
        print(f"[{i}]  {model}")

    def model_shapes(model_name: str, tp_size: int) -> List[Tuple[int, int]]:
        KNs = []
        for KN, tp_split_dim in copy.deepcopy(WEIGHT_SHAPES[model_name]):
            KN[tp_split_dim] = KN[tp_split_dim] // tp_size
            KNs.append(KN)
        return KNs

    model_bench_data = []
    models_tps = list(itertools.product(args.models, args.tp_sizes))
    for model, tp_size in models_tps:
        Ms = args.batch_sizes
        KNs = model_shapes(model, tp_size)
        MKNs = []
        for m in Ms:
            for k, n in KNs:
                MKNs.append((m, k, n))

        data = run(args.dtype, args.sweep_schedules, MKNs)
        model_bench_data.append(data)

    # Print all results
    for data, model_tp in zip(model_bench_data, models_tps):
        model, tp_size = model_tp
        print(f"== Results {args.dtype} {model}-TP{tp_size} ====")
        print_timers(data)

    timestamp = int(time.time())

    all_data = []
    for d in model_bench_data:
        all_data.extend(d)
    # pickle all data
    with open(f"model_bench-{args.dtype}-{timestamp}.pkl", "wb") as f:
        pkl.dump(all_data, f)


if __name__ == "__main__":

    def to_torch_dtype(dt):
        if dt == "bfloat16":
            return torch.bfloat16
        if dt == "float16":
            return torch.float16
        raise ValueError("unsupported dtype")

    parser = FlexibleArgumentParser(
        description="""
Benchmark Machete GEMM.

    To run square GEMMs:
        python3 ./benchmarks/kernels/benchmark_machete.py --dtype float16 square_bench --dim-start 128 --dim-end 512 --dim-increment 64
    
    To run constant N and K and sweep M:
        python3 ./benchmarks/kernels/benchmark_machete.py --dtype float16 range_bench --dim-start 128 --dim-end 512 --dim-increment 64 --n-constant 16384 --k-constant 16384
    
    To run dimensions from a model:
        python3 ./benchmarks/kernels/benchmark_machete.py --dtype float16 model_bench --models meta-llama/Llama-2-7b-hf --batch-sizes 16 --tp-sizes 1
    
    Output:
        - a .pkl file, that is a list of raw torch.benchmark.utils.Measurements for the pytorch and cutlass implementations for the various GEMMs.
            """,  # noqa: E501
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--atype",
        type=to_torch_dtype,
        required=True,
        help="Available options are ['bfloat16', 'float16']",
    )
    parser.add_argument(
        "--sweep-schedules",
        action="store_true",
        help="Run a sweep over all supported schedules",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    square_parser = subparsers.add_parser("square_bench")
    square_parser.add_argument("--dim-start", type=int, required=True)
    square_parser.add_argument("--dim-end", type=int, required=True)
    square_parser.add_argument("--dim-increment", type=int, required=True)
    square_parser.set_defaults(func=run_square_bench)

    range_parser = subparsers.add_parser("range_bench")
    range_parser.add_argument("--dim-start", type=int, required=True)
    range_parser.add_argument("--dim-end", type=int, required=True)
    range_parser.add_argument("--dim-increment", type=int, required=True)
    range_parser.add_argument("--m-constant", type=int, default=None)
    range_parser.add_argument("--n-constant", type=int, default=None)
    range_parser.add_argument("--k-constant", type=int, default=None)
    range_parser.set_defaults(func=run_range_bench)

    model_parser = subparsers.add_parser("model_bench")
    model_parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=DEFAULT_MODELS,
        choices=WEIGHT_SHAPES.keys(),
    )
    model_parser.add_argument("--tp-sizes",
                              nargs="+",
                              type=int,
                              default=DEFAULT_TP_SIZES)
    model_parser.add_argument("--batch-sizes",
                              nargs="+",
                              type=int,
                              default=DEFAULT_BATCH_SIZES)
    model_parser.set_defaults(func=run_model_bench)

    args = parser.parse_args()
    args.func(args)
