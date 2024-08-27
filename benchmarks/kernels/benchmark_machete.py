import argparse
import copy
import itertools
import math
import pickle as pkl
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Iterable, List, Optional, Tuple

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
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_rows, quantize_weights)
from vllm.scalar_type import ScalarType, scalar_types
from vllm.utils import FlexibleArgumentParser

DEFAULT_MODELS = ["meta-llama/Llama-3-8b", "meta-llama/Llama-2-70b-hf"]
DEFAULT_BATCH_SIZES = [1, 16, 32, 64, 128, 256, 512, 1024]
DEFAULT_TP_SIZES = [1]


def terse_type_name(dt):
    return {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.int8: "int8",
        torch.float8_e4m3fn: "fp8",
        torch.bfloat16: "bf16",
        torch.float: "float",
        torch.int: "int",
    }[dt]

@dataclass
class BenchmarkTensors:
    a: torch.Tensor
    w_ref: torch.Tensor
    w_q: torch.Tensor
    group_size: int
    wtype: ScalarType
    w_s: torch.Tensor
    w_zp: Optional[torch.Tensor]


def quantize_and_pack_contiguous(w: torch.Tensor, wtype: ScalarType,
                                 stype: torch.dtype, zero_points: bool,
                                 group_size: int) -> Dict[str, torch.Tensor]:
    if stype is not None:
        w = w.to(stype)

    w_ref, w_q, w_s, w_zp = quantize_weights(w,
                                             wtype,
                                             group_size=group_size,
                                             zero_points=zero_points)
    w_q = pack_rows(w_q, wtype.size_bits, *w_q.shape)

    return {"w_ref": w_ref, "w_q": w_q, "w_s": w_s, "w_zp": w_zp}


def torch_matmul_f16_create_bench_fn(bt: BenchmarkTensors) -> Callable:
    a, w = bt.a, bt.w_ref
    if a.dtype not in [torch.float16, torch.bfloat16]:
        a = a.to(torch.float16)
        w = w.to(torch.float16)
    return lambda: torch.matmul(a, w)


def cutlass_scaled_mm_create_bench_fn(bt: BenchmarkTensors) -> Callable:
    scale_a = torch.tensor(1.0, dtype=torch.float32, device=bt.a.device)
    scale_b = torch.tensor(1.0, dtype=torch.float32, device=bt.a.device)
    w_col_major = bt.w_ref.to(bt.a.dtype).t().contiguous().t()
    return lambda: ops.cutlass_scaled_mm(bt.a, w_col_major,
                   scale_a, scale_b, out_dtype=torch.float16)


def marlin_create_bench_fn(bt: BenchmarkTensors) -> Callable:
    device = bt.a.device

    workspace = MarlinWorkspace(bt.w_ref.shape[1], GPTQ_MARLIN_MIN_THREAD_N,
                                GPTQ_MARLIN_MAX_PARALLEL)

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

        fn = lambda: ops.gptq_marlin_gemm(a=bt.a,
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
        assert bt.wtype == scalar_types.uint4b8

        s_ch = torch.ones(bt.w_ref.shape[1],
                          dtype=torch.float32,
                          device=device)
        s_tok = torch.ones(bt.a.shape[0], dtype=torch.float32, device=device)
        fn = lambda: ops.marlin_qqq_gemm(a=bt.a,
                                         b_q_weight=w_q,
                                         s_group=w_s,
                                         s_tok=s_tok,
                                         s_ch=s_ch,
                                         workspace=workspace.scratch,
                                         size_m=bt.a.shape[0],
                                         size_n=bt.w_ref.shape[1],
                                         size_k=bt.w_ref.shape[0])

    return fn


def machete_create_bench_fn(bt: BenchmarkTensors,
                            otype=torch.dtype,
                            schedule=None) -> Callable:
    w_q = bt.w_q.t().contiguous().t()  # make col major
    w_q = ops.machete_prepack_B(w_q, bt.a.dtype, bt.wtype)

    w_zp = bt.w_zp
    if w_zp is not None:
        w_zp = -1 * bt.w_s * (w_zp.to(bt.w_s.dtype))

    return lambda: ops.machete_gemm(a=bt.a,
                                    b_q=w_q,
                                    b_type=bt.wtype,
                                    b_scales=bt.w_s,
                                    b_zeros=w_zp,
                                    out_type=otype,
                                    b_group_size=bt.group_size,
                                    schedule=schedule)


def make_bench_tensors(atype: torch.dtype, wtype: ScalarType,
                       stype: torch.dtype, group_size: int, m: int, n: int,
                       k: int) -> List[BenchmarkTensors]:
    assert wtype.is_integer(), "TODO: support floating point weights"

    # we want to make sure that weights don't fit into L2 cache between runs so
    #  we construct enough weights to exceed L2 cache, which is 50mb on a H100
    #  so we target total weight size > 2*50mb
    num_weights = math.ceil(2 * 50 * 1024**2 * 8 / (k * n * wtype.size_bits))

    a = (torch.randn((m, k), device="cuda") * 5).to(dtype=atype)
    weights = [(torch.randn((k, n), device="cuda") * 5).to(dtype=atype)
               for _ in range(num_weights)]

    return [
        BenchmarkTensors(a=a,
                         group_size=group_size,
                         wtype=wtype,
                         **quantize_and_pack_contiguous(w,
                                                        wtype,
                                                        stype,
                                                        zero_points=False,
                                                        group_size=group_size))
        for w in weights
    ]


# impl

# bench


def bench_fns(label: str, sub_label: str, description: str,
              fns: List[Callable]):
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
          stype: Optional[torch.dtype],
          otype: Optional[torch.dtype],
          group_size: int,
          m: int,
          k: int,
          n: int,
          label: str,
          sub_label: str,
          benchmark_marlinv1: bool = True,
          sweep_schedules: bool = True) -> Iterable[TMeasurement]:
    stype = atype if stype is None else stype
    otype = atype if otype is None else otype
    benchmark_tensors = make_bench_tensors(atype, wtype, stype, group_size, m,
                                           n, k)
    sub_label += f", L={len(benchmark_tensors)}"
    name_type_string = f"W{wtype}-A{terse_type_name(atype)}" +\
                       f"-S{terse_type_name(stype)}-O{terse_type_name(otype)}"+\
                       f"-G{group_size}"

    timers = []
    # pytorch impl
    timers.append(
        bench_fns(
            label, sub_label, "torch.matmul (fp16)",
            [torch_matmul_f16_create_bench_fn(bt) for bt in benchmark_tensors]))
    
    if atype == torch.int8 or atype == torch.float8_e4m3fn:
        timers.append(
            bench_fns(
                label, sub_label, 
                f"cutlass_scaled_mm ({terse_type_name(atype)})",
                [cutlass_scaled_mm_create_bench_fn(bt)
                 for bt in benchmark_tensors]))

    if benchmark_marlinv1 and atype != torch.float8_e4m3fn:
        timers.append(
            bench_fns(label, sub_label, f"marlin ({name_type_string})",
                      [marlin_create_bench_fn(bt)
                       for bt in benchmark_tensors]))

    # machete
    timers.append(
        bench_fns(label, sub_label, f"machete ({name_type_string})", [
            machete_create_bench_fn(bt, otype=otype)
            for bt in benchmark_tensors
        ]))

    if sweep_schedules:
        print("Finding best schedule for machete")
        best = None
        best_schedule = None
        schedules = ops.machete_supported_schedules(atype, wtype, stype)
        for schedule in reversed(schedules):
            res = bench_fns(label, sub_label, "machete", [
                machete_create_bench_fn(bt, schedule=schedule)
                for bt in benchmark_tensors
            ])

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


def run(args, MKNs: Iterable[Tuple[int, int, int]]) -> Iterable[TMeasurement]:
    results = []
    for m, k, n in MKNs:
        timers = bench(args.atype,
                       scalar_types.uint4b8,
                       args.stype,
                       args.otype,
                       128,
                       m,
                       k,
                       n,
                       f"{args.atype}-gemm",
                       f"MKN=({m}x{k}x{n})",
                       sweep_schedules=args.sweep_schedules)
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
    data = run(args, MKNs)

    make_output(data, MKNs, f"square_bench-{args.dtype}")


def run_range_bench(args):
    dim_sizes = list(range(args.dim_start, args.dim_end, args.dim_increment))
    n = len(dim_sizes)
    Ms = [args.m_constant] * n if args.m_constant is not None else dim_sizes
    Ks = [args.k_constant] * n if args.k_constant is not None else dim_sizes
    Ns = [args.n_constant] * n if args.n_constant is not None else dim_sizes
    MKNs = list(zip(Ms, Ks, Ns))
    data = run(args, MKNs)

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

        data = run(args, MKNs)
        model_bench_data.append(data)

    type_string = f"{args.atype}-{args.stype}-{args.otype}"

    # Print all results
    for data, model_tp in zip(model_bench_data, models_tps):
        model, tp_size = model_tp
        print(f"== Results {type_string} {model}-TP{tp_size} ====")
        print_timers(data)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    all_results = []
    for d in model_bench_data:
        all_results.extend(d)

    # pickle all data
    with open(f"model_bench-{type_string}-{timestr}.pkl", "wb") as f:
        args_dict = vars(args)
        args_dict.pop("func")
        pkl.dump({
            "args": args_dict, 
            "results": all_results,
        }, f)


if __name__ == "__main__":

    def to_torch_dtype(dt):
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "int8": torch.int8,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "int": torch.int,
            "float": torch.float,
        }[dt]

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
        help="Available options are "
        "['bfloat16', 'float16', 'int8', 'float8_e4m3fn']",
    )
    parser.add_argument(
        "--stype",
        type=to_torch_dtype,
        help="Available options are ['bfloat16', 'float16']",
    )
    parser.add_argument(
        "--otype",
        type=to_torch_dtype,
        help="Available options are "
        "['bfloat16', 'float16', 'int', 'float']",
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
