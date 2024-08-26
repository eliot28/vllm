import itertools
import math
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from functools import reduce
from copy import deepcopy

import jinja2
# yapf conflicts with isort for this block
# yapf: disable
from vllm_cutlass_library_extension import (DataType, EpilogueScheduleTag,
                                            EpilogueScheduleType,
                                            MixedInputKernelScheduleType,
                                            TileSchedulerTag,
                                            TileSchedulerType, VLLMDataType,
                                            VLLMDataTypeNames, VLLMDataTypeTag, 
                                            VLLMDataTypeVLLMScalarTypeTag,
                                            VLLMDataTypeTorchDataTypeTag,
                                            VLLMDataTypeSize,
                                            VLLMKernelScheduleTag)

# yapf: enable

#
#   Generator templating
#

DISPATCH_TEMPLATE = """
#include "../machete_mm_launcher.cuh"

namespace machete {

{% for impl_config in impl_configs %}
{% set type_sig = gen_type_sig(impl_config.types) -%}
{% for s in impl_config.schedules %}
extern torch::Tensor impl_{{type_sig}}_sch_{{gen_sch_sig(s)}}(PyTorchArguments);
{%- endfor %}

torch::Tensor gemm_dispatch_{{type_sig}}(PyTorchArguments args) {
  [[maybe_unused]] auto M = args.A.size(0);
  [[maybe_unused]] auto N = args.B.size(1);
  [[maybe_unused]] auto K = args.A.size(1);
    
  if (!args.schedule) {
    {%- for cond, s in impl_config.heuristic %}
    {%if cond is not none%}if ({{cond}})
    {%- else %}else
    {%- endif %}
        return impl_{{type_sig}}_sch_{{ gen_sch_sig(s) }}(args);{% endfor %}
  }

  {%- for s in impl_config.schedules %}
  if (*args.schedule == "{{ gen_sch_sig(s) }}")
    return impl_{{type_sig}}_sch_{{ gen_sch_sig(s) }}(args);
  {%- endfor %}
  TORCH_CHECK_NOT_IMPLEMENTED(false, "machete_gemm(..) is not implemented for "
                                     "schedule = ", *args.schedule);
}
{%- endfor %}

torch::Tensor gemm_dispatch(PyTorchArguments args) {
  auto out_type = args.out_type.value_or(args.A.scalar_type());
  auto a_type = args.A.scalar_type();

  {% for impl_config in impl_configs %}
  {% set t = impl_config.types -%}
  {% set type_sig = gen_type_sig(t) -%}
  {% set with_scales = t.b_scale != void -%}
  {% set with_zeropoints = t.b_zeropoint != void -%}
  if (a_type == {{TorchTypeTag[t.a]}}
      && args.btype == {{VLLMScalarTypeTag[t.b]}}
      && out_type == {{TorchTypeTag[t.d]}}
      && {%if with_scales%}args.scales && args.scales->scalar_type() == {{TorchTypeTag[t.b_scale]}}
      {%- else %}!args.scales{%endif%}
      && {%if with_zeropoints%}args.zeros && args.zeros->scalar_type() == {{TorchTypeTag[t.b_zeropoint]}}
      {%- else %}!args.zeros{%endif%}
  ) {
      return gemm_dispatch_{{type_sig}}(args);
  }
  {%- endfor %}
  
  TORCH_CHECK_NOT_IMPLEMENTED(
    false, "machete_mm(..) is not implemented for "
    "a_type=", args.A.scalar_type(),
    ", b_type=", args.btype.str(),
    ", out_type=", out_type,
    ", with_scales=", args.scales.has_value() ? toString(args.scales->scalar_type()) : "None",
    ", with_zeropoints=", args.zeros.has_value() ? toString(args.zeros->scalar_type()) : "None"
    "; implemented types are: \\n",
    {%- for impl_config in impl_configs %}
    {% set t = impl_config.types -%}
    {% set with_scales = t.b_scale != void -%}
    {% set with_zeropoints = t.b_zeropoint != void -%}
    "\\ta_type=", {{TorchTypeTag[t.a]}}, 
    ", b_type=", {{VLLMScalarTypeTag[t.b]}}.str(), 
    ", out_type=", {{TorchTypeTag[t.d]}},
    ", with_scales=", {%if with_scales%}{{TorchTypeTag[t.b_scale]}}
                      {%-else%}"None"{%endif%},
    ", with_zeropoints=", {%if with_zeropoints%}{{TorchTypeTag[t.b_zeropoint]}}
                          {%-else%}"None"{%endif%},
    "\\n",{%- endfor %}
    "");
}

}; // namespace machete
"""

IMPL_TEMPLATE = """
#include "../machete_mm_launcher.cuh"

namespace machete {
    
{% for sch in unique_schedules(impl_configs) %}
{% set sch_sig = gen_sch_sig(sch) -%}
struct sch_{{sch_sig}} {
  using TileShapeNM = Shape<{{
      to_cute_constant(sch.tile_shape_mn)|join(', ')}}>;
  using ClusterShape = Shape<{{
      to_cute_constant(sch.cluster_shape_mnk)|join(', ')}}>;
  // TODO: Reimplement
  // using KernelSchedule   = {{KernelScheduleTag[sch.kernel_schedule]}};
  using EpilogueSchedule = {{EpilogueScheduleTag[sch.epilogue_schedule]}};
  using TileScheduler    = {{TileSchedulerTag[sch.tile_scheduler]}};
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
};
{% endfor %}
    
{% for impl_config in impl_configs %}
{% set t = impl_config.types -%}
{% set schs = impl_config.schedules -%}
{% set type_sig = gen_type_sig(t) -%}

template<typename Sch>
using Kernel_{{type_sig}} = MacheteKernelTemplate<
  {{DataTypeTag[t.a]}},  // ElementA
  {{DataTypeTag[t.b]}},  // ElementB
  {{DataTypeTag[t.d]}},  // ElementD
  {{DataTypeTag[t.accumulator]}}, // Accumulator
  {{DataTypeTag[t.b_scale]}}, // Scales
  {{DataTypeTag[t.b_zeropoint]}}, // Zeropoints
  cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput,
  Sch>;

{% for sch in schs %}
{% set sch_sig = gen_sch_sig(sch) -%}
torch::Tensor 
impl_{{type_sig}}_sch_{{sch_sig}}(PyTorchArguments args) {
  return run_impl<Kernel_{{type_sig}}<sch_{{sch_sig}}>>(args);
}
{%- endfor %}
{%- endfor %}

}; // namespace machete
"""

PREPACK_TEMPLATE = """
#include "../machete_prepack_launcher.cuh"

namespace machete {

torch::Tensor prepack_B_dispatch(torch::Tensor B, at::ScalarType const& atype,
                                 vllm::ScalarType const& btype) {
                                   
  {%- for t in types %}
  {% set btype = unsigned_type_with_bitwidth(t.b_num_bits) %}
  if (atype == {{TorchTypeTag[t.a]}} &&
      btype.size_bits() == {{t.b_num_bits}}) {
    return prepack_impl<
      PrepackedLayoutBTemplate<
        {{DataTypeTag[t.a]}}, // ElementA
        {{DataTypeTag[btype]}}, // ElementB
        {{DataTypeTag[t.convert]}}, // ElementConvert
        {{DataTypeTag[t.accumulator]}}, // Accumulator
        cutlass::layout::ColumnMajor,
        cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput>
    >(B); 
  }
  {%- endfor %}
  
  TORCH_CHECK_NOT_IMPLEMENTED(false, 
    "prepack_B_dispatch(..) is not implemented for "
    "atype = ", atype,
    ", btype = ", btype.str());
}

}; // namespace machete
"""

TmaMI = MixedInputKernelScheduleType.TmaWarpSpecializedCooperativeMixedInput
TmaCoop = EpilogueScheduleType.TmaWarpSpecializedCooperative


@dataclass(frozen=True)
class ScheduleConfig:
    tile_shape_mn: Tuple[int, int]
    cluster_shape_mnk: Tuple[int, int, int]
    kernel_schedule: MixedInputKernelScheduleType
    epilogue_schedule: EpilogueScheduleType
    tile_scheduler: TileSchedulerType


@dataclass(frozen=True)
class TypeConfig:
    a: DataType
    b: Union[DataType, VLLMDataType]
    b_scale: DataType
    b_zeropoint: DataType
    d: DataType
    accumulator: DataType
    
@dataclass(frozen=True)
class PrepackTypeConfig:
    a: DataType
    b_num_bits: int
    convert: DataType
    accumulator: DataType

@dataclass
class ImplConfig:
    types: TypeConfig
    schedules: List[ScheduleConfig]
    heuristic: List[Tuple[Optional[str], ScheduleConfig]]


def generate_sch_sig(schedule_config: ScheduleConfig) -> str:
    tile_shape = (
        f"{schedule_config.tile_shape_mn[0]}x{schedule_config.tile_shape_mn[1]}"
    )
    cluster_shape = (f"{schedule_config.cluster_shape_mnk[0]}" +
                     f"x{schedule_config.cluster_shape_mnk[1]}" +
                     f"x{schedule_config.cluster_shape_mnk[2]}")
    kernel_schedule = VLLMKernelScheduleTag[schedule_config.kernel_schedule]\
        .split("::")[-1]
    epilogue_schedule = EpilogueScheduleTag[
        schedule_config.epilogue_schedule].split("::")[-1]
    tile_scheduler = TileSchedulerTag[schedule_config.tile_scheduler]\
        .split("::")[-1]

    return (f"{tile_shape}_{cluster_shape}_{kernel_schedule}" +
            f"_{epilogue_schedule}_{tile_scheduler}")


# mostly unique shorter sch_sig
def generate_terse_sch_sig(schedule_config: ScheduleConfig) -> str:
    kernel_terse_names_replace = {
        "KernelTmaWarpSpecializedCooperativeMixedInput_": "TmaMI_",
        "TmaWarpSpecializedCooperative_": "TmaCoop_",
        "StreamKScheduler": "streamK",
    }

    sch_sig = generate_sch_sig(schedule_config)
    for orig, terse in kernel_terse_names_replace.items():
        sch_sig = sch_sig.replace(orig, terse)
    return sch_sig


# unique type_name
def generate_type_signature(kernel_types: TypeConfig):
    a = VLLMDataTypeNames[kernel_types.a]
    b = VLLMDataTypeNames[kernel_types.b]
    d = VLLMDataTypeNames[kernel_types.d]
    accumulator = VLLMDataTypeNames[kernel_types.accumulator]
    element_scale = VLLMDataTypeNames[kernel_types.b_scale]
    element_zeropoint = VLLMDataTypeNames[
        kernel_types.b_zeropoint]

    return (f"{a}{b}{d}"
            f"{accumulator}{element_scale}{element_zeropoint}")


# non-unique shorter type_name
def generate_terse_type_signature(kernel_types: TypeConfig):
    a = VLLMDataTypeNames[kernel_types.a]
    b = VLLMDataTypeNames[kernel_types.b]

    return f"{a}{b}"


def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)

def to_cute_constant(value: List[int]):

    def _to_cute_constant(value: int):
        if is_power_of_two(value):
            return f"_{value}"
        else:
            return f"Int<{value}>"

    if isinstance(value, Iterable):
        return [_to_cute_constant(value) for value in value]
    else:
        return _to_cute_constant(value)


def unique_schedules(impl_configs: List[ImplConfig]):
    return list(set(sch
                    for impl_config in impl_configs 
                    for sch in impl_config.schedules))

def unsigned_type_with_bitwidth(num_bits):
    return {
        4: DataType.u4,
        8: DataType.u8,
        16: DataType.u16,
        32: DataType.u32,
        64: DataType.u64,
    }[num_bits]



template_globals = {
    "void": DataType.void,
    "DataTypeTag": VLLMDataTypeTag,
    "VLLMScalarTypeTag": VLLMDataTypeVLLMScalarTypeTag,
    "TorchTypeTag": VLLMDataTypeTorchDataTypeTag,
    "KernelScheduleTag": VLLMKernelScheduleTag,
    "EpilogueScheduleTag": EpilogueScheduleTag,
    "TileSchedulerTag": TileSchedulerTag,
    "to_cute_constant": to_cute_constant,
    "gen_sch_sig": generate_terse_sch_sig,
    "gen_type_sig": generate_type_signature,
    "unique_schedules": unique_schedules,
    "unsigned_type_with_bitwidth": unsigned_type_with_bitwidth,
}


def create_template(template_str):
    template = jinja2.Template(template_str)
    template.globals.update(template_globals)
    return template


mm_dispatch_template = create_template(DISPATCH_TEMPLATE)
mm_impl_template = create_template(IMPL_TEMPLATE)
prepack_dispatch_template = create_template(PREPACK_TEMPLATE)


def create_sources(impl_configs: List[ImplConfig], num_impl_files=16):
    sources = []

    sources.append((
        f"machete_mm_dispatch",
        mm_dispatch_template.render(impl_configs=impl_configs),
    ))
    
    prepack_types = [
        PrepackTypeConfig(
            a=impl_config.types.a, 
            b_num_bits=VLLMDataTypeSize[impl_config.types.b],
            convert=impl_config.types.b_scale 
            if impl_config.types.b_scale == DataType.void 
            else impl_config.types.b_scale,
            accumulator=impl_config.types.accumulator,
        ) for impl_config in impl_configs
    ]

    def prepacked_type_key(prepack_type: PrepackTypeConfig):
        # For now we we can just use the first accumulator type seen since
        # the tensor core shapes/layouts don't vary based on accumulator
        # type so we can generate less code this way
        return (prepack_type.a, prepack_type.b_num_bits, prepack_type.convert)

    unique_prepack_types = []
    prepack_types_seen = set()
    for prepack_type in prepack_types:
        key = prepacked_type_key(prepack_type)
        if key not in prepack_types_seen:
            unique_prepack_types.append(prepack_type)
            prepack_types_seen.add(key)

    sources.append((
        f"machete_prepack",
        prepack_dispatch_template.render(
            types=unique_prepack_types,
        ),
    ))
    
    # Split up impls across files
    num_impls = reduce(lambda x, y: x + len(y.schedules), impl_configs, 0)
    num_impls_per_file = math.ceil(num_impls / num_impl_files)
    
    files_impls: List[List[ImplConfig]] = [[]]
    
    curr_num_impls_assigned = 0
    curr_impl_in_file = 0
    curr_impl_configs = deepcopy(list(reversed(impl_configs)))

    while curr_num_impls_assigned < num_impls:
        room_left_in_file = num_impls_per_file - curr_impl_in_file
        if room_left_in_file == 0:
            files_impls.append([])
            room_left_in_file = num_impls_per_file
            curr_impl_in_file = 0
        
        curr_ic = curr_impl_configs[-1]
        if len(curr_ic.schedules) >= room_left_in_file:
            # Break appart the current impl config
            tmp_ic = deepcopy(curr_ic)
            tmp_ic.schedules = curr_ic.schedules[:room_left_in_file]
            curr_ic.schedules = curr_ic.schedules[room_left_in_file:]
            files_impls[-1].append(tmp_ic)
        else:
            files_impls[-1].append(curr_ic)
            curr_impl_configs.pop()
        curr_num_impls_assigned += len(files_impls[-1][-1].schedules)
        curr_impl_in_file += len(files_impls[-1][-1].schedules)
        
    for part, file_impls in enumerate(files_impls):
        sources.append((
            f"machete_mm_impl_part{part+1}",
            mm_impl_template.render(impl_configs=file_impls),
        ))

    return sources


def generate():
    # See csrc/quantization/machete/Readme.md, the Codegeneration for more info
    # about how this works
    SCRIPT_DIR = os.path.dirname(__file__)

    schedules = [
        ScheduleConfig(
            tile_shape_mn=tile_shape_mn,
            cluster_shape_mnk=cluster_shape_mnk,
            kernel_schedule=kernel_schedule,
            epilogue_schedule=epilogue_schedule,
            tile_scheduler=tile_scheduler,
        ) for tile_shape_mn, cluster_shape_mnk in (
            ((128, 16), (1, 1, 1)),
            ((128, 32), (1, 1, 1)),
            ((128, 64), (1, 1, 1)),
            ((128, 128), (1, 1, 1)),
        ) for kernel_schedule in (TmaMI, ) for epilogue_schedule in (TmaCoop, )
        for tile_scheduler in (TileSchedulerType.StreamK, )
    ]

    # For now we use the same heuristic for all types
    default_heuristic = [
        ("M > 64",
         ScheduleConfig(
             tile_shape_mn=(128, 128),
             cluster_shape_mnk=(1, 1, 1),
             kernel_schedule=TmaMI,
             epilogue_schedule=TmaCoop,
             tile_scheduler=TileSchedulerType.StreamK,
         )),
        ("M > 32",
         ScheduleConfig(
             tile_shape_mn=(128, 64),
             cluster_shape_mnk=(1, 1, 1),
             kernel_schedule=TmaMI,
             epilogue_schedule=TmaCoop,
             tile_scheduler=TileSchedulerType.StreamK,
         )),
        ("M > 16",
         ScheduleConfig(
             tile_shape_mn=(128, 32),
             cluster_shape_mnk=(1, 1, 1),
             kernel_schedule=TmaMI,
             epilogue_schedule=TmaCoop,
             tile_scheduler=TileSchedulerType.StreamK,
         )),
        (None,
         ScheduleConfig(tile_shape_mn=(128, 16),
                        cluster_shape_mnk=(1, 1, 1),
                        kernel_schedule=TmaMI,
                        epilogue_schedule=TmaCoop,
                        tile_scheduler=TileSchedulerType.StreamK))
    ]

    impl_configs = []

    GPTQ_kernel_types = list(
        (TypeConfig(
            a=a, b=b,
            b_scale=a,
            b_zeropoint=DataType.void,
            d=a, accumulator=DataType.f32,
        ) for b in (VLLMDataType.u4b8, VLLMDataType.u8b128)
         for a in (DataType.f16, DataType.bf16)))

    impl_configs += [
        ImplConfig(x[0], x[1], x[2])
        for x in zip(GPTQ_kernel_types, 
                     itertools.repeat(schedules),
                     itertools.repeat(default_heuristic))
    ]

    AWQ_kernel_types = list(
        (TypeConfig(
            a=a, b=b, 
            b_scale=a, 
            b_zeropoint=a,
            d=a, accumulator=DataType.f32,
        ) for b in (DataType.u4, DataType.u8)
         for a in (DataType.f16, DataType.bf16)))

    # impl_configs += [
    #     ImplConfig(x[0], x[1], x[2])
    #     for x in zip(AWQ_kernel_types, 
    #                  itertools.repeat(schedules),
    #                  itertools.repeat(default_heuristic))
    # ]
    
    QQQ_kernel_types = [
        *(TypeConfig(
            a=DataType.s8, 
            b=VLLMDataType.u4b8,
            b_scale=DataType.f16,
            b_zeropoint=DataType.void,
            d=d,
            accumulator=DataType.s32,
        ) for d in (DataType.s32, DataType.f16)),
        *(TypeConfig(
            a=DataType.e4m3, 
            b=VLLMDataType.u4b8,
            b_scale=DataType.f16,
            b_zeropoint=DataType.void,
            d=d,
            accumulator=DataType.f32,
        ) for d in (DataType.f32, DataType.f16)),
    ]

    impl_configs += [
        ImplConfig(x[0], x[1], x[2])
        for x in zip(QQQ_kernel_types, itertools.repeat(schedules),
                     itertools.repeat(default_heuristic))
    ]

    output_dir = os.path.join(SCRIPT_DIR, "generated")

    # Delete the "generated" directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create the "generated" directory
    os.makedirs(output_dir)

    # Render each group of configurations into separate files
    for filename, code in create_sources(impl_configs):
        filepath = os.path.join(output_dir, f"{filename}.cu")
        with open(filepath, "w") as output_file:
            output_file.write(code)
        print(f"Rendered template to {filepath}")


if __name__ == "__main__":
    generate()
