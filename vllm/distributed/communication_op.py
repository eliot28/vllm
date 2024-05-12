import pickle
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .parallel_state import (get_cpu_world_group,
                             get_tensor_model_parallel_group,
                             get_tensor_model_parallel_rank,
                             get_tensor_model_parallel_world_size,
                             get_tp_pynccl_communicator)


@contextmanager
def graph_capture_mode():
    # In graph capture, we have to be very careful about the collective
    # operations. The current status is:
    #     allreduce \ Mode   |  Eager  |  Graph  |
    # --------------------------------------------
    # custom allreduce       | enabled | enabled |
    # PyNccl                 | disabled| enabled |
    # torch.distributed      | enabled | disabled|
    #
    # Note that custom allreduce will have a runtime check, if the tensor size
    # is too large, it will fallback to the next available option.
    pynccl_comm = get_tp_pynccl_communicator()
    assert pynccl_comm is not None
    with pynccl_comm.change_state(enable=True,
                                  stream=torch.cuda.current_stream()):
        yield


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group.

    NOTE: This operation will be applied in-place on the input tensor if
    disable_custom_all_reduce is set to True. Otherwise, this operation may or
    may not be applied in place depending on whether custom all reduce is
    invoked for a particular tensor, which further depends on the tensor size
    and GPU topology.

    TLDR: always assume this function modifies its input, but use the return
    value as the output.
    """
    from vllm.distributed.device_communicators.custom_all_reduce import (
        custom_all_reduce)

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    out = custom_all_reduce(input_)
    if out is not None:
        return out
    pynccl_comm = get_tp_pynccl_communicator()
    if (pynccl_comm is not None and not pynccl_comm.disabled):
        pynccl_comm.all_reduce(input_)
    else:
        torch.distributed.all_reduce(input_,
                                     group=get_tensor_model_parallel_group())
    return input_


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    input_size = input_.size()
    # Allocate output tensor.
    output_tensor = torch.empty((world_size, ) + input_size,
                                dtype=input_.dtype,
                                device=input_.device)
    # All-gather.
    torch.distributed.all_gather_into_tensor(
        output_tensor, input_, group=get_tensor_model_parallel_group())
    # Reshape
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(input_size[:dim] +
                                          (world_size * input_size[dim], ) +
                                          input_size[dim + 1:])
    return output_tensor


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> torch.Tensor:
    """Gather the input tensor across model parallel group.

    NOTE: We assume that the input tensor is on the same device across
    all the ranks.
    """
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    # Allocate output tensor.
    if get_tensor_model_parallel_rank() == dst:
        gather_list = [torch.empty_like(input_) for _ in range(world_size)]
    else:
        gather_list = None
    # Gather.
    torch.distributed.gather(input_,
                             gather_list,
                             dst=dst,
                             group=get_tensor_model_parallel_group())
    if get_tensor_model_parallel_rank() == dst:
        output_tensor = torch.cat(gather_list, dim=dim)
    else:
        output_tensor = None
    return output_tensor


def broadcast(input_: torch.Tensor,
              src: int = 0,
              group: Optional[ProcessGroup] = None):
    """Broadcast the input tensor."""
    group = group or torch.distributed.group.WORLD
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return input_
    # Broadcast.
    torch.distributed.broadcast(input_, src=src, group=group)
    return input_


def broadcast_object_list(obj_list: List[Any],
                          src: int = 0,
                          group: Optional[ProcessGroup] = None):
    """Broadcast the input object list."""
    group = group or torch.distributed.group.WORLD
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return obj_list
    # Broadcast.
    torch.distributed.broadcast_object_list(obj_list, src=src, group=group)
    return obj_list


def _split_tensor_dict(
    tensor_dict: Dict[str, Union[torch.Tensor, Any]],
    keys: Optional[List[str]] = None,
) -> Tuple[List[Any], List[torch.Tensor]]:
    """Split the tensor dictionary into two parts:
    1. A list of (key, value) pairs. If the value is a tensor, it is replaced
         by its metadata. If keys are provided, only return the value.
    2. A list of tensors.

    `keys` is used to specify the keys to be included in the metadata list,
    which can make sure the order of the metadata list is consistent across
    different ranks.
    """
    from vllm import TensorMeta  # import here to avoid circular import
    metadata_list = []
    tensor_list = []
    used_keys = keys or tensor_dict.keys()
    for key in used_keys:
        value = tensor_dict[key]
        if isinstance(value, torch.Tensor):
            # Note: we cannot use `value.device` here,
            # because it contains not only the device type but also the device
            # index (e.g. "cuda:0"). We only need the device type.
            # receiving side will set the device index.
            device = "cpu" if value.is_cpu else "cuda"
            metadata_list.append(
                (key, TensorMeta(device, value.dtype, value.size())))
            tensor_list.append(value)
        else:
            metadata_list.append((key, value))
    if keys is not None:
        metadata_list = [value for key, value in metadata_list]
    return metadata_list, tensor_list


class TensorDictWithBoundedMetadata:
    """
    In the normal case, when we broadcast Python objects, we need two
    collective operations: one to broadcast the length of the object after
    serialization, and one to broadcast the serialized object.

    This class represents a dictionary of tensors with bounded metadata.
    The upperbound of the buffer size is known a priori. Therefore, we can
    pre-allocate a buffer for the metadata, and invoke only one collective
    operation to broadcast the metadata.

    The main benefit is we can save one broadcast call.

    Note: it depends on the feature of Python pickle that the serialized
    data contains a marker for the end of the data. Therefore, as long as
    the buffer size is larger than the serialized data, we can guarantee
    the deserialization is correct.
    """

    @classmethod
    def get_max_buffer_size_for_metadata(cls):
        metadata_list = cls.get_example_metadata_list()
        # Note: we only need the values of the metadata list.
        values = [value for key, value in metadata_list]
        metadata_list_bytes = pickle.dumps(values)
        ALIGN_BYTES = 128
        return ((len(metadata_list_bytes) + ALIGN_BYTES - 1) //
                ALIGN_BYTES) * ALIGN_BYTES

    # ===== subclass overrides starts =====
    # subclass should implement the `__init__` method, and set the `fields`
    # attribute to a list of field names, and implement the
    # `get_example_metadata` class method to provide an example metadata for
    # the fields. This is used to calculate the buffer size.
    fields: List[str]

    def __init__(self):
        pass

    @classmethod
    def get_example_metadata_list(cls):
        # Note: in general, if the example data contains cuda tensor,
        # use cpu tensor here to avoid creating cuda context during
        # the initialization of the class. The estimation of the buffer size
        # might be inaccurate (by one byte per field), but it is fine because
        # the buffer size will be aligned to 256 bytes.
        return {}

    # ===== subclass overrides ends =====
    # for type annotation
    size_upper_bound: int
    buffer: bytearray
    buffer_tensor: torch.Tensor

    def __init_subclass__(subclass):
        assert hasattr(subclass, "fields"), (
            f"Expecting a `fields` attribute in the subclass {subclass}")
        subclass.size_upper_bound = subclass.get_max_buffer_size_for_metadata()
        subclass.buffer = bytearray(subclass.size_upper_bound)
        subclass.buffer_tensor = torch.frombuffer(memoryview(subclass.buffer),
                                                  dtype=torch.uint8)


T = TypeVar("T", bound=TensorDictWithBoundedMetadata)


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None,
    src: int = 0,
    group: Optional[ProcessGroup] = None,
    metadata_group: Optional[ProcessGroup] = None,
    cls: Optional[Type[T]] = None,
) -> Union[Dict[Any, Union[torch.Tensor, Any]], T]:
    """Broadcast the input tensor dictionary.
    `group` is used to broadcast the tensors, while `metadata_group` is used
     to broadcast the metadata of the dict (e.g. dict structure, tensor sizes,
     dtypes). If `cls` is provided, we can know the length of the metadata
     roughly and allocate a buffer for it, then broadcasting metadata requires
     only one broadcast call. Otherwise, we need to broadcast the metadata
     length first, then broadcast the metadata.
    """
    group = group or torch.distributed.group.WORLD
    metadata_group = metadata_group or get_cpu_world_group()
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        assert tensor_dict is not None
        return tensor_dict

    from vllm import TensorMeta  # import here to avoid circular import

    rank = torch.distributed.get_rank()
    if rank == src:
        metadata_list: List[Tuple[Any, Any]] = []
        assert isinstance(
            tensor_dict,
            dict), (f"Expecting a dictionary, got {type(tensor_dict)}")
        if cls is not None:
            metadata_list, tensor_list = _split_tensor_dict(tensor_dict,
                                                            keys=cls.fields)
            s = pickle.dumps(metadata_list)
            cls.buffer_tensor[:len(s)].copy_(
                torch.frombuffer(s, dtype=torch.uint8))
            dist.broadcast(cls.buffer_tensor, src=src, group=metadata_group)
        else:
            metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
            # `metadata_list` lives in CPU memory.
            # `broadcast_object_list` involves serialization and
            # deserialization, all happening on CPU. Therefore,
            # we can use the CPU group.
            dist.broadcast_object_list([metadata_list],
                                       src=src,
                                       group=metadata_group)
        async_handles = []
        for tensor in tensor_list:
            if tensor.numel() == 0:
                # Skip broadcasting empty tensors.
                continue
            if tensor.is_cpu:
                # use metadata_group for CPU tensors
                handle = dist.broadcast(tensor,
                                        src=src,
                                        group=metadata_group,
                                        async_op=True)
            else:
                # use group for GPU tensors
                handle = dist.broadcast(tensor,
                                        src=src,
                                        group=group,
                                        async_op=True)
            async_handles.append(handle)
        for async_handle in async_handles:
            async_handle.wait()

    else:
        if cls is None:
            container = [None]
            dist.broadcast_object_list(container,
                                       src=src,
                                       group=metadata_group)
            recv_metadata_list = container[0]
            assert recv_metadata_list is not None
        else:
            dist.broadcast(cls.buffer_tensor, src=src, group=metadata_group)
            recv_value_list = pickle.loads(memoryview(cls.buffer))
            recv_metadata_list = list(zip(cls.fields, recv_value_list))
        tensor_dict = {}
        async_handles = []
        for key, value in recv_metadata_list:
            if isinstance(value, TensorMeta):
                tensor = torch.empty(value.size,
                                     dtype=value.dtype,
                                     device=value.device)
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    tensor_dict[key] = tensor
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    handle = dist.broadcast(tensor,
                                            src=src,
                                            group=metadata_group,
                                            async_op=True)
                else:
                    # use group for GPU tensors
                    handle = dist.broadcast(tensor,
                                            src=src,
                                            group=group,
                                            async_op=True)
                async_handles.append(handle)
                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
        for async_handle in async_handles:
            async_handle.wait()
    if cls is not None:
        return cls(**tensor_dict)
    return tensor_dict
