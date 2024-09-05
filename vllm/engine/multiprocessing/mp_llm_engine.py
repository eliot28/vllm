import pickle
from contextlib import contextmanager
from typing import Iterator, List, Optional, Tuple, Type, Union

import cloudpickle
import ray
import zmq

from vllm import AsyncEngineArgs, AsyncLLMEngine, LLMEngine
from vllm.config import (DecodingConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.engine.multiprocessing import (VLLM_RPC_SUCCESS_STR, RPCAbortRequest,
                                         RPCGenerateRequest, RPCStartupRequest,
                                         RPCUtilityRequest)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.usage.usage_lib import UsageContext

CONFIG_TYPE = Union[ModelConfig, DecodingConfig, ParallelConfig,
                    SchedulerConfig, LoRAConfig]

logger = init_logger(__name__)

POLLING_TIMEOUT_MS = 10000

HEALTHY_RESPONSE = (pickle.dumps(VLLM_RPC_SUCCESS_STR), )


class MPLLMEngine:
    """A multiprocessing wrapper for :class:`LLMEngine`.

    This class is used to wrap the :class:`LLMEngine` class to enable use
    in asynchronous manner. It runs a background loop and uses zeromq to 
    receive new requests and stream outputs incrementally to another process.
    
    The :class:`LLMEngine` is kicked off when a new RPCGenerateRequest 
    is received by the input_socket.
    
    The self.engine_loop checks the input_socket for new requests,
    adds them to the LLMEngine if there are any, calls the internal
    :class:`LLMEngine.step()` and sends the RequestOutputs back over
    the output_socket. 

    Args:
        worker_use_ray: Whether to use Ray for model workers. Required for
            distributed execution. Should be the same as
            `parallel_config.worker_use_ray`.
        engine_use_ray: Whether to make LLMEngine a Ray actor. If so, the
            async frontend will be executed in a separate process as the
            model workers.
        async_engine_args: AsyncLLMEngine args
        log_requests: Whether to log the requests.
    """

    _engine_class: Type[LLMEngine] = LLMEngine

    def __init__(self,
                 worker_use_ray: bool,
                 engine_use_ray: bool,
                 *args,
                 ipc_path: str,
                 use_async_sockets: bool,
                 log_requests: bool = True,
                 **kwargs) -> None:

        if engine_use_ray:
            raise NotImplementedError("Not yet supported.")

        self.worker_use_ray = worker_use_ray
        self.engine_use_ray = engine_use_ray
        self.log_requests = log_requests
        self.engine = self._init_engine(*args, **kwargs)

        self.ctx = zmq.Context()  # type: ignore[attr-defined]

        # Receive input from the client.
        self.input_socket = self.ctx.socket(zmq.constants.PULL)
        self.input_socket.bind(f"{ipc_path}_input_socket")

        # Send output stream back to client.
        self.output_socket = self.ctx.socket(zmq.constants.PUSH)
        self.output_socket.bind(f"{ipc_path}_output_socket")

        # Send health status back to client.
        self.health_socket = self.ctx.socket(zmq.constants.PUSH)
        self.health_socket.bind(f"{ipc_path}_health_socket")

        # IPC path for the data socket.
        self.data_ipc_path = f"{ipc_path}_data_socket"

        # Indicates if we do socket read/write async with
        # the GPU forward pass
        self.use_async_sockets = use_async_sockets

    @classmethod
    def from_engine_args(cls, engine_args: AsyncEngineArgs,
                         usage_context: UsageContext, ipc_path: str):
        """Creates an MPLLMEngine from the engine arguments."""

        engine_config = engine_args.create_engine_config()

        if engine_args.engine_use_ray:
            from vllm.executor import ray_utils
            ray_utils.assert_ray_available()

        # TODO: better abstraction?
        executor_class = AsyncLLMEngine._get_executor_cls(engine_config)

        return cls(
            executor_class.uses_ray,
            engine_args.engine_use_ray,
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            ipc_path=ipc_path,
            use_async_sockets=engine_config.model_config.use_async_output_proc)

    def cleanup(self):
        """Cleanup zeromq state on shutdown."""
        self.input_socket.close()
        self.output_socket.close()
        self.health_socket.close()
        self.ctx.destroy(linger=0)
        del self.engine

    def _init_engine(self, *args,
                     **kwargs) -> Union[LLMEngine, "ray.ObjectRef"]:
        """Initialize the LLMEngine"""

        if not self.engine_use_ray:
            engine_class = self._engine_class
        elif self.worker_use_ray:
            engine_class = ray.remote(num_cpus=0)(self._engine_class).remote
        else:
            raise NotImplementedError("Not supported yet!")
        return engine_class(*args, **kwargs)

    def run_background_loop(self):
        """Entrypoint that kicks off the background processing loop."""

        # Allow RPCClient to query data in startup phase.
        self.run_startup_loop()

        # Kick off core processing loop.
        self.run_engine_loop()

    @contextmanager
    def make_data_socket(
            self) -> Iterator[zmq.Socket]:  # type: ignore[name-defined]
        socket = self.ctx.socket(zmq.constants.ROUTER)
        try:
            socket.bind(self.data_ipc_path)
            yield socket
        finally:
            socket.close(linger=0)

    def run_startup_loop(self) -> None:
        """Loop over startup RPCStatupRequest from RPCClient."""

        with self.make_data_socket() as socket:

            # Loop until the RPCClient has all the data it needs.
            client_is_ready = False
            while not client_is_ready:
                try:
                    identity, message = socket.recv_multipart(copy=False)
                    request: RPCStartupRequest = pickle.loads(message.buffer)

                    # Handle the query from the Client.
                    if request == RPCStartupRequest.GET_MODEL_CONFIG:
                        response = self.engine.get_model_config()
                    elif request == RPCStartupRequest.GET_DECODING_CONFIG:
                        response = self.engine.get_decoding_config()
                    elif request == RPCStartupRequest.GET_LORA_CONFIG:
                        response = self.engine.get_lora_config()
                    elif request == RPCStartupRequest.GET_SCHEDULER_CONFIG:
                        response = self.engine.get_scheduler_config()
                    elif request == RPCStartupRequest.GET_PARALLEL_CONFIG:
                        response = self.engine.get_parallel_config()
                    elif request == RPCStartupRequest.GET_TRACING_ENABLED:
                        response = self.engine.is_tracing_enabled()
                    elif request == RPCStartupRequest.IS_SERVER_READY:
                        response = VLLM_RPC_SUCCESS_STR
                    elif request == RPCStartupRequest.CLIENT_IS_READY:
                        response = VLLM_RPC_SUCCESS_STR
                        # Breakout of loop once client is ready.
                        client_is_ready = True

                    socket.send_multipart((identity, pickle.dumps(response)),
                                          copy=False)

                except Exception as e:
                    socket.send_multipart((identity, pickle.dumps(e)),
                                          copy=False)

    def run_engine_loop(self) -> None:
        if self.use_async_sockets:
            self.engine.process_request_outputs_callback = \
                self.stream_outputs_and_get_inputs

        while True:
            # Block until there is a new request.
            if not self.engine.has_unfinished_requests():
                self.wait_for_new_input()

            # Handle any new input from the input socket.
            self.maybe_handle_new_input()

            # Engine step.
            try:
                request_outputs = self.engine.step()
            except Exception as e:
                # Fail all requests with this exception.
                request_outputs = (None, e)

            if not self.use_async_sockets:
                # Stream results to output socket.
                self.send_outputs(request_outputs)

    def wait_for_new_input(self):
        while self.input_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
            logger.debug("Waiting for new request.")

    def send_outputs(self, request_outputs: Union[List[RequestOutput],
                                                  Tuple[Optional[str],
                                                        BaseException]]):
        output_bytes = pickle.dumps(request_outputs)
        self.output_socket.send_multipart((output_bytes, ), copy=False)

    def stream_outputs_and_get_inputs(self,
                                      request_outputs: List[RequestOutput]):
        self.send_outputs(request_outputs)
        self.maybe_handle_new_input()

    def ack_check_health(self):
        self.health_socket.send_multipart(HEALTHY_RESPONSE, copy=False)

    def maybe_handle_new_input(self):
        """Handle new input with non-blocking IO"""
        while self.input_socket.poll(timeout=0) != 0:
            message = self.input_socket.recv(copy=False)
            request = cloudpickle.loads(message.buffer)

            if isinstance(request, RPCGenerateRequest):
                self._handle_generate_request(request)
            elif isinstance(request, RPCAbortRequest):
                self._handle_abort_request(request)
            elif isinstance(request, RPCUtilityRequest):
                self._handle_utility_request(request)
            else:
                raise ValueError(f"Unknown RPCRequest: {request}")

    def _handle_generate_request(self, request: RPCGenerateRequest):
        request_id = request.request_id
        try:
            self.engine.add_request(
                request_id=request_id,
                inputs=request.inputs,
                params=request.sampling_params,
                lora_request=request.lora_request,
                trace_headers=request.trace_headers,
                prompt_adapter_request=request.prompt_adapter_request,
            )
        except Exception as e:
            self.engine.abort_request(request_id)
            self.send_outputs((request_id, e))

    def _handle_abort_request(self, request: RPCAbortRequest):
        self.engine.abort_request(request.request_id)

    def _handle_utility_request(self, request: RPCUtilityRequest):
        if request == RPCUtilityRequest.DO_LOG_STATS:
            self.engine.do_log_stats()
        elif request == RPCUtilityRequest.CHECK_HEALTH:
            self.engine.check_health()
            self.ack_check_health()


def run_mp_engine(engine_args: AsyncEngineArgs, usage_context: UsageContext,
                  ipc_path: str):
    engine = MPLLMEngine.from_engine_args(engine_args=engine_args,
                                          usage_context=usage_context,
                                          ipc_path=ipc_path)

    engine.run_background_loop()
