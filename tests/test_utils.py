import asyncio
import sys
from typing import (TYPE_CHECKING, Any, AsyncIterator, Awaitable, Protocol,
                    Tuple, TypeVar)

import pytest

from vllm.utils import merge_async_iterators

if sys.version_info < (3, 10):
    if TYPE_CHECKING:
        _AwaitableT = TypeVar("_AwaitableT", bound=Awaitable[Any])
        _AwaitableT_co = TypeVar("_AwaitableT_co",
                                 bound=Awaitable[Any],
                                 covariant=True)

        class _SupportsSynchronousAnext(Protocol[_AwaitableT_co]):

            def __anext__(self) -> _AwaitableT_co:
                ...

    def anext(i: _SupportsSynchronousAnext[_AwaitableT], /) -> _AwaitableT:
        return i.__anext__()


@pytest.mark.asyncio
async def test_merge_async_iterators():

    async def mock_async_iterator(idx: int) -> AsyncIterator[str]:
        try:
            while True:
                yield f"item from iterator {idx}"
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    iterators = [mock_async_iterator(i) for i in range(3)]
    merged_iterator: AsyncIterator[Tuple[int, str]] = merge_async_iterators(
        *iterators)

    async def stream_output(generator: AsyncIterator[Tuple[int, str]]):
        async for idx, output in generator:
            print(f"idx: {idx}, output: {output}")

    task = asyncio.create_task(stream_output(merged_iterator))
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for iterator in iterators:
        try:
            await asyncio.wait_for(anext(iterator), 1)
        except StopAsyncIteration:
            # All iterators should be cancelled and print this message.
            print("Iterator was cancelled normally")
        except (Exception, asyncio.CancelledError) as e:
            raise AssertionError() from e
