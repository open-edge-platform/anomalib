import asyncio
import concurrent.futures
from collections.abc import Callable
from typing import Any


async def run_in_process_pool(func: Callable, *args, **kwargs) -> Any:
    """
    Run a synchronous function in a separate process using ProcessPoolExecutor.
    This is useful for CPU-bound tasks that would block the event loop if run directly.

    This function safely handles the asyncio loop by using a different approach that
    doesn't rely on get_running_loop() which can fail in certain contexts.

    Args:
        func (callable): The synchronous function to be executed in a separate process.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the function execution.
    """
    # Try to get the running loop first, fall back to event loop if needed
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, get the event loop
        loop = asyncio.get_event_loop()

    with concurrent.futures.ProcessPoolExecutor() as pool:
        return await loop.run_in_executor(pool, func, *args, **kwargs)
