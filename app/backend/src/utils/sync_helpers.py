import asyncio
from functools import wraps


def sync_agen_method(agen_method):
    @wraps(agen_method)
    def wrapper(self, *args, **kwargs):
        agen = agen_method(self, *args, **kwargs)
        try:
            while True:
                yield asyncio.run(agen.__anext__())
        except StopAsyncIteration:
            return
    return wrapper


def sync_method(async_method):
    @wraps(async_method)
    def wrapper(self, *args, **kwargs):
        return asyncio.run(async_method(self, *args, **kwargs))
    return wrapper
