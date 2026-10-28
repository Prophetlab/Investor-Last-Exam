"""Reference-counted HTTPX transports for the SDK's public injection hook."""

from threading import Lock

import httpx

_lock = Lock()
_clients: dict[tuple[float, int, int], tuple[httpx.Client, int]] = {}


def acquire_client(timeout: float, connections: int, keepalive: int) -> httpx.Client:
    if connections < 1 or not 0 <= keepalive <= connections:
        raise ValueError("invalid Data Lake connection pool limits")
    key = (timeout, connections, keepalive)
    with _lock:
        if key in _clients:
            client, count = _clients[key]
            _clients[key] = (client, count + 1)
            return client
        client = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=connections, max_keepalive_connections=keepalive
            ),
            follow_redirects=False,
        )
        _clients[key] = (client, 1)
        return client


def release_client(client: httpx.Client) -> None:
    with _lock:
        for key, (registered, count) in tuple(_clients.items()):
            if registered is client:
                if count == 1:
                    del _clients[key]
                    client.close()
                else:
                    _clients[key] = (client, count - 1)
                return
