"""The stack's network surface: one loopback address for everything that
listens, and the checks that keep web pages away from the local ports."""

# Every listener binds this address: both model servers, the client proxy
# and the app's local API. Nothing is reachable from the network.
HOST = "127.0.0.1"


def loopback_host(headers, port: int) -> bool:
    """Does the request address this machine by name? A web page reaching a
    local port through DNS rebinding names its own site in Host instead."""
    host = (headers.get("Host") or "").strip().lower()
    return host in (f"{HOST}:{port}", f"localhost:{port}")


def from_web_page(headers) -> bool:
    """Browsers send Origin with every request a page makes to another
    site; programs on this machine — URLSession, curl, agents — send none."""
    return bool(headers.get("Origin"))
