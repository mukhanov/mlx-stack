"""Client proxy (:4000): the one local endpoint every harness talks to —
pi, omp and opencode in OpenAI format, Claude Code in Anthropic format.

OpenAI routes are relayed to the model server as they came, except for the
model name: mlx_lm.server loads whatever repository a request names, so the
stack's aliases are resolved here ("local" -> the main model, "small" ->
the helper) and any other name is refused. `/v1/messages` (Anthropic
Messages) is translated to a chat completion and the reply is translated
back, SSE included.

Runs as a thread of the stack controller: no extra process.
"""
import hmac
import json
import time
import uuid
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCAL_MODEL = "local"        # the main model's alias in every client config
SMALL_MODEL = "small"        # the helper's alias in every client config
MESSAGES_PATH = "/v1/messages"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
MODELS_PATH = "/v1/models"
HEALTH_PATHS = ("/health", "/health/liveliness")
FORWARDED_HEADERS = ("content-type", "accept", "user-agent")

# OpenAI finish_reason -> Anthropic stop_reason
STOP_REASONS = {"stop": "end_turn", "length": "max_tokens",
                "tool_calls": "tool_use", "function_call": "tool_use",
                "content_filter": "refusal"}
# model-server status -> Anthropic error type
ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error",
               403: "permission_error", 404: "not_found_error",
               413: "request_too_large", 429: "rate_limit_error",
               503: "overloaded_error"}
# Anthropic clients echo thinking blocks back with their signature; the
# proxy drops incoming thinking blocks, so any opaque string will do
THINKING_SIGNATURE = "local-llm-proxy"

_KEEPALIVE = b": keepalive "


def build_routes(main_model: str, main_port: int, small_model: str | None = None,
                 small_port: int = 0) -> dict:
    """Request `model` -> (port, the model name that server loads), in
    listing order. mlx_lm.server loads whatever a request names, so aliases
    must be resolved here — and anything else refused, or a stray name
    would unload the running model."""
    routes = {main_model: (main_port, main_model),
              LOCAL_MODEL: (main_port, main_model)}
    if small_model and small_port:
        routes[SMALL_MODEL] = (small_port, small_model)
        routes[small_model] = (small_port, small_model)
    return routes


def anthropic_error(kind: str, message: str) -> dict:
    return {"type": "error", "error": {"type": kind, "message": message}}


def openai_error(message: str, code: str | None = None) -> dict:
    return {"error": {"message": message, "type": "invalid_request_error"
                      if code else "server_error", "code": code}}


def error_kind(status: int) -> str:
    return ERROR_TYPES.get(status, "api_error")


def upstream_error(data: bytes) -> str:
    """The message of a model-server error body: mlx_lm.server answers
    {"error": "..."}, llama-server {"error": {"message": "..."}}."""
    try:
        payload = json.loads(data)
    except ValueError:
        payload = None
    if isinstance(payload, dict) and payload.get("error") is not None:
        error = payload["error"]
        if isinstance(error, dict):
            error = error.get("message") or json.dumps(error)
        return str(error)
    return data.decode("utf-8", "replace").strip()[:500] or "HTTP error"


def anthropic_usage(usage: dict | None, output_tokens: int = 0) -> dict:
    """OpenAI usage -> Anthropic. Anthropic's input_tokens exclude the
    cached tokens; OpenAI's prompt_tokens include them."""
    usage = usage or {}
    out = {}
    if usage.get("prompt_tokens") is not None:
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        out["input_tokens"] = max(0, usage["prompt_tokens"] - cached)
        out["cache_creation_input_tokens"] = 0
        out["cache_read_input_tokens"] = cached
    out["output_tokens"] = usage.get("completion_tokens") or output_tokens
    return out


def sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


def _image_url(block: dict) -> str | None:
    source = block.get("source") or {}
    if source.get("type") == "base64" and source.get("data") is not None:
        return f"data:{source.get('media_type', 'image/png')};base64,{source['data']}"
    if source.get("type") == "url" and source.get("url"):
        return source["url"]
    return None


def _text(content) -> str:
    """The text of a string or of a list of Anthropic content blocks."""
    if isinstance(content, str):
        return content
    return "\n\n".join(b["text"] for b in content or []
                       if isinstance(b, dict) and b.get("type") == "text"
                       and b.get("text"))


def _tool_choice(choice) -> object:
    kind = (choice or {}).get("type", "auto")
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool":
        return {"type": "function", "function": {"name": choice.get("name")}}
    return "auto"


def anthropic_to_chat(body: dict, model: str) -> dict:
    """Anthropic Messages request -> OpenAI chat completion request for
    `model` (the name the model server loads).

    tool_result blocks become tool messages right after the assistant turn
    that made the calls; thinking blocks are dropped (the model did not
    produce them)."""
    chat = {"model": model}
    messages = []
    system = _text(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    for message in body.get("messages") or []:
        role, content = message.get("role"), message.get("content")
        if role == "assistant":
            texts, calls = [], []
            for block in content if isinstance(content, list) else []:
                if block.get("type") == "text":
                    texts.append(block.get("text") or "")
                elif block.get("type") == "tool_use":
                    calls.append({
                        "id": block.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {"name": block.get("name") or "",
                                     "arguments": json.dumps(block.get("input") or {})}})
            if texts or calls:
                messages.append({"role": "assistant",
                                 "content": "\n\n".join(texts) or None,
                                 "tool_calls": calls})
            continue
        if role != "user":
            continue
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
            continue
        parts = []   # everything that is not a tool result, in order
        for block in content or []:
            kind = block.get("type")
            if kind == "tool_result":
                inner = block.get("content")
                result = [{"type": "text", "text": _text(inner)}]
                if isinstance(inner, list):
                    result += [{"type": "image_url", "image_url": {"url": url}}
                               for b in inner if isinstance(b, dict)
                               and b.get("type") == "image"
                               and (url := _image_url(b))]
                if block.get("is_error"):
                    result[0]["text"] = "Error: " + (result[0]["text"] or "the tool failed")
                messages.append({"role": "tool",
                                 "tool_call_id": block.get("tool_use_id"),
                                 "content": result if len(result) > 1 else result[0]["text"]})
            elif kind == "text" and block.get("text"):
                parts.append({"type": "text", "text": block["text"]})
            elif kind == "image" and (url := _image_url(block)):
                parts.append({"type": "image_url", "image_url": {"url": url}})
            elif kind == "document":
                parts.append({"type": "text",
                              "text": "[document omitted: this model cannot read files]"})
        if parts:
            images = any(p["type"] == "image_url" for p in parts)
            messages.append({"role": "user",
                             "content": parts if images
                             else "\n\n".join(p["text"] for p in parts)})
    chat["messages"] = messages
    for key in ("max_tokens", "temperature", "top_p", "top_k"):
        if body.get(key) is not None:
            chat[key] = body[key]
    if body.get("stop_sequences"):
        chat["stop"] = body["stop_sequences"]
    if body.get("stream"):
        chat["stream"] = True
        chat["stream_options"] = {"include_usage": True}
    # only client-defined tools: server tools (web_search, bash, …) carry
    # no input_schema and nothing here could run them
    tools = [{"type": "function",
              "function": {"name": t["name"], "description": t.get("description") or "",
                           "parameters": t["input_schema"]}}
             for t in body.get("tools") or []
             if isinstance(t, dict) and t.get("name")
             and isinstance(t.get("input_schema"), dict)]
    if tools:
        chat["tools"] = tools
        chat["tool_choice"] = _tool_choice(body.get("tool_choice"))
    return chat


def _tool_input(arguments) -> dict:
    try:
        value = json.loads(arguments) if isinstance(arguments, str) else arguments
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def chat_to_anthropic(resp: dict, model: str) -> dict:
    """OpenAI chat completion -> Anthropic Messages response."""
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = []
    if message.get("reasoning"):
        content.append({"type": "thinking", "thinking": message["reasoning"],
                        "signature": THINKING_SIGNATURE})
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        content.append({"type": "tool_use",
                        "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": function.get("name") or "",
                        "input": _tool_input(function.get("arguments"))})
    if not content:
        content.append({"type": "text", "text": ""})   # a message needs a block
    return {"id": "msg_" + str(resp.get("id", "")).removeprefix("chatcmpl-"),
            "type": "message", "role": "assistant", "model": model,
            "content": content,
            "stop_reason": STOP_REASONS.get(choice.get("finish_reason"), "end_turn"),
            "stop_sequence": None,
            "usage": anthropic_usage(resp.get("usage"))}


def parse_upstream_line(line: bytes):
    """One line of the model server's stream -> ("keepalive", prompt_tokens)
    | ("chunk", dict) | ("done", None) | None."""
    line = line.rstrip(b"\r\n")
    if line.startswith(_KEEPALIVE):
        try:
            return "keepalive", int(line[len(_KEEPALIVE):].decode().partition("/")[2])
        except ValueError:
            return None
    if not line.startswith(b"data:"):
        return None
    payload = line[5:].strip()
    if payload == b"[DONE]":
        return "done", None
    try:
        value = json.loads(payload)
    except ValueError:
        return None
    return ("chunk", value) if isinstance(value, dict) else None


class MessageStream:
    """Anthropic Messages SSE built from parsed chat-completion chunks.

    feed() returns the SSE text for one upstream event, finish() closes the
    message. Text and thinking stream as they arrive; tool calls are
    buffered and written as complete tool_use blocks at the end — mlx_lm
    .server sends each call whole, llama-server in argument fragments."""

    def __init__(self, model):
        self.model = model
        self.id = "msg_" + uuid.uuid4().hex[:24]
        self.started = False
        self.prompt_tokens = 0
        self.usage = None
        self.finish_reason = None
        self.next_index = 0
        self.open_type = None      # "text" | "thinking" | None
        self.open_index = 0
        self.produced = 0          # text/thinking/tool deltas seen
        self.tools = {}            # OpenAI tool-call index -> {id, name, arguments}

    def _start(self) -> list[str]:
        self.started = True
        return [sse({"type": "message_start", "message": {
            "id": self.id, "type": "message", "role": "assistant",
            "model": self.model, "content": [], "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": self.prompt_tokens, "output_tokens": 1,
                      "cache_creation_input_tokens": 0,
                      "cache_read_input_tokens": 0}}})]

    def _open(self, kind: str) -> list[str]:
        self.open_type, self.open_index = kind, self.next_index
        self.next_index += 1
        return [sse({"type": "content_block_start", "index": self.open_index,
                     "content_block": {"type": kind, kind: ""}})]

    def _stop(self) -> list[str]:
        if self.open_type is None:
            return []
        out = []
        if self.open_type == "thinking":
            out.append(sse({"type": "content_block_delta", "index": self.open_index,
                            "delta": {"type": "signature_delta",
                                      "signature": THINKING_SIGNATURE}}))
        out.append(sse({"type": "content_block_stop", "index": self.open_index}))
        self.open_type = None
        return out

    def _delta(self, delta: dict) -> str:
        return sse({"type": "content_block_delta", "index": self.open_index,
                    "delta": delta})

    def feed(self, event: str, value) -> list[str]:
        if event == "keepalive":
            # "processed/total" is the uncached part of the prompt — exactly
            # Anthropic's input_tokens; the cached part comes with the usage
            self.prompt_tokens = value
            out = [] if self.started else self._start()
            out.append(sse({"type": "ping"}))
            return out
        out = [] if self.started else self._start()
        if event != "chunk":
            return out
        for choice in value.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("reasoning"):
                if self.open_type != "thinking":
                    out += self._stop() + self._open("thinking")
                out.append(self._delta({"type": "thinking_delta",
                                        "thinking": delta["reasoning"]}))
                self.produced += 1
            if delta.get("content"):
                if self.open_type != "text":
                    out += self._stop() + self._open("text")
                out.append(self._delta({"type": "text_delta", "text": delta["content"]}))
                self.produced += 1
            for call in delta.get("tool_calls") or []:
                function = call.get("function") or {}
                tool = self.tools.setdefault(call.get("index", len(self.tools)),
                                             {"id": None, "name": None, "arguments": ""})
                tool["id"] = tool["id"] or call.get("id")
                tool["name"] = tool["name"] or function.get("name")
                tool["arguments"] += function.get("arguments") or ""
                self.produced += 1
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        if isinstance(value.get("usage"), dict):
            self.usage = value["usage"]
        return out

    def finish(self) -> list[str]:
        out = [] if self.started else self._start()
        out += self._stop()
        if not self.produced:
            out += self._open("text") + self._stop()
        for tool in self.tools.values():
            out.append(sse({"type": "content_block_start", "index": self.next_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool["id"] or f"toolu_{uuid.uuid4().hex[:24]}",
                                "name": tool["name"] or "", "input": {}}}))
            if tool["arguments"]:
                out.append(sse({"type": "content_block_delta", "index": self.next_index,
                                "delta": {"type": "input_json_delta",
                                          "partial_json": tool["arguments"]}}))
            out.append(sse({"type": "content_block_stop", "index": self.next_index}))
            self.next_index += 1
        out.append(sse({"type": "message_delta",
                        "delta": {"stop_reason": STOP_REASONS.get(self.finish_reason, "end_turn"),
                                  "stop_sequence": None},
                        "usage": (anthropic_usage(self.usage, self.produced)
                                  if self.usage else {"output_tokens": self.produced})}))
        out.append(sse({"type": "message_stop"}))
        return out


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 65      # an idle keep-alive connection releases its thread

    def log_message(self, *args):
        pass            # one access-log line per request instead

    # --- request plumbing ---------------------------------------------------

    def _body(self) -> bytes:
        if (self.headers.get("Transfer-Encoding") or "").lower() != "chunked":
            return self.rfile.read(int(self.headers.get("Content-Length") or 0))
        chunks = []
        while True:
            line = self.rfile.readline(65536).strip()
            if line == b"0":
                while self.rfile.readline(65536).strip():   # trailers
                    pass
                return b"".join(chunks)
            chunks.append(self.rfile.read(int(line.split(b";")[0], 16)))
            self.rfile.read(2)   # the CRLF after each chunk

    def _is_anthropic(self) -> bool:
        return "anthropic-version" in self.headers

    def _authorized(self) -> bool:
        auth = self.headers.get("Authorization") or ""
        key = (auth[7:].strip() if auth[:7].lower() == "bearer "
               else self.headers.get("x-api-key") or "")
        return hmac.compare_digest(key.encode(), self.server.api_key.encode())

    def _reply(self, status: int, payload: dict) -> int:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return len(body)

    def _begin_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")   # no length: the body ends with the socket
        self.end_headers()
        self.close_connection = True

    def _upstream(self, port: int, method: str, body: bytes | None,
                  path: str | None = None):
        conn = HTTPConnection("127.0.0.1", port, timeout=self.server.upstream_timeout)
        headers = {"Accept-Encoding": "identity"}
        for name in FORWARDED_HEADERS:
            if self.headers.get(name):
                headers[name] = self.headers[name]
        conn.request(method, path or self.path, body=body, headers=headers)
        return conn, conn.getresponse()

    def _access(self, t0: float, status: int, model, port: int, size: int,
                stream: bool = False) -> None:
        level = " ERROR" if status >= 500 else " WARN" if status >= 400 else ""
        self.server.access(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} {self.command} {self.path}"
            f" {model or '-'} ->{port or '-'} {status}"
            f" {(time.monotonic() - t0) * 1000:.0f}ms {size}B"
            f"{' stream' if stream else ''}{level}")

    # --- routes ---------------------------------------------------------------

    def do_GET(self):
        t0 = time.monotonic()
        path = self.path.split("?", 1)[0]
        if path in HEALTH_PATHS:
            return self._reply(200, {"status": "ok"})
        if not self._authorized():
            return self._deny(t0)
        if path == MODELS_PATH:
            size = self._reply(200, self._model_list())
            return self._access(t0, 200, None, 0, size)
        self._relay(None, None, t0)

    def do_POST(self):
        t0 = time.monotonic()
        raw = self._body()      # read first: keep-alive needs a drained socket
        if not self._authorized():
            return self._deny(t0)
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = None      # not ours to judge: the model server answers
        model = payload.get("model") if isinstance(payload, dict) else None
        route = self.server.routes.get(model) if isinstance(model, str) else None
        if isinstance(payload, dict) and route is None:
            return self._unknown_model(model, t0)
        if route and model != route[1]:
            # an alias: the model server knows the repository, not the alias
            payload["model"] = route[1]
            raw = json.dumps(payload).encode()
        if self.path.split("?", 1)[0] == MESSAGES_PATH and route:
            return self._messages(raw, model, route, t0)
        self._relay(raw, model, route, t0)


    def _model_list(self) -> dict:
        names = list(self.server.routes)
        if self._is_anthropic():
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            data = [{"type": "model", "id": n, "display_name": f"Local MLX ({n})",
                     "created_at": now} for n in names]
            return {"data": data, "first_id": names[0], "last_id": names[-1],
                    "has_more": False}
        return {"object": "list", "data": [
            {"id": n, "object": "model", "created": self.server.started}
            for n in names]}

    def _deny(self, t0: float) -> None:
        if self._is_anthropic():
            payload = anthropic_error("authentication_error",
                                      "invalid x-api-key or Authorization header")
        else:
            payload = openai_error("Invalid API key")
        self._access(t0, 401, None, 0, self._reply(401, payload))

    def _unknown_model(self, model, t0: float) -> None:
        if self._is_anthropic():
            payload = anthropic_error("not_found_error", f"model: {model}")
        else:
            payload = openai_error(f"The model `{model}` does not exist", "model_not_found")
        self._access(t0, 404, model, 0, self._reply(404, payload))

    def _unreachable(self, model, port: int, t0: float) -> None:
        message = (f"the model server on 127.0.0.1:{port} is not reachable"
                   " (still loading, stopped or crashed)")
        payload = (anthropic_error("overloaded_error", message)
                   if self._is_anthropic() else openai_error(message))
        self._access(t0, 503, model, port, self._reply(503, payload))

    def _relay(self, body: bytes | None, model, route, t0: float) -> None:
        port = route[0] if route else self.server.main_port
        try:
            conn, resp = self._upstream(port, self.command, body)
        except (OSError, HTTPException):
            return self._unreachable(model, port, t0)
        if not resp.getheader("Content-Type", "").startswith("text/event-stream"):
            data = resp.read()
            conn.close()
            self.send_response(resp.status)
            self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
                return self._access(t0, resp.status, model, port, len(data))
            except OSError:
                return self._access(t0, 499, model, port, 0)
        self._begin_stream()
        size, status = 0, resp.status
        try:
            while chunk := resp.read1(65536):
                try:
                    self.wfile.write(chunk)
                except OSError:
                    status = 499       # the client is gone
                    break
                size += len(chunk)
        except (OSError, HTTPException):
            status = 502              # the model server died mid-stream
        finally:
            conn.close()
        self._access(t0, status, model, port, size, stream=True)

    def _messages(self, raw: bytes, requested, route, t0: float) -> None:
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            size = self._reply(400, anthropic_error("invalid_request_error",
                                                     "request body must be a JSON object"))
            return self._access(t0, 400, requested, 0, size)
        port = route[0]
        chat = json.dumps(anthropic_to_chat(body, route[1])).encode()
        try:
            conn, resp = self._upstream(port, "POST", chat, CHAT_COMPLETIONS_PATH)
        except (OSError, HTTPException):
            return self._unreachable(requested, port, t0)
        if resp.status != 200:
            data = resp.read()
            conn.close()
            status = resp.status
            size = self._reply(status, anthropic_error(error_kind(status), upstream_error(data)))
            return self._access(t0, status, requested, port, size)
        if not body.get("stream"):
            data = resp.read()
            conn.close()
            try:
                reply = json.loads(data)
            except ValueError:
                message = "the model server returned a response that is not JSON"
                size = self._reply(502, anthropic_error("api_error", message))
                return self._access(t0, 502, requested, port, size)
            size = self._reply(200, chat_to_anthropic(reply, requested))
            return self._access(t0, 200, requested, port, size)

        self._begin_stream()
        stream, size, status = MessageStream(requested), 0, 200
        failure, gone = None, False
        try:
            for line in resp:
                event = parse_upstream_line(line)
                if event is None or event[0] == "done":
                    continue
                try:
                    for text in stream.feed(*event):
                        self.wfile.write(text.encode())
                        size += len(text)
                except OSError:
                    gone = True     # the client left: stop reading the model
                    break
        except (OSError, HTTPException):
            failure = "the model server stopped responding mid-stream"
        conn.close()
        if not gone:
            tail = (stream.finish() if failure is None
                    else [sse({"type": "error",
                               "error": {"type": "api_error", "message": failure}})])
            try:
                for text in tail:
                    self.wfile.write(text.encode())
                    size += len(text)
            except OSError:
                gone = True
        if gone:
            status = 499
        elif failure:
            status = 502
        self._access(t0, status, requested, port, size, stream=True)


class ProxyServer(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True

    def __init__(self, port: int, routes: dict, api_key: str,
                 upstream_timeout: float, access):
        """`routes` comes from build_routes(); `access(line)` receives one
        access-log line per request."""
        super().__init__(("127.0.0.1", port), ProxyHandler)
        self.routes = routes
        self.main_port = routes[LOCAL_MODEL][0]
        self.api_key, self.upstream_timeout = api_key, upstream_timeout
        self.started = int(time.time())
        self.access = access
