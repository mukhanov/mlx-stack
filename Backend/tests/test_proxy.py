import json
import socket
import threading
import unittest
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from local_llm import proxy

KEY = "test-key"
MAIN, SMALL = "org/main-model", "org/small-model"
ANTHROPIC = {"anthropic-version": "2023-06-01"}
USAGE = {"prompt_tokens": 21, "completion_tokens": 3, "total_tokens": 24,
         "prompt_tokens_details": {"cached_tokens": 5}}


def chunk(delta, finish=None):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def parsed(sse_texts):
    """[(event type, data dict)] for a list of SSE strings."""
    out = []
    for text in sse_texts:
        event, data = text.splitlines()[:2]
        out.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def types(sse_texts):
    return [kind for kind, _ in parsed(sse_texts)]


class RoutingTests(unittest.TestCase):
    def test_aliases_resolve_to_the_model_each_server_loads(self):
        routes = proxy.build_routes(MAIN, 19797, SMALL, 8081)
        self.assertEqual(list(routes), [MAIN, "local", "small", SMALL])
        self.assertEqual(routes["local"], (19797, MAIN))
        self.assertEqual(routes[MAIN], (19797, MAIN))
        self.assertEqual(routes["small"], (8081, SMALL))
        self.assertEqual(routes[SMALL], (8081, SMALL))

    def test_without_a_helper_there_is_no_small_route(self):
        routes = proxy.build_routes(MAIN, 19797, SMALL, 0)
        self.assertEqual(list(routes), [MAIN, "local"])
        self.assertNotIn("small", routes)


class RequestTranslationTests(unittest.TestCase):
    def body(self, **override):
        body = {
            "model": "local", "max_tokens": 64, "temperature": 0.2, "top_k": 5,
            "system": [{"type": "text", "text": "Be brief.",
                        "cache_control": {"type": "ephemeral"}}],
            "stop_sequences": ["END"], "stream": True,
            "metadata": {"user_id": "u"}, "thinking": {"type": "adaptive"},
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Weather in Paris?"},
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/jpeg",
                                                 "data": "AAA"}}]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "…", "signature": "sig"},
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                     "input": {"city": "Paris"}}]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "only thinking"}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1",
                     "content": [{"type": "text", "text": "18C"}]},
                    {"type": "tool_result", "tool_use_id": "toolu_2",
                     "content": "boom", "is_error": True},
                    {"type": "text", "text": "Thanks"}]},
            ],
            "tools": [
                {"name": "get_weather", "description": "Weather by city",
                 "input_schema": {"type": "object"}},
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 1},
            ],
            "tool_choice": {"type": "tool", "name": "get_weather"},
        }
        body.update(override)
        return body

    def test_messages_system_and_sampling_are_carried_over(self):
        chat = proxy.anthropic_to_chat(self.body(), MAIN)
        self.assertEqual(chat["model"], MAIN)   # the server's name, not the alias
        self.assertEqual(chat["messages"][0], {"role": "system", "content": "Be brief."})
        self.assertEqual(chat["messages"][1], {"role": "user", "content": [
            {"type": "text", "text": "Weather in Paris?"},
            {"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64,AAA"}}]})
        self.assertEqual(chat["messages"][2], {
            "role": "assistant", "content": "Let me check.",
            "tool_calls": [{"id": "toolu_1", "type": "function",
                            "function": {"name": "get_weather",
                                         "arguments": '{"city": "Paris"}'}}]})
        self.assertEqual(chat["messages"][3:5], [
            {"role": "tool", "tool_call_id": "toolu_1", "content": "18C"},
            {"role": "tool", "tool_call_id": "toolu_2", "content": "Error: boom"}])
        self.assertEqual(chat["messages"][5:], [{"role": "user", "content": "Thanks"}])
        for key, value in (("max_tokens", 64), ("temperature", 0.2), ("top_k", 5),
                           ("stop", ["END"]), ("stream", True),
                           ("stream_options", {"include_usage": True})):
            self.assertEqual(chat[key], value)
        for absent in ("metadata", "thinking"):
            self.assertNotIn(absent, chat)

    def test_tools_become_openai_functions_and_tool_choice_is_mapped(self):
        chat = proxy.anthropic_to_chat(self.body(), MAIN)
        self.assertEqual(chat["tools"], [{"type": "function", "function": {
            "name": "get_weather", "description": "Weather by city",
            "parameters": {"type": "object"}}}])
        self.assertEqual(chat["tool_choice"], {"type": "function",
                                               "function": {"name": "get_weather"}})
        for kind, expected in (({"type": "any"}, "required"), ({"type": "auto"}, "auto"),
                               ({"type": "none"}, "none"), (None, "auto")):
            chat = proxy.anthropic_to_chat(self.body(tool_choice=kind), MAIN)
            self.assertEqual(chat["tool_choice"], expected)
        bare = proxy.anthropic_to_chat(self.body(tools=[]), MAIN)
        self.assertNotIn("tools", bare)
        self.assertNotIn("tool_choice", bare)

    def test_plain_text_turns_and_non_stream_requests(self):
        chat = proxy.anthropic_to_chat(self.body(
            stream=False, system="sys", stop_sequences=None,
            messages=[{"role": "user", "content": "hi"},
                      {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}]),
            MAIN)
        self.assertEqual(chat["messages"], [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "tool_calls": []}])
        self.assertNotIn("stream", chat)
        self.assertNotIn("stream_options", chat)
        self.assertNotIn("stop", chat)


class ResponseTranslationTests(unittest.TestCase):
    def test_text_reasoning_and_tool_calls_map_to_anthropic_blocks(self):
        resp = {"id": "chatcmpl-abc", "model": MAIN, "choices": [{
            "index": 0, "finish_reason": "tool_calls",
            "message": {"role": "assistant", "content": "Checking.",
                         "reasoning": "think",
                         "tool_calls": [{"id": "call_1", "type": "function",
                                         "function": {"name": "get_weather",
                                                      "arguments": '{"city": "Paris"}'}}]}}],
            "usage": USAGE}
        out = proxy.chat_to_anthropic(resp, "local")
        self.assertEqual(out["id"], "msg_abc")
        self.assertEqual(out["type"], "message")
        self.assertEqual(out["model"], "local")   # the name the client asked for
        self.assertEqual(out["stop_reason"], "tool_use")
        self.assertEqual([b["type"] for b in out["content"]], ["thinking", "text", "tool_use"])
        self.assertEqual(out["content"][0], {"type": "thinking", "thinking": "think",
                                             "signature": proxy.THINKING_SIGNATURE})
        self.assertEqual(out["content"][2], {"type": "tool_use", "id": "call_1",
                                             "name": "get_weather",
                                             "input": {"city": "Paris"}})
        # Anthropic's input_tokens exclude the cached ones; the sum is the prompt
        self.assertEqual(out["usage"], {"input_tokens": 16, "output_tokens": 3,
                                        "cache_creation_input_tokens": 0,
                                        "cache_read_input_tokens": 5})

    def test_stop_reasons_bad_arguments_and_empty_answers(self):
        def reply(finish, **message):
            return {"id": "chatcmpl-x", "choices": [{"index": 0, "finish_reason": finish,
                                                     "message": {"role": "assistant", **message}}]}
        def stop_reason(finish, **message):
            return proxy.chat_to_anthropic(reply(finish, **message), "m")["stop_reason"]

        self.assertEqual(stop_reason("length", content="cut"), "max_tokens")
        self.assertEqual(stop_reason("stop", content="ok"), "end_turn")
        broken = reply("tool_calls", tool_calls=[
            {"id": "c", "type": "function", "function": {"name": "t", "arguments": '{"city":'}}])
        self.assertEqual(proxy.chat_to_anthropic(broken, "m")["content"][0]["input"], {})
        empty = proxy.chat_to_anthropic(reply("stop", content=None), "m")
        self.assertEqual(empty["content"], [{"type": "text", "text": ""}])
        self.assertEqual(empty["usage"], {"output_tokens": 0})


class StreamTranslationTests(unittest.TestCase):
    def setUp(self):
        self.stream = proxy.MessageStream("local")

    def test_mlx_stream_text_thinking_and_whole_tool_call(self):
        out = self.stream.feed("keepalive", 4321)
        out += self.stream.feed("chunk", chunk({"reasoning": "hmm"}))
        out += self.stream.feed("chunk", chunk({"content": "Hello"}))
        out += self.stream.feed("chunk", chunk({"content": " there"}))
        out += self.stream.feed("chunk", chunk({"tool_calls": [
            {"index": 0, "id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}]}))
        out += self.stream.feed("chunk", chunk({}, "tool_calls"))
        out += self.stream.feed("chunk", {"choices": [], "usage": USAGE})
        out += self.stream.finish()
        events = parsed(out)
        self.assertEqual(types(out), [
            "message_start", "ping",
            "content_block_start", "content_block_delta",
            "content_block_delta", "content_block_stop",
            "content_block_start", "content_block_delta",
            "content_block_delta", "content_block_stop",
            "content_block_start", "content_block_delta", "content_block_stop",
            "message_delta", "message_stop"])
        self.assertEqual(events[0][1]["message"]["model"], "local")
        self.assertEqual(events[0][1]["message"]["usage"]["input_tokens"], 4321)
        self.assertEqual(events[2][1]["content_block"], {"type": "thinking", "thinking": ""})
        self.assertEqual(events[3][1]["delta"], {"type": "thinking_delta", "thinking": "hmm"})
        self.assertEqual(events[4][1]["delta"]["type"], "signature_delta")
        self.assertEqual(events[6][1]["content_block"], {"type": "text", "text": ""})
        self.assertEqual("".join(e[1]["delta"]["text"] for e in events[7:9]), "Hello there")
        self.assertEqual(events[10][1]["content_block"], {
            "type": "tool_use", "id": "call_1", "name": "get_weather", "input": {}})
        self.assertEqual(events[11][1]["delta"], {"type": "input_json_delta",
                                                  "partial_json": '{"city":"Paris"}'})
        indexes = [e[1]["index"] for e in events if e[0] == "content_block_start"]
        self.assertEqual(indexes, [0, 1, 2])
        for opened in indexes:   # every block is closed, after its own deltas
            stops = [i for i, e in enumerate(events)
                     if e[0] == "content_block_stop" and e[1]["index"] == opened]
            self.assertEqual(len(stops), 1)
        self.assertEqual(events[-2][1], {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"input_tokens": 16, "output_tokens": 3,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 5}})

    def test_fragmented_tool_calls_are_reassembled(self):
        first = {"index": 0, "id": "call_a", "type": "function",
                 "function": {"name": "read", "arguments": '{"pa'}}
        second = {"index": 1, "id": "call_b", "type": "function",
                  "function": {"name": "grep", "arguments": '{"q":"x"}'}}
        tail_a = 'th":"a.txt"}'          # completes first: {"path":"a.txt"}
        for call in (first, second,
                     {"index": 0, "function": {"arguments": tail_a}},
                     {"index": 1, "function": {"arguments": ""}}):
            self.stream.feed("chunk", chunk({"tool_calls": [call]}))
        out = self.stream.finish()
        tools = [(e["content_block"]["name"], e["content_block"]["id"], e["content_block"]["input"])
                 for kind, e in parsed(out) if kind == "content_block_start"
                 and e["content_block"]["type"] == "tool_use"]
        self.assertEqual(tools, [("read", "call_a", {}), ("grep", "call_b", {})])
        deltas = [e["delta"]["partial_json"] for kind, e in parsed(out)
                  if kind == "content_block_delta"]
        self.assertEqual(deltas, ['{"path":"a.txt"}', '{"q":"x"}'])
        self.assertEqual(parsed(out)[-2][1]["delta"]["stop_reason"], "end_turn")
        self.assertEqual(parsed(out)[-2][1]["usage"], {"output_tokens": 4})

    def test_an_empty_stream_is_still_a_well_formed_message(self):
        out = self.stream.finish()
        self.assertEqual(types(out), ["message_start", "content_block_start",
                                      "content_block_stop", "message_delta", "message_stop"])
        self.assertEqual(parsed(out)[0][1]["message"]["usage"]["input_tokens"], 0)
        self.assertEqual(parsed(out)[-2][1]["delta"]["stop_reason"], "end_turn")

    def test_upstream_lines(self):
        self.assertEqual(proxy.parse_upstream_line(b": keepalive 64/512\n\n"), ("keepalive", 512))
        self.assertEqual(proxy.parse_upstream_line(b"data: [DONE]\n\n"), ("done", None))
        self.assertEqual(proxy.parse_upstream_line(b'data: {"choices": []}\n\n'),
                         ("chunk", {"choices": []}))
        self.assertIsNone(proxy.parse_upstream_line(b": keepalive oops\n"))
        self.assertIsNone(proxy.parse_upstream_line(b"data: not json\n"))
        self.assertIsNone(proxy.parse_upstream_line(b"\n"))


class _FakeModel(BaseHTTPRequestHandler):
    """A model server that behaves like mlx_lm.server: it answers for the
    one model it was given and refuses any other name."""
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.server.requests.append({"path": self.path, "raw": raw})
        body = json.loads(raw)
        if body.get("model") != self.server.model:
            return self._json(404, {"error": "Cannot find an appropriate cached snapshot"
                                            f" folder for {body.get('model')}"})
        text = str(body["messages"][-1].get("content"))
        if "FAIL400" in text:
            return self._json(400, {"error": "context too long: 9 tokens"})
        tool = None
        if body.get("tools"):
            tool = {"id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}
        finish = "tool_calls" if tool else "stop"
        base = {"id": "chatcmpl-1", "model": body["model"]}
        if body.get("stream"):
            first = chunk({"tool_calls": [dict(tool, index=0)]} if tool else {"content": "Hello"})
            pieces = [chunk({"role": "assistant"}), first, chunk({}, finish)]
            if (body.get("stream_options") or {}).get("include_usage"):
                pieces.append({"choices": [], "usage": USAGE})
            out = ": keepalive 64/512\n\n"
            for piece in pieces:
                data = {**base, "object": "chat.completion.chunk", **piece}
                out += "data: " + json.dumps(data) + "\n\n"
            out += "data: [DONE]\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            return self.wfile.write(out.encode())
        message = {"role": "assistant", "content": None if tool else "Hello from backend"}
        if tool:
            message["tool_calls"] = [tool]
        self._json(200, {**base, "object": "chat.completion",
                         "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                         "usage": USAGE})

    def _json(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class ProxyServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main, cls.small = cls._serve(MAIN), cls._serve(SMALL)
        cls.lines = []
        cls.proxy = proxy.ProxyServer(
            0, proxy.build_routes(MAIN, cls.main.server_address[1],
                                  SMALL, cls.small.server_address[1]),
            KEY, 5, cls.lines.append)
        threading.Thread(target=cls.proxy.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        for server in (cls.proxy, cls.main, cls.small):
            server.shutdown()
            server.server_close()

    @staticmethod
    def _serve(model):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeModel)
        server.model, server.requests, server.daemon_threads = model, [], True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def call(self, method, path, body=None, key=KEY, headers=None):
        head = {"Content-Type": "application/json"}
        if key is not None:
            head["Authorization"] = f"Bearer {key}"
        head.update(headers or {})
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        conn = HTTPConnection("127.0.0.1", self.proxy.server_address[1], timeout=10)
        conn.request(method, path, body=body, headers=head)
        resp = conn.getresponse()
        data, head = resp.read(), dict(resp.getheaders())
        conn.close()
        return resp.status, head, data

    def chat(self, text, model=MAIN, **extra):
        return {"model": model, "max_tokens": 32,
                "messages": [{"role": "user", "content": text}], **extra}

    def test_requests_naming_the_model_itself_are_relayed_untouched(self):
        raw = ('{ "model": "%s", "top_k": 40, "min_p": 0.05, "user": "u",'
               ' "messages": [{"role": "user", "content": "hi"}] }' % MAIN).encode()
        status, head, data = self.call("POST", "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        self.assertEqual(self.main.requests[-1]["raw"], raw)
        self.assertEqual(self.main.requests[-1]["path"], "/v1/chat/completions")
        self.assertEqual(json.loads(data)["choices"][0]["message"]["content"], "Hello from backend")
        self.assertEqual(head["Content-Type"], "application/json")

    def test_aliases_are_replaced_by_the_model_each_server_loads(self):
        before = len(self.main.requests), len(self.small.requests)
        for model in ("local", "small"):
            status, _, _ = self.call("POST", "/v1/chat/completions", self.chat("hi", model))
            self.assertEqual(status, 200)
        self.assertEqual((len(self.main.requests), len(self.small.requests)),
                         (before[0] + 1, before[1] + 1))
        # everything else about the request is left as it was
        self.assertEqual(json.loads(self.main.requests[-1]["raw"]),
                         self.chat("hi", MAIN))
        self.assertEqual(json.loads(self.small.requests[-1]["raw"]),
                         self.chat("hi", SMALL))
        self.assertIn("local ->%d 200" % self.main.server_address[1], self.lines[-2])
        self.assertIn("small ->%d 200" % self.small.server_address[1], self.lines[-1])

    def test_any_other_model_is_refused_before_it_reaches_a_model_server(self):
        before = len(self.main.requests), len(self.small.requests)
        status, _, data = self.call("POST", "/v1/chat/completions", self.chat("hi", "org/other"))
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(data)["error"]["code"], "model_not_found")
        status, _, data = self.call("POST", "/v1/messages", self.chat("hi", "org/other"),
                                    headers=ANTHROPIC)
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(data)["error"]["type"], "not_found_error")
        status, _, _ = self.call("POST", "/v1/chat/completions",
                                 self.chat("hi", ["not", "a", "name"]))
        self.assertEqual(status, 404)
        self.assertEqual((len(self.main.requests), len(self.small.requests)), before)

    def test_streams_are_relayed_byte_for_byte(self):
        status, head, data = self.call("POST", "/v1/chat/completions",
                                       self.chat("hi", stream=True))
        self.assertEqual(status, 200)
        self.assertTrue(head["Content-Type"].startswith("text/event-stream"))
        self.assertTrue(data.startswith(b": keepalive 64/512\n\ndata: "))
        self.assertTrue(data.endswith(b"data: [DONE]\n\n"))
        self.assertNotIn(b"input_json_delta", data)   # OpenAI wire format, not Anthropic

    def test_the_static_key_is_required_except_for_health(self):
        status, _, data = self.call("POST", "/v1/chat/completions", self.chat("hi"), key="wrong")
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(data)["error"]["message"], "Invalid API key")
        status, _, _ = self.call("POST", "/v1/chat/completions", self.chat("hi"),
                                 key=None, headers={"x-api-key": KEY})
        self.assertEqual(status, 200)
        self.assertEqual(self.call("GET", "/health/liveliness", key=None)[0], 200)

    def test_anthropic_messages_become_chat_completions_and_come_back_as_a_message(self):
        body = {"model": "small", "max_tokens": 32, "system": "Be brief.",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"name": "get_weather", "description": "w",
                           "input_schema": {"type": "object"}}]}
        status, _, data = self.call("POST", "/v1/messages", body, headers=ANTHROPIC)
        self.assertEqual(status, 200)
        sent = json.loads(self.small.requests[-1]["raw"])
        self.assertEqual(self.small.requests[-1]["path"], "/v1/chat/completions")
        self.assertEqual(sent["model"], SMALL)
        self.assertEqual(sent["messages"][0], {"role": "system", "content": "Be brief."})
        self.assertEqual(sent["tools"][0]["function"]["name"], "get_weather")
        self.assertNotIn("stream", sent)
        reply = json.loads(data)
        self.assertEqual(reply["model"], "small")
        self.assertEqual(reply["stop_reason"], "tool_use")
        self.assertEqual(reply["content"][0]["input"], {"city": "Paris"})
        self.assertEqual(reply["usage"]["cache_read_input_tokens"], 5)

    def test_anthropic_streams_end_in_a_complete_message(self):
        body = {"model": "local", "max_tokens": 32, "stream": True,
                "messages": [{"role": "user", "content": "hi"}]}
        status, head, data = self.call("POST", "/v1/messages", body,
                                       headers={**ANTHROPIC, "x-api-key": KEY})
        self.assertEqual(status, 200)
        self.assertTrue(head["Content-Type"].startswith("text/event-stream"))
        sent = json.loads(self.main.requests[-1]["raw"])
        self.assertEqual(sent["model"], MAIN)
        self.assertEqual(sent["stream_options"], {"include_usage": True})
        events = parsed(data.decode().split("\n\n")[:-1])
        self.assertEqual([kind for kind, _ in events], [
            "message_start", "ping", "content_block_start", "content_block_delta",
            "content_block_stop", "message_delta", "message_stop"])
        self.assertEqual(events[0][1]["message"]["model"], "local")
        self.assertEqual(events[0][1]["message"]["usage"]["input_tokens"], 512)
        self.assertEqual(events[3][1]["delta"], {"type": "text_delta", "text": "Hello"})
        self.assertEqual(events[-2][1]["delta"]["stop_reason"], "end_turn")
        self.assertEqual(events[-2][1]["usage"]["output_tokens"], 3)

    def test_model_server_errors_keep_their_message_in_anthropic_shape(self):
        body = {"model": "local", "max_tokens": 8,
                "messages": [{"role": "user", "content": "FAIL400"}]}
        status, _, data = self.call("POST", "/v1/messages", body, headers=ANTHROPIC)
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(data), {"type": "error", "error": {
            "type": "invalid_request_error", "message": "context too long: 9 tokens"}})

    def test_an_unreachable_model_server_is_reported_not_hidden(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            dead = probe.getsockname()[1]   # released on close: nothing listens there
        lines = []
        orphan = proxy.ProxyServer(0, proxy.build_routes("org/dead", dead), KEY, 2, lines.append)
        threading.Thread(target=orphan.serve_forever, daemon=True).start()

        def post(path, body, headers):
            conn = HTTPConnection("127.0.0.1", orphan.server_address[1], timeout=10)
            conn.request("POST", path, body=json.dumps(body).encode(), headers={
                "Content-Type": "application/json", **headers})
            resp = conn.getresponse()
            out = resp.status, json.loads(resp.read())
            conn.close()
            return out

        try:
            status, data = post("/v1/messages", {"model": "local", "max_tokens": 8,
                                                 "messages": [{"role": "user", "content": "hi"}]},
                                {"x-api-key": KEY, **ANTHROPIC})
            self.assertEqual(status, 503)
            self.assertEqual(data["error"]["type"], "overloaded_error")
            self.assertIn(str(dead), data["error"]["message"])
            status, data = post("/v1/chat/completions", self.chat("hi", "local"),
                                {"Authorization": f"Bearer {KEY}"})
            self.assertEqual(status, 503)
            self.assertEqual(data["error"]["type"], "server_error")
            self.assertEqual([line.split()[-4] for line in lines], ["503", "503"])
            self.assertTrue(all(line.endswith("ERROR") for line in lines))
        finally:
            orphan.shutdown()
            orphan.server_close()

    def test_the_model_list_names_the_stack_models_in_each_clients_shape(self):
        _, _, data = self.call("GET", "/v1/models")
        listing = json.loads(data)
        self.assertEqual(listing["object"], "list")
        self.assertEqual([m["id"] for m in listing["data"]], [MAIN, "local", "small", SMALL])
        _, _, data = self.call("GET", "/v1/models", headers=ANTHROPIC)
        listing = json.loads(data)
        self.assertEqual([m["id"] for m in listing["data"]], [MAIN, "local", "small", SMALL])
        self.assertEqual((listing["first_id"], listing["last_id"], listing["has_more"]),
                         (MAIN, SMALL, False))


if __name__ == "__main__":
    unittest.main()
