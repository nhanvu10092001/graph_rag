import json
import os
import threading
import time

import requests


class _Function:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id: str, function: _Function):
        self.id = id
        self.type = "function"
        self.function = function


class _Message:
    def __init__(self, content: str, tool_calls: list[_ToolCall], reasoning_content: str | None):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class _RateLimiter:
    """Sliding-window rate limiter that allows concurrent bursts up to RPM."""

    def __init__(self, rpm: int):
        self._rpm = rpm
        self._lock = threading.Lock()
        self._timestamps: list[float] = []

    def wait(self):
        while True:
            with self._lock:
                now = time.time()
                self._timestamps = [t for t in self._timestamps if now - t < 60.0]
                if len(self._timestamps) < self._rpm:
                    self._timestamps.append(now)
                    return
                sleep_until = self._timestamps[0] + 60.0
            time.sleep(max(0, sleep_until - time.time()) + 0.05)


class AnthropicProxyModel:
    def __init__(
        self,
        name: str,
        base_url: str = "http://localhost:8080",
        rpm: int = 10,
        max_workers: int = 3,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.max_workers = max_workers
        self._limiter = _RateLimiter(rpm)

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        out = []
        for t in tools:
            func = t["function"]
            out.append({
                "name": func["name"],
                "description": func["description"],
                "input_schema": func["parameters"],
            })
        return out

    @staticmethod
    def _convert_tool_choice(tc: str) -> dict:
        if tc == "required":
            return {"type": "any"}
        return {"type": "auto"}

    @staticmethod
    def _convert_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
        system_prompt = None
        anthropic_msgs: list[dict] = []

        for msg in messages:
            role = msg["role"]

            if role == "system":
                system_prompt = msg["content"]

            elif role == "user":
                anthropic_msgs.append({"role": "user", "content": msg["content"]})

            elif role == "assistant":
                blocks: list[dict] = []
                text = msg.get("content")
                if text:
                    blocks.append({"type": "text", "text": text})
                for tc in msg.get("tool_calls", []):
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": args,
                    })
                if not blocks:
                    blocks.append({"type": "text", "text": ""})
                anthropic_msgs.append({"role": "assistant", "content": blocks})

            elif role == "tool":
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": msg["tool_call_id"],
                    "content": msg["content"],
                }
                if (
                    anthropic_msgs
                    and anthropic_msgs[-1]["role"] == "user"
                    and isinstance(anthropic_msgs[-1]["content"], list)
                ):
                    anthropic_msgs[-1]["content"].append(tool_result)
                else:
                    anthropic_msgs.append({"role": "user", "content": [tool_result]})

        return system_prompt, anthropic_msgs

    @staticmethod
    def _parse_response(data: dict) -> _Message:
        content_text = ""
        tool_calls: list[_ToolCall] = []

        for block in data.get("content", []):
            if block["type"] == "text":
                content_text += block.get("text", "")
            elif block["type"] == "tool_use":
                args = block.get("input", {})
                tool_calls.append(
                    _ToolCall(
                        id=block["id"],
                        function=_Function(
                            name=block["name"],
                            arguments=json.dumps(args),
                        ),
                    )
                )

        return _Message(
            content=content_text or None,
            tool_calls=tool_calls,
            reasoning_content=None,
        )

    def forward(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        enable_thinking: bool = False,
    ) -> _Message:
        system_prompt, anthropic_msgs = self._convert_messages(messages)

        body: dict = {
            "model": self.name,
            "max_tokens": 16384,
            "messages": anthropic_msgs,
            "thinking": {"type": "disabled"},
        }
        if system_prompt:
            body["system"] = system_prompt
        if tools:
            body["tools"] = self._convert_tools(tools)
            body["tool_choice"] = self._convert_tool_choice(tool_choice)

        self._limiter.wait()

        resp = requests.post(
            f"{self.base_url}/v1/messages",
            json=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": "none",
                "anthropic-version": "2023-06-01",
            },
            timeout=120,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"Proxy error {resp.status_code}: {resp.text}")

        return self._parse_response(resp.json())
