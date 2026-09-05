import threading
import time

from litellm import completion


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


class OpenAICompatibleModel:
    def __init__(
        self,
        name: str,
        base_url: str = "http://localhost:8000/v1",
        rpm: int | None = None,
        max_workers: int = 3,
    ) -> None:
        self.name = name
        self.base_url = base_url
        self.max_workers = max_workers
        self._limiter = _RateLimiter(rpm) if rpm else None

    def forward(self, messages, tools=[], tool_choice="auto", enable_thinking=False):
        if self._limiter:
            self._limiter.wait()

        response = completion(
            model=f"openai/{self.name}",
            api_base=self.base_url,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            api_key="_",
            timeout=120,
        )
        message = response["choices"][0]["message"]

        if not hasattr(message, "reasoning_content"):
            message.reasoning_content = None

        return message
