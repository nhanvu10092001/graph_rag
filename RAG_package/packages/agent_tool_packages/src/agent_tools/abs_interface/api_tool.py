"""AbstractApiTool — Template Method for tools that call a FastAPI backend.

Design Patterns:
    - Template Method: execute() defines the fixed flow; subclass overrides
      build_request(), parse_response(), and optionally poll_result().
    - Strategy: Subclass decides whether the API is sync (returns result immediately)
      or async (returns job ID → need to poll a GET endpoint for result).

This replaces AbstractServiceTool for tools that delegate to an existing
FastAPI service (e.g. image_app) instead of directly publishing to MQ.

Flow:
    Agent calls tool
      → execute()
          1. build_request()     → HTTP method, path, body/files
          2. POST to FastAPI     → dispatch the job
          3. poll_result()       → optionally poll GET endpoint until done
          4. parse_response()    → transform final response for agent
"""

import asyncio
import os
from abc import abstractmethod
from typing import Any, Literal

import httpx
from pydantic import BaseModel

from .base_tool import AbstractBaseTool


class AbstractApiTool(AbstractBaseTool):
    """Base class for tools that call a FastAPI backend to execute work.

    Subclass contract:
        MUST override:
            - name, description, input_schema  (from AbstractBaseTool)
            - api_path: str                    (FastAPI endpoint path, e.g. "/img_blur/image_blur/")
            - http_method: str                 ("POST" | "GET")
            - build_request(**kwargs)          → dict with "json" and/or "files"
            - parse_response(data) -> Any      → transform API response for agent

        MAY override:
            - poll_endpoint: str | None        (GET path template for polling, e.g. "/img_blur/get_blur_image/{image_id}")
            - poll_timeout: int                (seconds, default 60)
            - poll_interval: float             (seconds, default 2.0)
            - poll_result(dispatch_data, **kwargs) → poll GET endpoint until done
            - is_multipart: bool               (True for file upload endpoints)

        DO NOT override:
            - execute(**kwargs)                (template method)
    """

    # ── Abstract config (subclass MUST define) ──────────────────────────

    @property
    @abstractmethod
    def api_path(self) -> str:
        """FastAPI endpoint path (e.g. '/img_blur/image_blur/')."""
        raise NotImplementedError

    @property
    def http_method(self) -> Literal["POST", "GET"]:
        """HTTP method for the dispatch request. Default POST."""
        return "POST"

    # ── Abstract methods (subclass MUST implement) ──────────────────────

    @abstractmethod
    def build_request(self, **kwargs) -> dict:
        """Build the HTTP request payload.

        Returns a dict that may contain:
            - "json": dict       → for JSON body requests
            - "files": dict      → for multipart file uploads
            - "data": dict       → for form data
            - "params": dict     → for query parameters

        Do NOT include base_url or path — those come from api_path.
        """
        raise NotImplementedError

    @abstractmethod
    def parse_response(self, data: Any) -> Any:
        """Transform the final API response into the tool's output for the agent.

        Args:
            data: Parsed JSON response from the API (after polling if applicable).

        Returns:
            Structured result dict for the agent.
        """
        raise NotImplementedError

    # ── Optional config (subclass MAY override) ─────────────────────────

    @property
    def base_url(self) -> str:
        """Base URL of the FastAPI service. Reads from IMAGE_APP_BASE_URL env var."""
        return os.getenv("IMAGE_APP_BASE_URL", "http://localhost:8000")

    @property
    def poll_endpoint(self) -> str | None:
        """GET endpoint template for polling results.

        Use {image_id} as placeholder, e.g. "/img_blur/get_blur_image/{image_id}"
        If None, the dispatch response is used directly (no polling).
        """
        return None

    @property
    def poll_timeout(self) -> int:
        """Maximum seconds to poll for result. Override for slower tasks."""
        return 60

    @property
    def poll_interval(self) -> float:
        """Seconds between poll attempts."""
        return 2.0

    @property
    def is_multipart(self) -> bool:
        """Whether build_request returns files for multipart upload."""
        return False

    # ── Template Method (DO NOT override) ───────────────────────────────

    async def execute(self, **kwargs) -> Any:
        """Execute the full dispatch → (optional poll) → parse flow.

        This is the Template Method. Do NOT override this.

        Flow:
            1. build_request()    → construct HTTP payload
            2. POST/GET to API    → dispatch the job
            3. poll_result()      → if poll_endpoint set, poll until done
            4. parse_response()   → transform result for agent
        """
        # 1. Build request
        request_kwargs = self.build_request(**kwargs)
        url = f"{self.base_url}{self.api_path}"

        # 2. Dispatch to FastAPI
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            if self.http_method == "POST":
                response = await client.post(url, **request_kwargs)
            else:
                response = await client.get(url, **request_kwargs)

            if response.status_code >= 400:
                error_text = self._safe_text(response)
                return {
                    "status": "error",
                    "error": f"API returned {response.status_code}: {error_text}",
                }

            dispatch_data = self._safe_parse(response)

            # 3. Poll for result if poll_endpoint is configured
            if self.poll_endpoint:
                result_data = await self._poll_result(client, dispatch_data, **kwargs)
                return self.parse_response(result_data)

            # 4. No polling — use dispatch response directly
            return self.parse_response(dispatch_data)

    async def _poll_result(
        self, client: httpx.AsyncClient, dispatch_data: dict, **kwargs
    ) -> Any:
        """Poll the GET endpoint until a successful result is returned.

        Override this if you need custom polling logic.
        """
        # Extract image_id from dispatch response or kwargs
        image_id = ""
        if isinstance(dispatch_data, dict):
            image_id = dispatch_data.get("image_id", "")
        if not image_id:
            image_id = kwargs.get("image_id", "")
        poll_url = f"{self.base_url}{self.poll_endpoint.format(image_id=image_id)}"

        iterations = int(self.poll_timeout / self.poll_interval)
        for _ in range(iterations):
            await asyncio.sleep(self.poll_interval)
            resp = await client.get(poll_url)
            if resp.status_code == 200:
                return self._safe_parse(resp)
            # 400 = "not processed yet", keep polling
            # 404 = "not found", keep polling briefly in case of race condition

        return {
            "status": "timeout",
            "message": f"Polling timed out after {self.poll_timeout}s for image_id={image_id}",
        }

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _safe_parse(response: httpx.Response) -> Any:
        """Safely parse an httpx response, handling binary/JSON/text."""
        content_type = response.headers.get("content-type", "")

        if "application/json" in content_type:
            try:
                return response.json()
            except Exception:
                pass

        if "image/" in content_type or "octet-stream" in content_type:
            return {
                "status": "completed",
                "content_type": content_type,
                "content_length": len(response.content),
                "message": "Image processed successfully (binary response received).",
            }

        # Unknown content-type — try JSON, then text
        try:
            return response.json()
        except Exception:
            pass
        try:
            return {"raw_text": response.text}
        except Exception:
            return {"status": "completed", "message": "Response received (binary, non-decodable)."}

    @staticmethod
    def _safe_text(response: httpx.Response) -> str:
        """Safely get text from response, handling binary."""
        try:
            return response.text
        except Exception:
            return f"(binary response, {len(response.content)} bytes)"
