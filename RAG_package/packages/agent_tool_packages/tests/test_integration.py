"""Integration test for Agent Tool Package.

Tests:
    1. Interface contracts — all tools implement required properties
    2. Registry — tools auto-register via decorator and can be exported
    3. LangChain integration — tools convert to StructuredTool correctly
    4. Template Method — ServiceTool.execute() flow works with mocked infra
    5. Deep Agent integration — all tools can be loaded into create_deep_agent
"""

import asyncio
import importlib
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _reload_all_tools():
    """Clear registry and reload all tool modules to re-trigger @register."""
    from agent_tools.core.registry import ServiceToolRegistry
    ServiceToolRegistry.clear()
    import agent_tools.general.weather_tool
    import agent_tools.general.time_tool
    import agent_tools.general.web_search_tool
    import agent_tools.general.knowledge_base_tool
    import agent_tools.image.blur_tool
    import agent_tools.image.ocr_receipt_invoice_tool
    import agent_tools.image.ocr_driver_license_tool
    import agent_tools.image.upload_tool
    importlib.reload(agent_tools.general.weather_tool)
    importlib.reload(agent_tools.general.time_tool)
    importlib.reload(agent_tools.general.web_search_tool)
    importlib.reload(agent_tools.general.knowledge_base_tool)
    importlib.reload(agent_tools.image.blur_tool)
    importlib.reload(agent_tools.image.ocr_receipt_invoice_tool)
    importlib.reload(agent_tools.image.ocr_driver_license_tool)
    importlib.reload(agent_tools.image.upload_tool)


# ═══════════════════════════════════════════════════════════════════════
# Test 1: Interface Contracts
# ═══════════════════════════════════════════════════════════════════════


class TestInterfaceContracts:
    """Verify all tools implement AbstractBaseTool contract."""

    def setup_method(self):
        """Fresh registry for each test."""
        _reload_all_tools()

    def test_all_tools_have_name(self):
        from agent_tools.core.registry import ServiceToolRegistry

        for name, tool in ServiceToolRegistry._tools.items():
            assert isinstance(tool.name, str)
            assert len(tool.name) > 0
            assert tool.name == name

    def test_all_tools_have_description(self):
        from agent_tools.core.registry import ServiceToolRegistry

        for _, tool in ServiceToolRegistry._tools.items():
            assert isinstance(tool.description, str)
            assert len(tool.description) > 10  # meaningful description

    def test_all_tools_have_input_schema(self):
        from pydantic import BaseModel

        from agent_tools.core.registry import ServiceToolRegistry

        for _, tool in ServiceToolRegistry._tools.items():
            assert issubclass(tool.input_schema, BaseModel)

    def test_all_tools_have_execute(self):
        from agent_tools.core.registry import ServiceToolRegistry

        for _, tool in ServiceToolRegistry._tools.items():
            assert callable(tool.execute)

    def test_all_tools_have_as_langchain_tool(self):
        from agent_tools.core.registry import ServiceToolRegistry

        for _, tool in ServiceToolRegistry._tools.items():
            assert callable(tool.as_langchain_tool)


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Registry
# ═══════════════════════════════════════════════════════════════════════


class TestRegistry:
    """Verify ServiceToolRegistry auto-collects and exports tools."""

    def setup_method(self):
        from agent_tools.core.registry import ServiceToolRegistry
        ServiceToolRegistry.clear()

    def test_register_decorator(self):
        from agent_tools.abs_interface.utility_tool import AbstractUtilityTool
        from agent_tools.core.registry import ServiceToolRegistry
        from pydantic import BaseModel

        class DummyInput(BaseModel):
            x: str = "test"

        @ServiceToolRegistry.register
        class DummyTool(AbstractUtilityTool):
            @property
            def name(self):
                return "dummy"

            @property
            def description(self):
                return "A dummy tool for testing"

            @property
            def input_schema(self):
                return DummyInput

            async def call_api(self, **kwargs):
                return {"result": "ok"}

        assert "dummy" in ServiceToolRegistry._tools
        assert ServiceToolRegistry.get_tool("dummy") is not None

    def test_register_raw(self):
        from agent_tools.core.registry import ServiceToolRegistry

        mock_tool = MagicMock()
        mock_tool.name = "mcp_test_tool"
        ServiceToolRegistry.register_raw(mock_tool)
        assert len(ServiceToolRegistry._raw_tools) == 1

    def test_get_all_langchain_tools(self):
        from agent_tools.core.registry import ServiceToolRegistry

        _reload_all_tools()

        tools = ServiceToolRegistry.get_all_langchain_tools()
        assert len(tools) > 0

        # Verify they're all LangChain tools
        from langchain_core.tools import BaseTool

        for tool in tools:
            assert isinstance(tool, BaseTool)

    def test_expected_tools_registered(self):
        from agent_tools.core.registry import ServiceToolRegistry

        _reload_all_tools()

        expected = {
            "get_weather",
            "get_time",
            "search_web",
            "search_knowledge_base",
            "blur_image",
            "ocr_receipt_invoice",
            "ocr_driver_license",
            "upload_image",
        }
        registered = set(ServiceToolRegistry._tools.keys())
        assert expected == registered, f"Missing: {expected - registered}, Extra: {registered - expected}"

    def test_list_tools(self):
        from agent_tools.core.registry import ServiceToolRegistry

        _reload_all_tools()

        listing = ServiceToolRegistry.list_tools()
        assert len(listing) >= 4
        for item in listing:
            assert "name" in item
            assert "description" in item
            assert "type" in item


# ═══════════════════════════════════════════════════════════════════════
# Test 3: LangChain Integration
# ═══════════════════════════════════════════════════════════════════════


class TestLangChainIntegration:
    """Verify tools convert to valid LangChain StructuredTools."""

    def setup_method(self):
        _reload_all_tools()

    def test_conversion(self):
        from agent_tools.core.registry import ServiceToolRegistry

        tools = ServiceToolRegistry.get_all_langchain_tools()
        for tool in tools:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "args_schema")
            assert tool.name
            assert tool.description

    def test_tool_schemas_valid(self):
        """Verify input schemas produce valid JSON Schema."""
        from agent_tools.core.registry import ServiceToolRegistry

        tools = ServiceToolRegistry.get_all_langchain_tools()
        for tool in tools:
            schema = tool.args_schema.model_json_schema()
            assert "properties" in schema
            assert "type" in schema


# ═══════════════════════════════════════════════════════════════════════
# Test 4: Template Method — ServiceTool.execute()
# ═══════════════════════════════════════════════════════════════════════


class TestServiceToolExecute:
    """Verify the Template Method flow of AbstractServiceTool.execute()."""

    @pytest.mark.asyncio
    async def test_fire_and_forget(self):
        """FIRE_AND_FORGET strategy returns immediately after dispatch."""
        from agent_tools.image.upload_tool import UploadImageTool

        tool = UploadImageTool()

        mock_producer = MagicMock()
        mock_s3 = MagicMock()

        with (
            patch("agent_tools.core.clients.get_producer", return_value=mock_producer),
            patch("agent_tools.core.clients.get_s3_client", return_value=mock_s3),
            patch("agent_tools.core.clients.get_s3_bucket", return_value="test-bucket"),
            patch("requests.get") as mock_requests_get,
        ):
            mock_response = MagicMock()
            mock_response.content = b"fake image bytes"
            mock_response.raise_for_status = MagicMock()
            mock_requests_get.return_value = mock_response

            result = await tool.execute(image_url="http://example.com/test.jpg")

            assert result["status"] == "uploaded"
            assert "image_id" in result
            assert "s3_path" in result
            mock_producer.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_poll_success(self):
        """REDIS_POLL strategy dispatches and polls Redis."""
        from agent_tools.image.blur_tool import BlurTool

        tool = BlurTool()

        mock_producer = MagicMock()
        redis_result = {"s3_blurred_path": "blur_processed/test.jpeg"}

        with (
            patch.dict(os.environ, {"FASTAPI_BLUR_WEBHOOK_URL": "http://test/webhook"}),
            patch("agent_tools.core.clients.get_producer", return_value=mock_producer),
            patch(
                "agent_tools.core.polling.poll_redis",
                new_callable=AsyncMock,
                return_value=redis_result,
            ),
        ):
            result = await tool.execute(image_id="test-id", s3_path="img_upload/test-id")

            assert result["status"] == "completed"
            assert result["s3_blurred_path"] == "blur_processed/test.jpeg"
            mock_producer.send.assert_called_once()

            # Verify payload contains injected request_id and webhook
            call_kwargs = mock_producer.send.call_args
            payload = call_kwargs[1]["value"] if "value" in call_kwargs[1] else call_kwargs[0][0]
            assert "request_id" in payload
            assert "webhook" in payload

    @pytest.mark.asyncio
    async def test_mongodb_poll_success(self):
        """MONGODB_POLL strategy dispatches and polls MongoDB."""
        from agent_tools.image.ocr_receipt_invoice_tool import OcrReceiptInvoiceTool

        tool = OcrReceiptInvoiceTool()

        mock_producer = MagicMock()
        mongo_doc = {
            "image_id": "test-id",
            "ocr_result": {
                "re_inv_ocr_status": "processed",
                "re_inv_ocr_path": "ocr_processed/test.json",
            },
        }
        mock_s3 = MagicMock()
        mock_s3.download_fileobj = MagicMock(
            side_effect=lambda bucket, key, stream: stream.write(b'{"merchant": "Test Shop"}')
        )

        with (
            patch("agent_tools.core.clients.get_producer", return_value=mock_producer),
            patch(
                "agent_tools.core.polling.poll_mongodb",
                new_callable=AsyncMock,
                return_value=mongo_doc,
            ),
            patch("agent_tools.core.clients.get_s3_client", return_value=mock_s3),
            patch("agent_tools.core.clients.get_s3_bucket", return_value="test-bucket"),
        ):
            result = await tool.execute(
                image_id="test-id",
                s3_path="img_upload/test-id",
                language="Japanese",
            )

            assert result["status"] == "completed"
            assert result["image_id"] == "test-id"
            mock_producer.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_poll_timeout(self):
        """REDIS_POLL strategy raises ToolTimeoutError on timeout."""
        from agent_tools.core.polling import ToolTimeoutError
        from agent_tools.image.blur_tool import BlurTool

        tool = BlurTool()
        mock_producer = MagicMock()

        with (
            patch("agent_tools.core.clients.get_producer", return_value=mock_producer),
            patch(
                "agent_tools.core.polling.poll_redis",
                new_callable=AsyncMock,
                side_effect=ToolTimeoutError("REDIS_POLL", 60, "test-request-id"),
            ),
        ):
            with pytest.raises(ToolTimeoutError):
                await tool.execute(image_id="test-id", s3_path="img_upload/test-id")


# ═══════════════════════════════════════════════════════════════════════
# Test 5: Utility Tools
# ═══════════════════════════════════════════════════════════════════════


class TestUtilityTools:
    """Verify utility tools call APIs correctly."""

    @pytest.mark.asyncio
    async def test_weather_tool(self):
        from agent_tools.general.weather_tool import WeatherTool

        tool = WeatherTool()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "location": {"name": "Tokyo", "country": "Japan"},
            "current": {
                "temp_c": 22.0,
                "condition": {"text": "Sunny"},
                "humidity": 45,
                "wind_kph": 12.0,
                "feelslike_c": 21.0,
            },
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch.dict(os.environ, {"WEATHER_API_KEY": "test-key"}),
            patch("requests.get", return_value=mock_response),
        ):
            result = await tool.execute(city="Tokyo")
            assert result["city"] == "Tokyo"
            assert result["temp_c"] == 22.0
            assert result["condition"] == "Sunny"

    @pytest.mark.asyncio
    async def test_time_tool(self):
        from agent_tools.general.time_tool import GetTimeTool

        tool = GetTimeTool()
        result = await tool.execute(timezone_str="UTC")
        assert "datetime" in result
        assert result["timezone"] == "UTC"
        assert "day_of_week" in result

    @pytest.mark.asyncio
    async def test_time_tool_invalid_tz(self):
        from agent_tools.general.time_tool import GetTimeTool

        tool = GetTimeTool()
        result = await tool.execute(timezone_str="Invalid/Zone")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_weather_no_api_key(self):
        from agent_tools.general.weather_tool import WeatherTool

        tool = WeatherTool()
        with patch.dict(os.environ, {"WEATHER_API_KEY": ""}, clear=False):
            result = await tool.execute(city="Tokyo")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_web_search_no_api_key(self):
        from agent_tools.general.web_search_tool import WebSearchTool

        tool = WebSearchTool()
        with patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=False):
            result = await tool.execute(query="test")
            assert "error" in result


# ═══════════════════════════════════════════════════════════════════════
# Test 6: Full Integration — load all tools into agent
# ═══════════════════════════════════════════════════════════════════════


class TestFullIntegration:
    """Simulate loading all tools the way Deep Agent would."""

    def setup_method(self):
        _reload_all_tools()

    def test_load_all_tools_for_agent(self):
        """Verify all tools can be loaded and are compatible with agent frameworks."""
        from agent_tools.core.registry import ServiceToolRegistry

        all_tools = ServiceToolRegistry.get_all_langchain_tools()

        # Should have 8 tools total (4 general + 4 image)
        assert len(all_tools) == 8, f"Expected 8 tools, got {len(all_tools)}: {[t.name for t in all_tools]}"

        # All tools should have unique names
        names = [t.name for t in all_tools]
        assert len(names) == len(set(names)), f"Duplicate tool names: {names}"

        # All tools should be invocable (have coroutine)
        for tool in all_tools:
            assert tool.coroutine is not None or tool.func is not None, f"Tool {tool.name} not callable"

        print("\n✅ All tools loaded successfully:")
        for tool in all_tools:
            print(f"  - {tool.name}: {tool.description[:60]}...")

    def test_tool_names_for_llm(self):
        """Verify tool names are LLM-friendly (no spaces, lowercase)."""
        from agent_tools.core.registry import ServiceToolRegistry

        for name, tool in ServiceToolRegistry._tools.items():
            assert " " not in name, f"Tool name '{name}' has spaces"
            assert name == name.lower() or "_" in name, f"Tool name '{name}' not snake_case"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
