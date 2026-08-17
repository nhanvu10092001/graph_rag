"""AbstractBaseTool — Root interface for all agent tools.

Design Patterns:
    - Adapter Pattern: as_langchain_tool() converts internal interface → LangChain StructuredTool
    - Registry Pattern: subclasses use @ServiceToolRegistry.register decorator

Mirrors the role of ConsumerInterface in mq_packages — defines the contract
that all tool types must satisfy.
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel


class AbstractBaseTool(ABC):
    """Base interface for ALL agent tools.

    Every tool — whether it dispatches to MQ, queries a DB, calls an API,
    or wraps an MCP server — must implement this interface.

    Subclasses:
        - AbstractServiceTool: dispatch via MQ, poll result
        - AbstractQueryTool: query DB directly
        - AbstractUtilityTool: call external API directly
        - MCPToolBridge: wrap MCP server tools
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name. LLM uses this to identify the tool."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description. LLM reads this to decide when to call the tool.

        Write clear, specific descriptions. Examples:
            Good: "Blur faces and license plates in an image. Input: image_id from upload_image."
            Bad:  "Process image."
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def input_schema(self) -> type[BaseModel]:
        """Pydantic model defining tool input parameters.

        Example:
            class BlurInput(BaseModel):
                image_id: str = Field(description="UUID of the uploaded image")
                s3_path: str = Field(description="S3 object key of the image")
        """
        raise NotImplementedError

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool and return result.

        This is the main entry point called by the agent runtime.
        Each subclass implements a different execution strategy.
        """
        raise NotImplementedError

    def as_langchain_tool(self) -> StructuredTool:
        """Adapter: convert this tool into a LangChain StructuredTool.

        This allows any AbstractBaseTool to be registered with Deep Agent
        or any LangChain-compatible agent framework.
        """
        return StructuredTool.from_function(
            coroutine=self._execute_from_model,
            name=self.name,
            description=self.description,
            args_schema=self.input_schema,
        )

    async def _execute_from_model(self, **kwargs) -> Any:
        """Bridge between LangChain's call convention and our execute()."""
        return await self.execute(**kwargs)
